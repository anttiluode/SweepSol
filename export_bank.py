"""Export real frozen-model word states for the offline SweepSol workbench.

The item-word frame mirrors Theory/common.py (shots mode). We capture one
decoder block's output, never generate tokens, and fit a PCA basis on the
whole exported vocabulary. Only the leading PCs are exported, not full states.
"""
import argparse
import hashlib
import json
from pathlib import Path

SHOTS = ("Item: rose. Is it a flower? yes\n"
         "Item: lion. Is it a fish? no\n"
         "Item: chair. Is it a tool? no\n"
         "Item: wolf. Is it an animal? yes\n"
         "Item: cup. Is it a cup? yes\n"
         "Item: lion. Is it a tiger? no\n")
EXTRA = "oak birch pine maple forest lumber table chair wood firewood fire smoke sauna steam water lake boat fish salmon guitar violin music piano sound metal steel hammer axe tool saw cabbage lettuce carrot onion garden soil bread oven kitchen heat stone rock mountain snow ice winter spring summer".split()


def find_layers(model):
    for path in ("model.layers", "gpt_neox.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for name in path.split("."):
                obj = getattr(obj, name)
            if len(obj):
                return obj
        except (AttributeError, TypeError):
            pass
    raise ValueError("Unsupported decoder layout")


def encode_word(tok, word):
    prefix = SHOTS + "Item: "
    text = prefix + word + ". Is it?"
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = len(prefix), len(prefix) + len(word)
    hit = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    if len(hit) != 1 or enc["offset_mapping"][hit[0]][1] > end:
        return None
    return enc["input_ids"], hit[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    p.add_argument("--words", type=int, default=8000)
    p.add_argument("--pcs", type=int, default=128)
    p.add_argument("--layer", type=int, default=None, help="zero-based decoder block; default middle")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--include", default="", help="comma-separated words to include if single-token")
    p.add_argument("--out", type=Path, default=Path("bank.json"))
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args()
    if a.words < 32 or a.pcs < 2 or a.batch < 1:
        p.error("Need words >= 32, pcs >= 2, batch >= 1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(41)
    kw = dict(revision=a.revision, local_files_only=a.local_files_only)
    tok = AutoTokenizer.from_pretrained(a.model, **kw)
    dtype = getattr(torch, a.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, **kw)
    model = model.to(a.device).eval()
    layers = find_layers(model)
    layer = len(layers) // 2 if a.layer is None else a.layer
    if not 0 <= layer < len(layers):
        p.error("Layer out of range")
    candidates = EXTRA + [w.strip() for w in a.include.split(",") if w.strip()]
    candidates += [t[1:] for t, _ in sorted(tok.get_vocab().items(), key=lambda z: z[1])
                   if t[:1] in ("Ġ", "▁")]
    words, encoded, seen = [], [], set()
    for w in candidates:
        if not w.isalpha() or len(w) < 3 or not (w.islower() or w.istitle()) or w in seen:
            continue
        item = encode_word(tok, w)
        if item is None:
            continue
        seen.add(w)
        words.append(w)
        encoded.append(item)
        if len(words) >= a.words:
            break
    if len(words) < 32:
        raise RuntimeError("Too few eligible vocabulary entries")
    print(f"{len(words)} single-token words, block {layer}, {a.dtype}", flush=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    captured, rows = {}, []
    def hook(mod, inp, out):
        captured["state"] = out[0] if isinstance(out, (tuple, list)) else out
    handle = layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for s in range(0, len(words), a.batch):
                batch = encoded[s:s+a.batch]
                n = max(len(ids) for ids, _ in batch)
                x = torch.full((len(batch), n), pad, dtype=torch.long, device=a.device)
                mask = torch.zeros_like(x)
                for i, (ids, pos) in enumerate(batch):
                    x[i, :len(ids)] = torch.tensor(ids, device=a.device)
                    mask[i, :len(ids)] = 1
                positions = torch.tensor([pos for _, pos in batch], device=a.device)
                captured.clear()
                model.base_model(input_ids=x, attention_mask=mask, use_cache=False)
                state = captured.pop("state")
                rows.append(state[torch.arange(len(batch), device=a.device), positions].float().cpu())
                del state
                if s == 0 or s // a.batch % 25 == 0:
                    print(f"Captured {min(s+a.batch,len(words))}/{len(words)}", flush=True)
    finally:
        handle.remove()
    names = {type(m).__name__ for m in model.modules()}
    centered = not any("RMSNorm" in n for n in names)
    resolved = getattr(model.config, "_commit_hash", None)
    del model, layers
    if a.device.startswith("cuda"):
        torch.cuda.empty_cache()
    X = torch.cat(rows)
    del rows
    if centered:
        X -= X.mean(1, keepdim=True)
    X -= X.mean(0, keepdim=True)
    if not torch.isfinite(X).all():
        raise ValueError("Nonfinite hidden states")
    k = min(a.pcs, X.shape[0]-1, X.shape[1])
    q = min(k+16, X.shape[0], X.shape[1])
    print(f"Fitting {k} PCs on CPU (randomized PCA, seed 41)", flush=True)
    _, singular, V = torch.pca_lowrank(X, q=q, center=False, niter=4)
    Y = X @ V[:, :k]
    keep = Y.norm(dim=1) > 1e-9
    if not keep.all():
        words = [w for w, ok in zip(words, keep.tolist()) if ok]
        Y = Y[keep]
    if len(words) < 4:
        raise ValueError("Degenerate projection")
    fraction = float(Y.square().sum()/X.square().sum())
    data = dict(words=words, vectors=Y.tolist(), meta=dict(
        source=a.model, requested_revision=a.revision, resolved_revision=resolved,
        layer=layer, dtype=a.dtype, frame="Theory shots / item token / decoder block output",
        pca="randomized, CPU, seed 41, niter 4", pcs=k, retained_energy=fraction,
        row_centered=centered, explicit_demo_words=EXTRA, extra_include=a.include,
        word_order_sha256=hashlib.sha256("\n".join(words).encode()).hexdigest()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix+".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",",":"), allow_nan=False), encoding="utf-8")
    tmp.replace(a.out)
    print(f"Saved {a.out} ({len(words)} words; retained energy {fraction:.3f})")


if __name__ == "__main__":
    main()
