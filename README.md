# SweepSol — the anchored scalar-drive reader

**Explore mode, 26 November 2026.** An independent companion to Claude's Sweep. The idea is to look through a frozen conceptual map without immediately moving the anchor, and discover which direction to look from scalar feedback rather than being handed a target vector.

## Run the workbench

Clone the repository and open `index.html` directly in Edge or Chrome. No web server, no JS packages, no GPU needed for the browser. The initial **synthetic illustrative ring** is hand-authored; it is not an LLM experiment.

```powershell
git clone https://github.com/anttiluode/SweepSol
cd SweepSol
start index.html
```

Use **Step** to observe left/right pairs, **Run** for continuous scanning, and uncheck automatic commit to hold the anchor still while probing. Width, side offset, sampling rate, coverage and commit threshold are independent controls. Choose a fixed PCA band or Adaptive. In Pursuit mode type a target and Apply; the evaluator sees that target but the walker receives only scalar feedback for the anchor and sampled candidates. Change the target while running to inspect retargeting. Export the trace as JSON.

## Real frozen-model map (RTX 3060)

Requires Python, PyTorch with CUDA and transformers. Use your existing environment if it already runs Theory. The export is a one-time GPU forward pass over batches; no training, generation or activation writeback.

```powershell
python3.13 export_bank.py --device cuda --words 8000 --pcs 128 --out bank.json
```

Then load bank.json in the page. For a quick plumbing run use `--words 256 --pcs 32`. If the model is already cached and you want to prohibit downloads use `--local-files-only`. Default model Qwen/Qwen3-1.7B, float16, middle decoder block; `--layer` and `--dtype` are adjustable. Explicit demo words are included when they are single tokens in the item frame; other candidates come from the vocabulary in token-ID order. If a target is missing, try another word or use `--include` with a single-token alternative. The exporter saves only the leading PCA coordinates; "all cached PCs" does **not** mean the original full residual stream. PCA is fitted on the exported pool, not on a held-out set.

Optional headless runner (Node.js, no npm dependencies):

```powershell
node run.js --self-test
node run.js demo oak sauna 100 guitar
node run.js bank.json oak sauna 100 guitar
node run.js bank.json oak - 100
```

The fifth positional argument switches the target halfway through. Each run writes trace.json. The browser and CLI use the same engine.

## What the engine actually does

- **Anchor / slow state:** current word and the last committed displacement (heading).
- **Probe / fast state:** a direction that can rotate from scalar feedback without moving the anchor. A local side axis is derived from candidate displacements orthogonal to that direction; a deterministic alternator selects plus/minus side offsets.
- **Grain:** different leading or offset PCA bands rank local candidates; the adaptive reader tries each available band and then uses a simple per-band utility estimate with periodic exploration. The search angle is measured in a local plane in the common cached coordinate space, not a true high-dimensional cone.
- **Coverage:** each probed word increments a decaying trace; the trace reduces the score of future revisits. Default decay 0.04 per beat, not per second.
- **Scalar drive:** an evaluator returns noisy (optional) cosine warmth in [0,1] for sampled words. The walker has no target ID or vector. Feedback from a pair updates the fast axis; automatic commit requires a sampled improvement over the anchor in pursuit mode. The forage utility instead combines neighborhood similarity and recency. The target can change without changing the bank.
- **Commit:** a separate action after a two-probe pair. Manual commit allows a human to take the best sampled candidate even when it fails the automatic threshold. Empty sectors are logged, not silently replaced with global searches.

This is a conceptual port, not a line-for-line import of KyberDyyni1. No claim of biological fidelity, LLM reasoning improvement, emergent alternation, or superiority over ordinary search is made. The clock is an adjustable browser timer, not the original ECG homeostat. Layer is stored in the bank metadata; live layer switching is not implemented. The map is a local two-dimensional view of candidate displacements, not a global semantic map.

## Harvest lineage

- [KyberDyyni1 / fork conclusion](https://github.com/anttiluode/KyberDyyni1/blob/main/FORK_CONCLUSION.md): stable reference, fast control vector, structured low-travel alternation; do not hard-wire the biological width/frequency correlation.
- [Theory / ocular.py](https://github.com/anttiluode/Theory/blob/main/ocular.py): frozen hidden-state bank and PCA bands.
- [Theory / common.py](https://github.com/anttiluode/Theory/blob/main/common.py): item-word prompt frame.
- [ConjunctionReservoir](https://github.com/anttiluode/ConjunctionReservoir): inspiration for decaying coverage memory.
- Vollan et al., adaptive theta sweeps, Science DOI 10.1126/science.aef4184 (biological inspiration, not a validated equivalence).

## Status and receipts

The initial engine self-test was executed on the synthetic ring during development and passed six workbench checks: alternation, held anchor, coverage decay, different grain dimensions, scalar-only steering with at most three oracle calls for initialization and a pair, and unchanged bank. This is wiring sanity, not a scientific result. The later minor trace and grain-scheduling changes, browser UI and real Qwen export still require an end-to-end run. No actual LLM result has been collected. No competitive kill gates.

The important question for a later experiment is whether this kind of reader can discover useful relationships under an independently defined reward, rather than merely climb a cosine surface in a map that already contains the answer. Retargeting on a hidden cosine goal is an initial workbench for that question, not its solution.


## Swarm — many readers, one frozen map

**New:** open [swarm.html](swarm.html) directly in Edge/Chrome after pulling, and load your existing bank.json. No new Qwen pass. Choose Forage, Random hidden goals (learn roads), Common hidden goal (learn roads), or Common hidden goal (frozen roads). Up to 16 simultaneous SweepSol walkers, each with its own anchor, fast axis, grain and private coverage. Shared coverage is updated after each round, so every active walker sees the same previous-round marks. Shared coverage and road strength are separate knobs; set both to zero for independent readers. New colonies reset coverage but retain roads. The viewer plots the first two normalized cached PC coordinates, which are only a projection; the actual search uses higher-dimensional cached vectors.

Roads are directed committed edges, credited only after an exact target has become the anchor. Cycles are erased from the credited path (not the work count). Successful paths leave a slow memory; no model weights or cached vectors are modified. Each reader sees only scalar cosine warmth for its anchor and selected probes, and uses the existing alternating sector and commit policy. There is no hard-coded semantic word list in the controller. Shared observations and interior nodes of successful paths are logged separately, not assumed to be true semantic bridges.

### Matched road transfer experiment (Node.js)

```powershell
node swarm.js --self-test
node swarm.js --bank demo --train 8 --test 8 --lanes 4 --beats 80
node swarm.js --bank bank.json --train 32 --test 32 --lanes 4 --beats 160 --horizon 32 --out swarm-qwen.json
```

Training pairs are sampled reproducibly; the same unordered endpoint pair cannot occur in both training and test. Endpoints and intermediate words can recur across the split. Training learns roads from successful paths only. The test runs exactly the same ordered pairs, seeds, shared coverage, local horizon and beat budget twice, with road preference zero and with the learned roads enabled. The road table is frozen in both test arms. Read success counts, scalar calls for all trials, budgeted beats, and paired road-only / baseline-only successes together (successful-only means are selection-biased). The run records all paths and probe traces, the split, and the complete road table. The local horizon is per grain, not a fixed global graph. Neighbour discovery still scans the bank on cache misses, so scalar calls are not a total compute or latency measure.

This is exploratory; it is not a claim of improvement, nor a test of independent task rewards. Random pairs may be disconnected or hard, and a sparse global road table may generalize poorly or distract the walker. The small synthetic self-test (4 train/4 test pairs) returned 2 training successes and 3/4 successes in each test arm with identical paired outcomes. The eight wiring checks passed, but real Qwen swarm results and browser end-to-end behavior are not yet measured here. The older single-walker defaults are unchanged; the swarm opts into common-space forage utility to avoid cross-band raw cosine comparisons.


## Temperature update and the map-before-reader result

See our independent companion repository **[Sweep](https://github.com/anttiluode/Sweep)**, particularly [fields.py and its Qwen report](https://github.com/anttiluode/Sweep/tree/main/runs/fields_qwen_h20) and [the Pythia checkpoint experiment](https://github.com/anttiluode/Sweep/tree/main/runs/develop_pythia-410m_h20).

SweepSol now exposes base temperature in both browser pages. Zero gives cold commitment; a nonzero base temperature allows observed downhill reward steps by Metropolis-style acceptance, heats when progress stalls, and cools on positive commits. Forage additionally uses temperature to sample candidate scores (Gumbel-max) and adjusts heat based on recent anchor repetition. The optional avoid-previous-anchors switch is independent. The swarm runner accepts `--temperature 0.01` and `--avoid-revisit 1`; use `--temperature 0` for a cold comparison. Each trace records temperature, reward delta, downhill commits and eligible candidate count. The controller uses scalar feedback, not target coordinates.

Also changed: forage utility defaults to a common full cached-space cosine, so raw narrow-band cosine is no longer used to compare band utilities; negative probe scores cannot automatically trigger forage commitment; the side axis uses centered candidate residuals before falling back to the old construction. This last change is an exploratory mitigation, not a proven fix for right-side blindness. The sector still may be empty. Original pre-temperature implementation is preserved in git history.

```powershell
git pull
start index.html
start swarm.html
node swarm.js --self-test
node swarm.js --bank bank.json --train 32 --test 32 --temperature 0.01 --out swarm-qwen-warm.json
node swarm.js --bank bank.json --train 32 --test 32 --temperature 0 --out swarm-qwen-cold.json
```

Sweep's independent checkpoint results distinguish geometric navigability (already high at random initialization) from mature-map consistency (final-map step cosine rises from 0.003 at step0 to 0.270 at step512, 0.369 at step1000, and 0.484 at step143000). These numbers are from Sweep, not a reproduction by SweepSol. Final-map cosine is a geometric proxy, not a language-model judgement of the meaning of an individual step. The same 300 pairs were selected using the final map; successful temperature paths only were scored.

Updated engine and swarm synthetic checks were run against the committed sources; both browser inline scripts and CLI parsed. Real-bank performance and browser end-to-end behavior of the update remain unmeasured. This is an independent port of the acceptance principle, not an exact reproduction of Sweep's Python algorithm.
