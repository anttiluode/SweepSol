/* SweepSol: independent scalar-drive workbench. */
const Sweep = (function SweepCore(){
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const norm=a=>Math.sqrt(dot(a,a));
  const scale=(a,s)=>a.map(x=>x*s);
  const add=(a,b)=>a.map((x,i)=>x+b[i]);
  const sub=(a,b)=>a.map((x,i)=>x-b[i]);
  const unit=a=>{const n=norm(a);return n>1e-10?scale(a,1/n):a.map(()=>0)};
  const clip=(x,a,b)=>Math.max(a,Math.min(b,x));
  const rnd=seed=>()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296};
  const angle=(a,b)=>Math.acos(clip(dot(a,b),-1,1))*180/Math.PI;
  function prepare(raw){
    if(!Array.isArray(raw.words)||!Array.isArray(raw.vectors)||raw.words.length!==raw.vectors.length||raw.words.length<4)throw Error("Need at least four words and matching vectors");
    const d=raw.vectors[0].length;
    if(d<2||new Set(raw.words).size!==raw.words.length||raw.vectors.some(v=>v.length!==d||v.some(x=>!Number.isFinite(x))||norm(v)<1e-10))throw Error("Invalid or duplicate bank rows");
    const x=raw.vectors.map(unit);
    const bands=[{name:"all cached PCs",start:0,end:d}];
    for(const [s,e] of [[0,16],[4,20],[8,24],[0,64]])if(e<=d&&!(s===0&&e===d))bands.push({name:`PC ${s+1}–${e}`,start:s,end:e});
    return {words:raw.words,x,d,bands,views:bands.map(b=>x.map(v=>unit(v.slice(b.start,b.end)))),meta:raw.meta||{}};
  }
  function demo(){
    const r=rnd(73), words=["oak","birch","pine","maple","forest","lumber","table","chair","wood","firewood","fire","smoke","sauna","steam","water","lake","boat","fish","salmon","guitar","violin","music","piano","sound","metal","steel","hammer","axe","tool","saw","cabbage","lettuce","carrot","onion","garden","soil","bread","oven","kitchen","heat","stone","rock","mountain","snow","ice","winter","spring","summer"];
    // Hand-authored curved toy manifold; NOT language-model data.
    const vectors=words.map((w,i)=>{const t=i*2*Math.PI/words.length;return Array.from({length:32},(_,j)=>j===0?Math.cos(t):j===1?Math.sin(t):j===2?.4*Math.cos(3*t):j===3?.4*Math.sin(3*t):.04*(r()-.5)+.12*Math.sin(t*(1+j%4)+j))});
    return {words,vectors,meta:{source:"Synthetic illustrative ring (not LLM evidence)",layer:null}};
  }
  function oracle(bank,target,noise=0,seed=7){
    let t=target, calls=0;const r=rnd(seed);
    return {read(i){calls++;return clip((1+dot(bank.x[i],bank.x[t]))/2+noise*(2*r()-1),0,1)}, retarget(i){t=i},count(){return calls},truth(i){return (1+dot(bank.x[i],bank.x[t]))/2}};
  }
  class Walker{
    constructor(bank,anchor,opts={}){
      this.b=bank;this.a=anchor;this.r=rnd(opts.seed??5);this.cover=new Float64Array(bank.words.length);
      this.heading=null;this.fast=null;this.side=1;this.pair=[];this.t=0;this.stall=0;this.visited=new Set();this.trace=[];
      this.band=0;this.bandStats=bank.bands.map(()=>({n:0,v:0}));this.reward=null;this.anchorReward=null;
      this.opts={width:60,offset:30,neighbors:64,decay:.04,coverage:.35,learn:.7,threshold:.015,grain:"auto",autoCommit:true,...opts};
    }
    setReward(fn){this.reward=fn;this.anchorReward=fn?fn(this.a):null;this.pair=[];this.side=1;this.lastCandidate=null;this.fast=this.heading?this.heading.slice():null;this.stall=0;}
    chooseBand(){
      if(this.opts.grain!=="auto")return +this.opts.grain;
      const k=Math.floor(this.t/2),s=this.bandStats;
      if(k<s.length)return k;
      if(k%4===0)return Math.floor(k/4)%s.length;
      return s.reduce((best,v,i)=>v.v+.08/Math.sqrt(v.n+1)>s[best].v+.08/Math.sqrt(s[best].n+1)?i:best,0);
    }
    neighborhood(band){
      const v=this.b.views[band],a=this.a, x=this.b.x;
      return v.map((p,i)=>({i,sim:dot(v[a],p)})).filter(p=>p.i!==a).sort((a,b)=>b.sim-a.sim).slice(0,Math.min(this.opts.neighbors,x.length-1)).map(p=>({...p,u:unit(sub(x[p.i],x[a]))}));
    }
    frame(cands){
      if(!this.heading||norm(this.heading)<.5)this.heading=cands[0].u.slice();
      if(!this.fast||norm(this.fast)<.5)this.fast=this.heading.slice();
      let axis=unit(this.fast);
      // On repeated empty/unsuccessful pairs explore a different local direction.
      if(this.stall>=3&&this.pair.length===0){axis=cands[Math.floor(this.r()*cands.length)].u.slice();this.fast=axis.slice();this.stall=0;}
      let side=null,best=0;
      for(const c of cands){const u=sub(c.u,scale(axis,dot(c.u,axis))),n=norm(u);if(n>best){best=n;side=unit(u)}}
      if(!side||best<1e-8){const basis=axis.map((_,j)=>j===axis.reduce((m,x,i)=>Math.abs(x)<Math.abs(axis[m])?i:m,0)?1:0);side=unit(sub(basis,scale(axis,dot(axis,basis))))}
      return {axis,side};
    }
    commit(i,r){
      if(i===this.a)return false;
      this.heading=unit(sub(this.b.x[i],this.b.x[this.a]));
      this.a=i;this.fast=unit(add(scale(this.fast||this.heading,.65),scale(this.heading,.35)));
      this.anchorReward=this.reward?r:null;this.stall=0;return true;
    }
    step(){
      const o=this.opts;
      for(let i=0;i<this.cover.length;i++)this.cover[i]*=1-o.decay;
      if(!this.pair.length){this.band=this.chooseBand();this.cands=this.neighborhood(this.band);this.plane=this.frame(this.cands)}
      const {axis,side}=this.plane,sgn=this.side;this.side=-this.side;
      const theta=sgn*o.offset*Math.PI/180;
      // The probe is a sector in the local 2D plane, not a cone in 2048D.
      let hit=null,best=-Infinity;
      for(const c of this.cands){
        const h=dot(c.u,axis),v=dot(c.u,side),length=Math.hypot(h,v);
        if(length<1e-8)continue;
        const delta=Math.atan2(Math.sin(Math.atan2(v,h)-theta),Math.cos(Math.atan2(v,h)-theta));
        if(Math.abs(delta)>o.width*Math.PI/180)continue;
        const score=.55*c.sim+.30*length+.15*Math.cos(delta)-o.coverage*this.cover[c.i]+(this.scoreBonus?this.scoreBonus(this.a,c.i):0);
        if(score>best){best=score;hit={...c,score,delta,prior:this.cover[c.i]}}
      }
      const old=this.a, rec={beat:this.t++,anchor:old,side:sgn,band:this.band,probe:hit?.i??null,score:hit?.score??null,reward:null,commit:false,after:old,axisOffset:angle(this.heading,axis),width:o.width};
      if(hit){this.cover[hit.i]+=1;this.visited.add(hit.i);rec.reward=this.reward?this.reward(hit.i):null;hit.reward=rec.reward;}
      this.pair.push(hit);
      if(this.pair.length===2){
        const hits=this.pair.filter(Boolean);
        let utility=0;
        if(hits.length){
          // Reward is observed only for chosen probes, never for all candidates.
          if(this.reward){
            let grad=axis.map(()=>0);
            for(const h of hits)grad=add(grad,scale(h.u,h.reward-this.anchorReward));
            if(hits.length===2)grad=add(grad,scale(sub(hits[0].u,hits[1].u),hits[0].reward-hits[1].reward));
            if(norm(grad)>1e-9)this.fast=unit(add(axis,scale(grad,o.learn*3)));
            hits.sort((a,b)=>b.reward-a.reward);
            utility=hits[0].reward-this.anchorReward;
          }else{
            const value=h=>(o.commonUtility?dot(this.b.x[this.a],this.b.x[h.i]):h.sim)-.3*h.prior;
            hits.sort((a,b)=>value(b)-value(a));
            utility=value(hits[0]);
          }
          const h=hits[0], eligible=this.reward?utility>o.threshold:utility>o.threshold;
          if(o.autoCommit&&eligible){rec.commit=this.commit(h.i,h.reward);rec.commitProbe=h.i;}
          if(!rec.commit)this.stall++;
          this.lastCandidate=h;
        }else {this.stall++;this.lastCandidate=null;}
        const s=this.bandStats[this.band];s.v=s.n?.8*s.v+.2*utility:utility;s.n++;
        this.pair=[];this.side=1;
      }
      rec.after=this.a;this.trace.push(rec);return rec;
    }
    manualCommit(){
      if(!this.lastCandidate||this.pair.length)return false;
      const old=this.a,i=this.lastCandidate.i;
      const ok=this.commit(i,this.lastCandidate.reward);
      if(ok)this.trace.push({event:"manual_commit",beat:this.t,anchor:old,after:i});
      this.lastCandidate=null;return ok;
    }
  }
  function selfTest(){
    const b=prepare(demo()),w=new Walker(b,0,{autoCommit:false,seed:3,grain:0,width:180});
    const fixed=b.x.map(v=>v.slice()),a=w.step(),c=w.step();
    if(a.side!==1||c.side!==-1||w.a!==0)throw Error("anchor/parity");
    if(a.probe===null||w.cover[a.probe]<=0)throw Error("coverage");
    const probe=a.probe, before=w.cover[probe];w.opts.width=0;w.step();if(w.cover[probe]>=before)throw Error("decay");
    const v0=b.views[0][0],v1=b.views[1][0];if(v0.length===v1.length)throw Error("grain");
    const p=new Walker(b,0,{autoCommit:false,width:180,grain:0,seed:3}),o=oracle(b,9);p.setReward(i=>o.read(i));
    p.step();p.step();if(p.a!==0||angle(p.heading,p.fast)<.0001||o.count()>3)throw Error("scalar feedback steering");
    if(JSON.stringify(fixed)!==JSON.stringify(b.x))throw Error("frozen bank");
    return {checks:["alternating probe","anchor frozen without commit","coverage decay","grain projection","scalar-only steering / <= 3 queries per pair","frozen bank"],sample:p.trace};
  }
  return {prepare,demo,oracle,Walker,selfTest};
})();
if(typeof module !== 'undefined') module.exports = Sweep;
