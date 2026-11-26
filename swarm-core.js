/* SweepSol swarm: shared coverage + successful directed paths. */
const Swarm = (function SwarmFactory(S){
  const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
  const unit=a=>{let n=Math.sqrt(dot(a,a));return a.map(v=>n>1e-10?v/n:0)};
  const rng=seed=>()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296};
  const key=(a,b)=>a+":"+b;
  class Roads{
    constructor(){this.edges=new Map();this.junctions=new Map();this.successes=0}
    credit(path){
      // Erase loops before assigning credit, not before counting work.
      const clean=[];
      for(const i of path){const j=clean.indexOf(i);if(j>=0)clean.length=j+1;else clean.push(i)}
      if(clean.length<2)return;
      this.successes++;
      for(let j=1;j<clean.length;j++){const k=key(clean[j-1],clean[j]);this.edges.set(k,(this.edges.get(k)||0)+1)}
      for(const i of clean.slice(1,-1))this.junctions.set(i,(this.junctions.get(i)||0)+1);
    }
    bonus(a,b){return Math.log1p(this.edges.get(key(a,b))||0)}
    snapshot(){return {edges:[...this.edges].sort(),junctions:[...this.junctions].sort(),successes:this.successes}}
    static from(s){const r=new Roads();r.edges=new Map(s.edges);r.junctions=new Map(s.junctions);r.successes=s.successes;return r}
  }
  // Cache only local geometry (no reward or target). Bounded LRU.
  class Geometry{
    constructor(bank,limit=1500){this.b=bank;this.cache=new Map();this.limit=limit;this.comparisons=0}
    neighbors(a,band,n){
      const k=a+":"+band+":"+n;
      if(this.cache.has(k)){const v=this.cache.get(k);this.cache.delete(k);this.cache.set(k,v);return v}
      const v=this.b.views[band],x=this.b.x;
      const rows=[];for(let i=0;i<v.length;i++)if(i!==a)rows.push({i,sim:dot(v[a],v[i])});
      this.comparisons+=rows.length;
      rows.sort((p,q)=>q.sim-p.sim||p.i-q.i);
      const out=rows.slice(0,n).map(p=>({...p,u:unit(x[p.i].map((z,j)=>z-x[a][j]))}));
      this.cache.set(k,out);if(this.cache.size>this.limit)this.cache.delete(this.cache.keys().next().value);
      return out;
    }
  }
  class Colony{
    constructor(bank,{roads=new Roads(),geometry=new Geometry(bank),roadWeight=.18,share=.22,sharedDecay=.04,horizon=32,width=90,grain="auto",threshold=.015,maxBeats=160,learn=true,coverage=.35,seed=7,temperature=.01,avoidRevisit=false}={}){
      this.b=bank;this.roads=roads;this.geometry=geometry;this.roadWeight=roadWeight;this.share=share;this.sharedDecay=sharedDecay;this.horizon=horizon;this.width=width;this.grain=grain;this.threshold=threshold;this.maxBeats=maxBeats;this.learn=learn;this.coverage=coverage;this.seed=seed;this.temperature=temperature;this.avoidRevisit=avoidRevisit;
      this.cover=new Float64Array(bank.words.length);this.agents=[];this.round=0;this.reads=new Set();this.meetings=new Map();this.observers=new Map();this.deposits=[];
    }
    add(start,target=null,seed=this.seed+this.agents.length){
      if(!Number.isInteger(start)||start<0||start>=this.b.words.length||target!==null&&(!Number.isInteger(target)||target<0||target>=this.b.words.length||target===start))throw Error("Invalid start or goal");
      const w=new S.Walker(this.b,start,{seed,neighbors:this.horizon,width:this.width,grain:this.grain,threshold:this.threshold,coverage:this.coverage,commonUtility:true,temperature:this.temperature,avoidRevisit:this.avoidRevisit});
      w.neighborhood=band=>this.geometry.neighbors(w.a,band,this.horizon);
      w.scoreBonus=(a,i)=>this.roadWeight*this.roads.bonus(a,i)-this.share*this.cover[i];
      const oracle=target===null?null:S.oracle(this.b,target,0,seed);
      w.setReward(oracle?i=>oracle.read(i):null);
      // The target is held by the evaluator, not passed to Walker.
      const a={id:this.agents.length,w,target,oracle,start,path:[start],done:false,success:false,firstSeen:null,seen:new Set(),empty:0,beats:0};
      this.agents.push(a);return a;
    }
    step(){
      this.round++;for(let i=0;i<this.cover.length;i++)this.cover[i]*=1-this.sharedDecay;
      // Simultaneous round: all readers see the previous shared trace and roads.
      const pending=[],credits=[];
      for(const a of this.agents){
        if(a.done)continue;
        const r=a.w.step();a.beats++;
        if(r.probe===null)a.empty++;
        else{
          pending.push([a.id,r.probe]);a.seen.add(r.probe);this.reads.add(r.probe);
          if(r.probe===a.target&&a.firstSeen===null)a.firstSeen=a.beats;
        }
        if(r.commit)a.path.push(r.after);
        if(a.target!==null&&a.w.a===a.target){
          a.done=true;a.success=true;
          if(this.learn)credits.push(a.path.slice());
        }else if(a.beats>=this.maxBeats)a.done=true;
      }
      for(const [id,i] of pending){
        this.cover[i]+=1;
        let ids=this.observers.get(i);if(!ids){ids=new Set();this.observers.set(i,ids)}
        ids.add(id);if(ids.size>1)this.meetings.set(i,ids.size);
      }
      for(const path of credits){this.roads.credit(path);this.deposits.push({round:this.round,path})}
      return this.agents.some(a=>!a.done);
    }
    result(){
      return {rounds:this.round,uniqueProbes:this.reads.size,meetings:[...this.meetings].sort((a,b)=>b[1]-a[1]),deposits:this.deposits,agents:this.agents.map(a=>({start:a.start,target:a.target,success:a.success,firstSeen:a.firstSeen,beats:a.beats,calls:a.oracle?.count()||0,empty:a.empty,downhill:a.w.trace.filter(r=>r.downhill).length,unique:a.seen.size,path:a.path,trace:a.w.trace}))};
    }
  }
  function pairs(n,train,test,seed=17){
    if(n<4||train<0||test<1||train+test>n*(n-1)/2)throw Error("Too many distinct pairs");
    const r=rng(seed),used=new Set(),out=[];
    while(out.length<train+test){let a=Math.floor(r()*n),b=Math.floor(r()*n);if(a===b)continue;const k=key(Math.min(a,b),Math.max(a,b));if(used.has(k))continue;used.add(k);out.push([a,b])}
    return {train:out.slice(0,train),test:out.slice(train)};
  }
  function aggregate(rows,budget){
    const ok=rows.filter(a=>a.success),mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
    return {n:rows.length,success:ok.length,successRate:ok.length/rows.length,meanCallsAll:mean(rows.map(a=>a.calls)),meanCallsSuccessful:mean(ok.map(a=>a.calls)),meanBeatsSuccessful:mean(ok.map(a=>a.beats)),meanBudgetedBeats:mean(rows.map(a=>a.success?a.beats:budget)),meanEmptyRate:mean(rows.map(a=>a.empty/a.beats))};
  }
  function experiment(bank,{train=32,test=32,lanes=4,seed=17,maxBeats=160,horizon=32,width=90,roadWeight=.18,share=.22,grain="auto",threshold=.015,temperature=.01,avoidRevisit=false,progress=()=>{}}={}){
    if(![train,test,lanes,seed,maxBeats,horizon].every(Number.isInteger)||train<0||test<1||lanes<1||maxBeats<1||horizon<2)throw Error("Invalid experiment options");
    const split=pairs(bank.words.length,train,test,seed),roads=new Roads(),geometry=new Geometry(bank);
    function run(list,arm,learn,weight){
      const rows=[];
      for(let j=0;j<list.length;j+=lanes){
        const c=new Colony(bank,{roads,geometry,roadWeight:weight,share,horizon,width,grain,threshold,maxBeats,learn,seed,temperature,avoidRevisit});
        list.slice(j,j+lanes).forEach(([a,b],i)=>c.add(a,b,seed+j+i+1000));
        while(c.step()){}
        rows.push(...c.result().agents);progress({arm,done:rows.length,total:list.length,roads:roads.edges.size});
      }
      return rows;
    }
    const tr=run(split.train,"train",true,roadWeight);
    const frozen=JSON.stringify(roads.snapshot());
    const off=run(split.test,"no_roads",false,0),on=run(split.test,"roads",false,roadWeight);
    if(JSON.stringify(roads.snapshot())!==frozen)throw Error("Holdout modified frozen roads");
    const paired=on.map((a,i)=>({start:a.start,target:a.target,roads:a.success,noRoads:off[i].success,roadCalls:a.calls,baseCalls:off[i].calls,roadBeats:a.beats,baseBeats:off[i].beats}));
    return {meta:bank.meta,words:bank.words,config:{train,test,lanes,seed,maxBeats,horizon,width,roadWeight,share,grain,threshold,temperature,avoidRevisit},split,roads:roads.snapshot(),summary:{train:tr.length?aggregate(tr,maxBeats):null,noRoads:aggregate(off,maxBeats),roads:aggregate(on,maxBeats),roadOnly:paired.filter(x=>x.roads&&!x.noRoads).length,baseOnly:paired.filter(x=>!x.roads&&x.noRoads).length,both:paired.filter(x=>x.roads&&x.noRoads).length,geometryComparisons:geometry.comparisons},paired,runs:{train:tr,noRoads:off,roads:on}};
  }
  function selfTest(){
    const b=S.prepare(S.demo()),before=JSON.stringify(b.x),r=new Roads(),g=new Geometry(b);
    r.credit([0,1,2,1,3]);if(r.edges.size!==2||r.bonus(0,1)<=0||r.bonus(1,2)!==0)throw Error("loop-erased road credit");
    const c=new Colony(b,{roads:r,geometry:g,learn:false,width:180,maxBeats:12,horizon:8});
    c.add(0);c.add(0);const snap=JSON.stringify(r.snapshot());c.step();
    if(!c.reads.size||![...c.cover].some(x=>x>0))throw Error("shared trace missing");
    if(c.agents.some(a=>a.w.trace[0].anchor!==0||a.w.trace[0].after!==0))throw Error("look moved anchor early");
    while(c.step()){}if(JSON.stringify(r.snapshot())!==snap)throw Error("frozen roads modified");
    const test=experiment(b,{train:4,test:4,lanes:2,maxBeats:20,horizon:8,width:180,seed:33});
    const ids=x=>new Set(x.map(([a,b])=>key(Math.min(a,b),Math.max(a,b))));
    const a=ids(test.split.train);if([...ids(test.split.test)].some(k=>a.has(k)))throw Error("split leakage");
    if(test.paired.length!==4||test.runs.roads.some((a,i)=>a.start!==test.runs.noRoads[i].start||a.target!==test.runs.noRoads[i].target))throw Error("unmatched evaluation");
    if(JSON.stringify(b.x)!==before)throw Error("bank changed");
    // Direct verify road and coverage hooks change candidate scoring.
    const w=c.agents[0].w; c.roadWeight=1;c.share=0;
    if(w.scoreBonus(0,1)<=w.scoreBonus(0,2))throw Error("road hook inactive");
    c.roadWeight=0;c.share=1;c.cover[1]=2;c.cover[2]=0;
    if(w.scoreBonus(0,1)>=w.scoreBonus(0,2))throw Error("coverage hook inactive");
    return {checks:["loop-erased successful paths","shared coverage","held anchor on first look","frozen evaluation roads","disjoint unordered train/test pairs","matched test pairs","unchanged bank","road and coverage scoring hooks"],toySummary:test.summary};
  }
  return {Roads,Geometry,Colony,pairs,experiment,selfTest,rng};
})(typeof module !== 'undefined' ? require('./engine.js') : Sweep);
if(typeof module !== 'undefined') module.exports = Swarm;
