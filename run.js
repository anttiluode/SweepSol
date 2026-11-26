// Optional headless runner. The browser is the primary interface.
const fs=require("node:fs"), S=require("./engine.js");
const args=process.argv.slice(2);
if(args.includes("--self-test")){console.log(JSON.stringify(S.selfTest(),null,2));process.exit(0)}
const b=S.prepare(args[0]&&args[0]!=="demo"?JSON.parse(fs.readFileSync(args[0],"utf8")):S.demo());
const find=w=>{let i=b.words.indexOf(w);if(i<0)throw Error("Word not in bank: "+w);return i};
const start=find(args[1]||"oak"),target=args[2]&&args[2]!=="-"?find(args[2]):null;
const steps=Number(args[3]||100), switchTarget=args[4]?find(args[4]):null;
if(!Number.isInteger(steps)||steps<1)throw Error("Steps must be a positive integer");
const w=new S.Walker(b,start), o=target===null?null:S.oracle(b,target);
if(switchTarget!==null&&!o)throw Error("A first target is required for a switch");
w.setReward(o?i=>o.read(i):null);
let events=[{beat:0,target:target===null?null:b.words[target]}];
for(let j=0;j<steps;j++){
  if(switchTarget!==null&&j===Math.floor(steps/2)){o.retarget(switchTarget);w.setReward(i=>o.read(i));events.push({beat:j,target:b.words[switchTarget]})}
  const r=w.step();r.oracleCalls=o?.count()??0;
  console.log(j,r.side===1?"L":"R",b.words[r.anchor],"look",r.probe===null?"[empty]":b.words[r.probe],b.bands[r.band].name,r.reward===null?"":r.reward.toFixed(3),r.commit?"COMMIT "+b.words[r.after]:"");
}
fs.writeFileSync("trace.json",JSON.stringify({meta:b.meta,words:b.words,bands:b.bands,events,trace:w.trace,summary:{unique:w.visited.size,oracleCalls:o?.count()??0}},null,2));
console.log("Final:",b.words[w.a],"unique probes:",w.visited.size,"scalar calls:",o?.count()??0,"→ trace.json");
