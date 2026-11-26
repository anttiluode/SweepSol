/* Node.js, no dependencies. */
const fs=require("node:fs"), S=require("./engine.js"), C=require("./swarm-core.js");
const argv=process.argv.slice(2);
if(argv.includes("--self-test")){console.log(JSON.stringify(C.selfTest(),null,2));process.exit(0)}
const p={};for(let i=0;i<argv.length;i+=2){if(!argv[i].startsWith("--")||argv[i+1]===undefined)throw Error("Use --key value");p[argv[i].slice(2)]=argv[i+1]}
const valid=new Set(["bank","train","test","lanes","beats","horizon","width","road-weight","share","seed","grain","out"]);
for(const k of Object.keys(p))if(!valid.has(k))throw Error("Unknown option "+k);
const raw=p.bank==="demo"?S.demo():JSON.parse(fs.readFileSync(p.bank||"bank.json","utf8"));
const bank=S.prepare(raw);
const n=(k,d)=>p[k]===undefined?d:Number(p[k]);
const result=C.experiment(bank,{train:n("train",32),test:n("test",32),lanes:n("lanes",4),maxBeats:n("beats",160),horizon:n("horizon",32),width:n("width",90),roadWeight:n("road-weight",.18),share:n("share",.22),seed:n("seed",17),grain:p.grain||"auto",progress:x=>console.log(x.arm,x.done+"/"+x.total,"roads",x.roads)});
const out=p.out||"swarm-results.json";
fs.writeFileSync(out,JSON.stringify(result,null,2));
const names=i=>bank.words[i];
console.table(result.summary.train?[{arm:"train",...result.summary.train},{arm:"no roads",...result.summary.noRoads},{arm:"roads",...result.summary.roads}]:[{arm:"no roads",...result.summary.noRoads},{arm:"roads",...result.summary.roads}]);
console.log("Road-only successes",result.summary.roadOnly,"Baseline-only successes",result.summary.baseOnly,"Both",result.summary.both);
console.log("Junctions (interior nodes of successful loop-erased training paths):");
console.table(result.roads.junctions.sort((a,b)=>b[1]-a[1]).slice(0,20).map(([i,n])=>({word:names(i),paths:n})));
console.log("Detailed trace:",out);
