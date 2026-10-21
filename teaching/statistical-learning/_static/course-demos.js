/* Small deterministic teaching simulations. No network calls or external data. */
(function (root) {
  'use strict';
  function rng(seed) {
    let a = (Number(seed) >>> 0) || 1;
    return function () {
      a += 0x6D2B79F5;
      let t = a;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function normal(random) {
    return Math.sqrt(-2 * Math.log(Math.max(random(), 1e-15))) * Math.cos(2 * Math.PI * random());
  }
  function tail(x) {
    if (x < 0) return 1 - tail(-x);
    const t = 1 / (1 + 0.2316419 * x);
    return Math.exp(-0.5*x*x) / Math.sqrt(2*Math.PI) * t *
      (0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))));
  }
  const cdf = x => 1-tail(x);
  function selectionP(values) {
    const ranked = values.map((v,i)=>({v:Math.abs(v),i})).sort((a,b)=>b.v-a.v);
    const z = ranked[0].v, m = ranked.length > 1 ? ranked[1].v : 0;
    return {index:ranked[0].i,z,m,naive:Math.min(1,2*tail(z)),selective:Math.min(1,tail(z)/tail(m))};
  }
  function bandit(means, horizon, seed) {
    const random=rng(seed), K=means.length;
    const tape=means.map(mu=>Array.from({length:horizon},()=>random()<mu?1:0));
    function run(ucb) {
      const counts=Array(K).fill(0), sums=Array(K).fill(0), points=[];
      let regret=0;
      for(let t=1;t<=horizon;t++) {
        let arm=(t-1)%K;
        if(ucb && t>K) {
          let best=-Infinity;
          for(let k=0;k<K;k++) {
            const score=sums[k]/counts[k]+Math.sqrt(2*Math.log(t)/counts[k]);
            if(score>best){best=score;arm=k;}
          }
        }
        sums[arm]+=tape[arm][counts[arm]];
        counts[arm]++;
        regret+=Math.max(...means)-means[arm];
        points.push([t,regret]);
      }
      return {counts,regret,points};
    }
    return {uniform:run(false),ucb:run(true)};
  }
  function riskExperiment(o) {
    const random=rng(o.seed), sample=prior=>Array.from({length:o.n},()=>normal(random)+(random()<prior?o.delta/2:-o.delta/2));
    const x1=o.kind==='pu'?Array.from({length:o.n},()=>normal(random)+o.delta/2):sample(o.a);
    const x2=sample(o.kind==='pu'?o.pi:o.b);
    if(o.kind==='uu' && Math.abs(o.a-o.b)<1e-10) return {identified:false};
    const truth=[],corrected=[],nonnegative=[];
    for(let j=0;j<=80;j++) {
      const t=-4+j*.1;
      const e1=x1.filter(x=>x<=t).length/o.n, e2=x2.filter(x=>x<=t).length/o.n;
      const actual=o.pi*cdf(t-o.delta/2)+(1-o.pi)*tail(t+o.delta/2);
      let risk;
      if(o.kind==='pu') {
        const neg=1-e2-o.assumed*(1-e1);
        risk=o.assumed*e1+neg;
        nonnegative.push([t,o.assumed*e1+Math.max(0,neg)]);
      } else {
        const pos=((1-o.b)*e1-(1-o.a)*e2)/(o.a-o.b);
        const neg=(o.a*(1-e2)-o.b*(1-e1))/(o.a-o.b);
        risk=o.pi*pos+(1-o.pi)*neg;
      }
      truth.push([t,actual]);corrected.push([t,risk]);
    }
    return {identified:true,truth,corrected,nonnegative};
  }
  const api={rng,normal,tail,cdf,selectionP,bandit,riskExperiment};
  if(typeof module!=='undefined' && module.exports) module.exports=api;
  if(typeof document==='undefined') return;
  const $=id=>document.getElementById(id);
  const num=id=>Number($(id).value);
  const fmt=(x,d=3)=>Number(x).toFixed(d);
  function safeSeed(id) {
    let n=Math.floor(num(id));
    if(!Number.isFinite(n)||n<1) n=1;
    n=Math.min(n,999999);$(id).value=n;return n;
  }
  function plot(target, series, options) {
    const W=660,H=310,L=65,R=20,T=24,B=50;
    const all=series.flatMap(s=>s.points), xs=all.map(v=>v[0]),ys=all.map(v=>v[1]);
    const xmin=options.xmin??Math.min(...xs),xmax=options.xmax??Math.max(...xs);
    let ymin=options.ymin??Math.min(0,...ys),ymax=options.ymax??Math.max(...ys);
    if(ymax-ymin<.01)ymax=ymin+1;
    const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;
    const X=x=>L+(x-xmin)/(xmax-xmin||1)*(W-L-R),Y=y=>H-B-(y-ymin)/(ymax-ymin)*(H-T-B);
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${options.title}"><title>${options.title}</title>`;
    for(let i=0;i<=4;i++) {
      const y=ymin+(ymax-ymin)*i/4, x=xmin+(xmax-xmin)*i/4;
      svg+=`<line class="chart-grid" x1="${L}" x2="${W-R}" y1="${Y(y)}" y2="${Y(y)}"/><text x="${L-9}" y="${Y(y)+4}" text-anchor="end">${fmt(y,2)}</text><text x="${X(x)}" y="${H-B+20}" text-anchor="middle">${fmt(x,options.xdigits??0)}</text>`;
    }
    svg+=`<line class="chart-axis" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/><text x="${(L+W-R)/2}" y="${H-7}" text-anchor="middle">${options.xlabel}</text><text x="${L}" y="15">${options.ylabel}</text>`;
    series.forEach((s,i)=> {
      svg+=`<polyline class="series-${i}" fill="none" stroke-width="2.4" ${s.dashed?'stroke-dasharray="6 4"':''} points="${s.points.map(([x,y])=>`${X(x).toFixed(2)},${Y(y).toFixed(2)}`).join(' ')}"/>`;
      if(s.dots)svg+=s.points.map(([x,y])=>`<circle class="series-${i}" cx="${X(x)}" cy="${Y(y)}" r="4" fill="var(--pst-color-background)" stroke-width="2.4"/>`).join('');
    });
    $(target).innerHTML=svg+'</svg>';
  }
  function wire(rootId, render, buttonId, seedId) {
    const section=$(rootId);if(!section)return;
    section.querySelectorAll('input,select').forEach(el=>el.addEventListener('input',()=>{
      const out=$(el.id+'-value');if(out)out.textContent=el.value;
      render();
    }));
    $(buttonId).addEventListener('click',()=>{$(seedId).value=(safeSeed(seedId)%999999)+1;render();});
    render();
  }
  function init() {
    // Keep solutions independently keyboard-operable even without a theme plugin.
    document.querySelectorAll('article .admonition.dropdown').forEach(box=>{
      const title=box.querySelector('.admonition-title');
      const d=document.createElement('details'), s=document.createElement('summary');
      d.className='course-solution';if(box.id)d.id=box.id;
      s.textContent=title?title.textContent:'Worked solutions';if(title)title.remove();
      d.append(s);while(box.firstChild)d.append(box.firstChild);box.replaceWith(d);
    });
    wire('risk-demo',()=>{
      const kind=$('risk-kind').value;
      document.querySelectorAll('.risk-uu-control').forEach(x=>x.hidden=kind!=='uu');
      $('risk-assumed-control').hidden=kind!=='pu';
      const o={kind,delta:num('risk-delta'),pi:num('risk-pi'),assumed:num('risk-assumed'),a:num('risk-a'),b:num('risk-b'),n:num('risk-n'),seed:safeSeed('risk-seed')};
      const r=riskExperiment(o);
      if(!r.identified){$('risk-chart').innerHTML='';$('risk-status').textContent='The two mixture priors are equal. This inversion is not identified; no corrected curve is computed.';return;}
      const series=[{points:r.truth},{points:r.corrected}];if(kind==='pu')series.push({points:r.nonnegative,dashed:true});
      plot('risk-chart',series,{title:'True and estimated classification risk across thresholds',xlabel:'Threshold t',ylabel:'Misclassification risk',xdigits:1});
      const min=Math.min(...r.corrected.map(x=>x[1]));
      $('risk-status').textContent=kind==='pu'?`Independent samples: ${o.n} positives and ${o.n} unlabelled observations. True π = ${fmt(o.pi,2)}; supplied π = ${fmt(o.assumed,2)}. Lowest corrected empirical risk on this grid: ${fmt(min)}. The non-negative correction changes the estimator; it does not guarantee lower test error.`:`Two independent samples of ${o.n}, target π = ${fmt(o.pi,2)}, and |a − b| = ${fmt(Math.abs(o.a-o.b),2)}. The inversion contains 1/(a − b), so similar mixtures amplify sampling noise. Lowest corrected empirical risk on the grid: ${fmt(min)}.`;
    },'risk-resample','risk-seed');
    wire('bandit-demo',()=>{
      const means=[.30,.45,num('bandit-mu')],horizon=num('bandit-horizon');
      const r=bandit(means,horizon,safeSeed('bandit-seed'));
      plot('bandit-chart',[{points:r.uniform.points},{points:r.ucb.points}],{title:'Cumulative pseudo-regret of uniform exploration and UCB',xlabel:'Round',ylabel:'Cumulative pseudo-regret',xmin:0});
      $('bandit-status').innerHTML=`<p>Arm means: ${means.map(x=>fmt(x,2)).join(', ')}. After ${horizon} rounds:</p><p><strong>Uniform:</strong> pseudo-regret ${fmt(r.uniform.regret,1)}; pull counts (${r.uniform.counts.join(', ')}).<br><strong>UCB:</strong> pseudo-regret ${fmt(r.ucb.regret,1)}; pull counts (${r.ucb.counts.join(', ')}).</p>`;
    },'bandit-resample','bandit-seed');
    wire('selection-demo',()=>{
      const p=num('selection-p'),alpha=num('selection-alpha'),seed=safeSeed('selection-seed');
      const random=rng(seed),z=Array.from({length:p},()=>normal(random)),s=selectionP(z);
      const points=z.map((x,i)=>[i+1,Math.abs(x)]);
      plot('selection-chart',[{points,dots:true},{points:[[s.index+1,s.z]],dots:true},{points:[[1,s.m],[Math.max(2,p),s.m]],dashed:true}],{title:'Absolute Gaussian statistics with selected maximum and truncation threshold',xlabel:'Candidate index (pink marks the selected statistic)',ylabel:'Absolute statistic',xmin:1,xmax:Math.max(2,p)});
      const repeat=rng(seed+1000003),B=3000;let naive=0,selective=0;
      for(let b=0;b<B;b++){const q=selectionP(Array.from({length:p},()=>normal(repeat)));naive+=q.naive<=alpha;selective+=q.selective<=alpha;}
      const nr=naive/B,sr=selective/B,se=r=>Math.sqrt(r*(1-r)/B);
      $('selection-status').innerHTML=`<div class="demo-metrics"><div><strong>${fmt(s.naive,4)}</strong>naive p-value</div><div><strong>${fmt(s.selective,4)}</strong>selective p-value</div><div><strong>${fmt(s.m)}</strong>truncation threshold m</div></div><p>Selected candidate ${s.index+1}: Z = ${fmt(z[s.index])}. Across ${B.toLocaleString()} null repetitions, naive rejection = ${fmt(nr)} (MC SE ${fmt(se(nr),4)}), selective rejection = ${fmt(sr)} (MC SE ${fmt(se(sr),4)}). The exact naive rejection probability is ${fmt(1-(1-alpha)**p)}; the target level is ${fmt(alpha,2)}.</p>`;
    },'selection-resample','selection-seed');
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})(typeof globalThis!=='undefined'?globalThis:this);
