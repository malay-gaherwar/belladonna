/* BELLADONNA Factoid Atlas — per-factoid scatter with data-driven colour modes.
   Renders N factoid dots via ImageData pixel-blit (same technique as the
   knowledge-graph atlas). Colour modes (source / drug_class / biomarker /
   setting / evidence / topic / year) all come from manifest.json. */
(() => {
'use strict';
const DATA = './data/';
const BUST = '_=' + Date.now();   // per-load cache-buster so repacked data is fresh
const bust = (u) => u + (u.includes('?') ? '&' : '?') + BUST;
const $ = (id) => document.getElementById(id);
const fmt = (n) => n == null ? '—' : n.toLocaleString('en-US');
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const _EB = new ArrayBuffer(4); new Uint32Array(_EB)[0] = 0xff;
const LE = new Uint8Array(_EB)[0] === 0xff;
const packRGBA = LE
  ? (r,g,b,a) => ((a&255)<<24)|((b&255)<<16)|((g&255)<<8)|(r&255)
  : (r,g,b,a) => ((r&255)<<24)|((g&255)<<16)|((b&255)<<8)|(a&255);
const hexToRgb = (h) => { const n = parseInt(h.slice(1),16); return [(n>>16)&255,(n>>8)&255,n&255]; };
function topicPalette(k){
  const base=['#E8177F','#3B5466','#F59E0B','#8B5CF6','#10B981','#EF4444','#06B6D4','#F472B6','#60A5FA',
    '#A855F7','#FACC15','#22C55E','#F97316','#14B8A6','#EC4899','#6366F1','#84CC16','#0EA5E9','#D946EF',
    '#EAB308','#FB7185','#34D399','#A78BFA','#FB923C','#4ADE80','#2DD4BF','#C084FC','#F87171'];
  return Array.from({length:k}, (_,i)=>hexToRgb(base[i%base.length]));
}
function yearColor(y,y0,y1){
  if(!y) return [90,100,116];
  const t=Math.max(0,Math.min(1,(y-y0)/Math.max(1,y1-y0)));
  return [Math.round(58+(232-58)*t), Math.round(84+(23-84)*t), Math.round(110+(127-110)*t)];
}
// HSL->RGB (h 0-360, s,l 0-1) for the combined tri-family shading
function hsl(h,s,l){ h=((h%360)+360)%360/360; const a=s*Math.min(l,1-l);
  const f=n=>{const k=(n+h*12)%12; return l-a*Math.max(-1,Math.min(k-3,9-k,1));};
  return [Math.round(f(0)*255),Math.round(f(8)*255),Math.round(f(4)*255)]; }
// drug_class = reds, biomarker = blues, setting = greens; shade by category index
const FAMILY_HUE={drug_class:6, biomarker:215, setting:135};
const FAMILY_REP={drug_class:'#e04b59', biomarker:'#3a86ff', setting:'#52b788'};
const FAMILY_NAME={drug_class:'Drug class', biomarker:'Biomarker', setting:'Disease setting'};
function familyRGB(dimKey,cat,K){ const t=(K<=2)?0.5:(cat-1)/(K-2);
  const h=FAMILY_HUE[dimKey]+34*(t-0.5); const l=0.46+0.28*t; return hsl(h,0.74,l); }

const S = {
  N:0, px:null, py:null,
  attr:{},                       // key -> typed array
  manifest:null, clusters:[], docs:null, docsLoading:false,
  docsShards:null, docShardSize:0, docCache:new Map(), docLoading:new Set(), docsAllLoaded:false,
  modes:[],                      // [{key,title,kind,colors,values,labels}]
  modeIdx:0,   // open on the combined tri-family view (all 3 categorisations at once)
  activeCat:new Set(), hideNone:true, showLabels:true,
  srcFilter:new Set(), query:[], qops:[], yMin:0, yMax:0, searchMask:null,   // query: [{dim,code,name,color}] chained left-to-right by qops ('and'|'or')
  pointSize:2, alpha:0.8,
  view:{tx:0,ty:0,scale:600}, canvas:null, ctx:null, dpr:1, width:0, height:0,
  dragging:false, moved:false, lastX:0, lastY:0, hoverIdx:-1, selected:-1,
  pendingDraw:false, img:null, imgW:0, imgH:0, buf32:null, bounds:null,
  catCounts:{},                  // modeKey -> Int32Array counts
};

const requestDraw = () => { if(S.pendingDraw) return; S.pendingDraw=true;
  requestAnimationFrame(()=>{S.pendingDraw=false; draw();}); };

// --------------- data ---------------
async function fetchJSON(u,onP){ const r=await fetch(bust(u)); if(!r.ok) throw new Error(`${u} ${r.status}`);
  if(!onP||!r.body) return r.json(); const total=+r.headers.get('content-length')||0;
  const rd=r.body.getReader(); const chunks=[]; let got=0;
  for(;;){ const {done,value}=await rd.read(); if(done) break; chunks.push(value); got+=value.byteLength; if(total) onP(got/total); }
  return JSON.parse(await new Blob(chunks).text()); }
async function fetchBin(u){ const r=await fetch(bust(u)); if(!r.ok) throw new Error(`${u} ${r.status}`); return new Uint8Array(await r.arrayBuffer()); }

async function loadData(){
  const [manifest, clusters, pts, attrs] = await Promise.all([
    fetchJSON(DATA+'manifest.json'), fetchJSON(DATA+'clusters.json').catch(()=>[]),
    fetchBin(DATA+'points.bin'), fetchBin(DATA+'attrs.bin'),
  ]);
  S.manifest=manifest; S.clusters=clusters||[];
  S.docsShards=manifest.docsShards||null; S.docShardSize=S.docsShards?S.docsShards.size:0;
  const N=manifest.n; S.N=N;
  const pos=new Float32Array(pts.buffer, pts.byteOffset, N*2);
  S.px=new Float32Array(N); S.py=new Float32Array(N);
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for(let i=0;i<N;i++){ const x=pos[i*2],y=pos[i*2+1]; S.px[i]=x; S.py[i]=y;
    if(x<minX)minX=x; if(x>maxX)maxX=x; if(y<minY)minY=y; if(y>maxY)maxY=y; }
  S.bounds={minX,minY,maxX,maxY};

  // parse attrs per manifest.attrLayout
  const RB=manifest.attrRecordBytes, L=manifest.attrLayout;
  const dv=new DataView(attrs.buffer, attrs.byteOffset, attrs.byteLength);
  const mk=(spec)=>{ const [t,off]=spec; const a = t==='u16'?new Uint16Array(N):new Uint8Array(N);
    for(let i=0;i<N;i++) a[i] = t==='u16'? dv.getUint16(i*RB+off,true) : dv.getUint8(i*RB+off); return a; };
  for(const k in L) S.attr[k]=mk(L[k]);

  // build modes (source + each dimension + topic + year)
  const srcColors = manifest.sources.map(s=>hexToRgb(s.color));
  S.modes=[{key:'source', title:'Source', kind:'source',
            colors:srcColors, values:S.attr.src,
            labels:manifest.sources.map(s=>({name:s.name,color:s.color,count:s.count}))}];
  for(const d of manifest.dimensions){
    S.modes.push({key:d.key, title:d.title, kind:'dim',
      colors:d.labels.map(l=>hexToRgb(l.color)), values:S.attr[d.key],
      labels:d.labels.map(l=>({name:l.name,color:l.color,code:l.code}))});
  }
  if(manifest.k>0 && S.attr.cluster){
    const tc=topicPalette(manifest.k);
    S.modes.push({key:'cluster', title:'Topic', kind:'cluster',
      colors:tc, values:S.attr.cluster,
      labels:Array.from({length:manifest.k},(_,i)=>{ const c=S.clusters[i]||{};
        return {name:(c.terms||[]).slice(0,3).join(', ')||('Topic '+i), color:`rgb(${tc[i].join(',')})`, count:c.count}; })});
  }
  S.modes.push({key:'year', title:'Year', kind:'year', colors:null, values:S.attr.year, labels:[]});

  // counts per category for each mode
  for(const m of S.modes){
    if(m.kind==='year') continue;
    const c=new Int32Array(m.labels.length);
    const v=m.values; for(let i=0;i<N;i++){ const k=v[i]; if(k<c.length) c[k]++; }
    S.catCounts[m.key]=c;
  }

  // --- combined tri-family encoding: drug=red, biomarker=blue, setting=green ---
  // each point's colour = its MOST-SPECIFIC categorisation (rarest non-none
  // label across the 3 dims), shaded within that dimension's hue family.
  const cdims=['drug_class','biomarker','setting'].filter(k=>S.attr[k] && S.modes.find(m=>m.key===k));
  if(cdims.length){
    const dimMode={}, ramp={};
    cdims.forEach(k=>{ dimMode[k]=S.modes.find(m=>m.key===k);
      const K=dimMode[k].labels.length, arr=[];
      for(let c=0;c<K;c++) arr.push(c===0?[58,63,75]:familyRGB(k,c,K)); ramp[k]=arr; });
    S.domDim=new Uint8Array(N); S.domCat=new Uint8Array(N); S.domCol=new Uint8Array(N*3);
    for(let i=0;i<N;i++){
      let bd=-1,bc=0,bn=Infinity;
      for(let di=0;di<cdims.length;di++){ const k=cdims[di], c=S.attr[k][i];
        if(c===0) continue; const gc=S.catCounts[k][c]||1; if(gc<bn){bn=gc;bd=di;bc=c;} }
      if(bd<0){ S.domCol[i*3]=58;S.domCol[i*3+1]=63;S.domCol[i*3+2]=75; }
      else { S.domDim[i]=bd+1; S.domCat[i]=bc; const rgb=ramp[cdims[bd]][bc];
             S.domCol[i*3]=rgb[0];S.domCol[i*3+1]=rgb[1];S.domCol[i*3+2]=rgb[2]; }
    }
    S.cdims=cdims; S.dimMode=dimMode; S.ramp=ramp; S.famActive=new Set(cdims.map((_,i)=>i+1));
    S.modes.unshift({key:'combined', title:'All (combined)', kind:'combined', colors:null, values:null, labels:[]});
  }

  S.srcFilter=new Set(manifest.sources.map(s=>s.id));
  const [y0,y1]=manifest.yearRange; S.yMin=y0; S.yMax=y1;
}

// --------------- canvas ---------------
function setupCanvas(){
  S.canvas=$('canvas'); S.ctx=S.canvas.getContext('2d');
  resize(); fitView();
  addEventListener('resize',resize);
  S.canvas.addEventListener('mousedown',e=>{S.dragging=true;S.moved=false;S.lastX=e.clientX;S.lastY=e.clientY;});
  addEventListener('mouseup',()=>S.dragging=false);
  let hr=0,le=null;
  S.canvas.addEventListener('mousemove',e=>{
    if(S.dragging){ const dx=e.clientX-S.lastX,dy=e.clientY-S.lastY; if(Math.abs(dx)+Math.abs(dy)>2)S.moved=true;
      S.view.tx+=dx; S.view.ty+=dy; S.lastX=e.clientX; S.lastY=e.clientY; requestDraw(); }
    else { le=e; if(!hr) hr=requestAnimationFrame(()=>{hr=0; if(le) hover(le);}); }
  });
  S.canvas.addEventListener('mouseleave',()=>{S.hoverIdx=-1; $('tt').style.display='none';});
  S.canvas.addEventListener('wheel',e=>{ e.preventDefault();
    const r=S.canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
    const wx=(mx-S.width/2-S.view.tx)/S.view.scale, wy=(my-S.height/2-S.view.ty)/S.view.scale;
    const f=e.deltaY<0?1.15:1/1.15; S.view.scale=Math.max(60,Math.min(120000,S.view.scale*f));
    S.view.tx=mx-S.width/2-wx*S.view.scale; S.view.ty=my-S.height/2-wy*S.view.scale; requestDraw();
  },{passive:false});
  S.canvas.addEventListener('click',e=>{ if(S.moved) return; const i=pickAt(e); if(i>=0) openDetails(i); });
}
function resize(){
  S.dpr=Math.min(2,devicePixelRatio||1);
  const r=S.canvas.getBoundingClientRect(); S.width=r.width; S.height=r.height;
  S.canvas.width=Math.round(r.width*S.dpr); S.canvas.height=Math.round(r.height*S.dpr);
  const W=S.canvas.width,H=S.canvas.height;
  if(S.imgW!==W||S.imgH!==H){ S.img=S.ctx.createImageData(W,H); S.buf32=new Uint32Array(S.img.data.buffer); S.imgW=W; S.imgH=H; }
  requestDraw();
}
function fitView(){ const b=S.bounds; if(!b||!S.width) return;
  const cx=(b.minX+b.maxX)/2, cy=(b.minY+b.maxY)/2;
  const sx=(b.maxX-b.minX)||1, sy=(b.maxY-b.minY)||1;
  S.view.scale=Math.min(S.width/sx,S.height/sy)*0.9; S.view.tx=-cx*S.view.scale; S.view.ty=-cy*S.view.scale; }

// --------------- filter ---------------
function mode(){ return S.modes[S.modeIdx]; }
function passes(i){
  if(!S.srcFilter.has(S.attr.src[i])) return false;
  const y=S.attr.year[i]; if(y && (y<S.yMin||y>S.yMax)) return false;
  if(S.searchMask && !S.searchMask[i]) return false;
  if(S.query.length){ const q=S.query, op=S.qops;
    let acc = S.attr[q[0].dim][i]===q[0].code;
    for(let j=1;j<q.length;j++){ const t=S.attr[q[j].dim][i]===q[j].code; acc = op[j-1]==='and'?(acc&&t):(acc||t); }
    if(!acc) return false; }
  const m=mode();
  if(m.kind==='dim'||m.kind==='cluster'){ if(!S.activeCat.has(m.values[i])) return false; }
  if(m.kind==='dim' && S.hideNone && m.values[i]===0) return false;
  if(m.kind==='combined'){ const dd=S.domDim[i]; if(dd>0 && !S.famActive.has(dd)) return false;
    if(dd===0 && S.hideNone) return false; }
  return true;
}

// --------------- render ---------------
function draw(){
  const ctx=S.ctx; if(!ctx||!S.buf32) return;
  const W=S.imgW,H=S.imgH,buf=S.buf32; buf.fill(0);
  const dpr=S.dpr, scale=S.view.scale*dpr, tx=S.view.tx*dpr+W/2, ty=S.view.ty*dpr+H/2;
  let pxR=S.pointSize*dpr; if(S.view.scale<200) pxR=Math.max(dpr,pxR*0.6);
  const r=Math.max(1,Math.round(pxR)), halfR=(r-1)>>1;
  const aByte=Math.round(S.alpha*255), aDim=Math.round(S.alpha*0.12*255);
  const m=mode(), v=m.values;

  // colour LUT for this mode
  let lut=null, yrBase=0, isYear=(m.kind==='year'), isCombined=(m.kind==='combined');
  if(isYear){ const [y0,y1]=S.manifest.yearRange; yrBase=y0; lut=new Uint32Array(y1-y0+1);
    for(let y=y0;y<=y1;y++){ const [cr,cg,cb]=yearColor(y,y0,y1); lut[y-y0]=packRGBA(cr,cg,cb,aByte); } }
  else if(!isCombined){ lut=new Uint32Array(m.colors.length);
    for(let i=0;i<m.colors.length;i++){ const [cr,cg,cb]=m.colors[i]; lut[i]=packRGBA(cr,cg,cb,aByte); } }
  const noneIdx = (m.kind==='dim') ? 0 : -1;

  const src=S.attr.src, yr=S.attr.year, sf=S.srcFilter, yMin=S.yMin, yMax=S.yMax, mask=S.searchMask;
  const cat=S.activeCat, catCheck=(m.kind==='dim'||m.kind==='cluster'), hideNone=S.hideNone;
  // per-category screen centroid accumulators (for on-plot labels)
  const nCat=(isYear||isCombined)?0:m.labels.length;
  const cSumX=nCat?new Float64Array(nCat):null, cSumY=nCat?new Float64Array(nCat):null, cCnt=nCat?new Int32Array(nCat):null;
  // combined tri-family bits
  const domDim=S.domDim, domCol=S.domCol, famAct=S.famActive;
  const aGrey=Math.round(S.alpha*0.30*255);
  // cross-dimension AND filter: only dims with ≥1 selected code constrain
  const q=S.query, qn=q.length, qop=S.qops, qa=q.map(x=>S.attr[x.dim]), qc=q.map(x=>x.code);
  let vis=0;
  for(let i=0;i<S.N;i++){
    if(!sf.has(src[i])) continue;
    const y=yr[i]; if(y && (y<yMin||y>yMax)) continue;
    if(mask && !mask[i]) continue;
    if(qn){ let acc=qa[0][i]===qc[0];
            for(let j=1;j<qn;j++){ const t=qa[j][i]===qc[j]; acc=qop[j-1]==='and'?(acc&&t):(acc||t); }
            if(!acc) continue; }
    const val=isCombined?0:v[i];
    if(catCheck && !cat.has(val)) continue;
    let dd=0;
    if(isCombined){ dd=domDim[i]; if(dd>0 && !famAct.has(dd)) continue; if(dd===0 && hideNone) continue; }
    const fx=S.px[i]*scale+tx, fy=S.py[i]*scale+ty;
    const cx=fx|0, cy=fy|0, x0=cx-halfR, y0=cy-halfR;
    if(x0>=W||y0>=H||x0+r<=0||y0+r<=0) continue;
    let rgba;
    if(isCombined){ const o=i*3; rgba = dd===0? packRGBA(58,63,75,aGrey) : packRGBA(domCol[o],domCol[o+1],domCol[o+2],aByte); }
    else if(isYear) rgba = y? lut[y-yrBase] : 0xff64748b;
    else { if(hideNone && val===noneIdx) continue; rgba=lut[val]; }
    const xa=x0<0?0:x0, xb=(x0+r)>W?W:(x0+r), ya=y0<0?0:y0, yb=(y0+r)>H?H:(y0+r);
    for(let yy=ya;yy<yb;yy++){ const base=yy*W; for(let xx=xa;xx<xb;xx++) buf[base+xx]=rgba; }
    if(nCat){ cSumX[val]+=fx; cSumY[val]+=fy; cCnt[val]++; }
    vis++;
  }
  ctx.putImageData(S.img,0,0);

  // on-plot category labels at each category's centroid (single-axis modes only;
  // the combined view shows the families in the side legend instead).
  if(S.showLabels && nCat){
    const minCnt=Math.max(8, vis*0.004);
    ctx.save(); ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.font=`700 ${Math.round(12*dpr)}px Inter, system-ui, sans-serif`; ctx.lineJoin='round';
    const order=[]; for(let c=0;c<nCat;c++) order.push(c);
    order.sort((a,b)=>cCnt[a]-cCnt[b]);   // big last → labels on top
    for(const c of order){ if(cCnt[c]<minCnt) continue; if(noneIdx>=0 && c===noneIdx) continue;
      const x=cSumX[c]/cCnt[c], y=cSumY[c]/cCnt[c];
      ctx.lineWidth=Math.round(3.5*dpr); ctx.strokeStyle='rgba(4,6,10,0.95)';
      ctx.strokeText(m.labels[c].name,x,y); ctx.fillStyle=m.labels[c].color; ctx.fillText(m.labels[c].name,x,y);
    }
    ctx.restore();
  }

  if(S.selected>=0 && passes(S.selected)){ const i=S.selected;
    const sx=S.px[i]*S.view.scale+S.width/2+S.view.tx, sy=S.py[i]*S.view.scale+S.height/2+S.view.ty;
    ctx.save(); ctx.scale(S.dpr,S.dpr); ctx.strokeStyle='#fff'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(sx,sy,S.pointSize+5,0,6.283); ctx.stroke(); ctx.restore(); }

  $('info').textContent=`${fmt(vis)} / ${fmt(S.N)} factoids · ${m.title} · zoom ${S.view.scale.toFixed(0)}×`;
  renderFloatLegend(m);
}

// --------------- pick / hover / details ---------------
function pickAt(e){
  const r=S.canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const rad=Math.max(5,S.pointSize+4), r2=rad*rad;
  let best=-1,bd=r2; const sc=S.view.scale,tx=S.view.tx,ty=S.view.ty,W2=S.width/2,H2=S.height/2;
  for(let i=0;i<S.N;i++){ if(!passes(i)) continue;
    const dx=S.px[i]*sc+W2+tx-mx, dy=S.py[i]*sc+H2+ty-my, d=dx*dx+dy*dy;
    if(d<bd){bd=d;best=i;} }
  return best;
}
function labelOf(modeKey,i){ const m=S.modes.find(x=>x.key===modeKey); const l=m.labels[m.values[i]]; return l?l.name:'—'; }
function hover(e){
  const i=pickAt(e), tt=$('tt');
  if(i<0){ S.hoverIdx=-1; tt.style.display='none'; return; }
  if(i!==S.hoverIdx){ S.hoverIdx=i;
    const src=S.manifest.sources[S.attr.src[i]];
    const _d=docAt(i); const txt=_d?_d.t:'(loading factoid text…)';
    tt.innerHTML=`<div>${esc(txt)}</div><div style="margin-top:5px;color:#93a0b2;font-size:.72rem">`+
      `${src?src.name:'?'} · ${S.attr.year[i]||'—'} · ${esc(labelOf('drug_class',i))} · ${esc(labelOf('biomarker',i))} · ${esc(labelOf('setting',i))}</div>`;
    tt.style.display='block';
  }
  const r=S.canvas.getBoundingClientRect(); let x=e.clientX-r.left+14, y=e.clientY-r.top+14;
  const b=tt.getBoundingClientRect(); if(x+b.width>S.width) x=e.clientX-r.left-b.width-14; if(y+b.height>S.height) y=e.clientY-r.top-b.height-14;
  tt.style.left=x+'px'; tt.style.top=y+'px';
}
function openDetails(i){ S.selected=i; renderDetails(i); requestDraw(); }
function renderDetails(i){
  const src=S.manifest.sources[S.attr.src[i]], doc=docAt(i);
  const chip=(modeKey)=>{ const m=S.modes.find(x=>x.key===modeKey); if(!m||!m.labels) return '';
    const l=m.labels[m.values[i]]; if(!l) return '';
    const col=l.color; return `<span class="tag" style="background:${col}22;color:${col};border:1px solid ${col}55">${esc(m.title)}: ${esc(l.name)}</span>`; };
  const dimChips = S.manifest.dimensions.map(d=>chip(d.key)).join('');
  $('det-body').innerHTML=
    `<span class="tag" style="background:${src.color}22;color:${src.color};border:1px solid ${src.color}55">${src.name}</span>`+
    `<div class="quote">${esc(doc&&doc.t?doc.t:'(loading…)')}</div>`+
    dimChips+
    (S.attr.year[i]?`<div class="kv"><span class="k">Year</span><span>${S.attr.year[i]}</span></div>`:'')+
    (doc&&doc.x?`<div class="kv"><span class="k">Document</span><span>${esc(doc.x)}</span></div>`:'');
  $('details').hidden=false;
}
// doc accessor: single docs.json (small N) OR lazy-loaded shards (large N)
function docAt(i){
  if(!S.docsShards) return S.docs ? S.docs[i] : null;
  const sh=(i/S.docShardSize)|0, arr=S.docCache.get(sh);
  if(arr) return arr[i - sh*S.docShardSize];
  loadShard(sh); return null;
}
function loadShard(sh, cb){
  if(S.docCache.has(sh)){ if(cb) cb(); return; }
  if(S.docLoading.has(sh)) return;
  S.docLoading.add(sh);
  fetchJSON(`${DATA}docs/${sh}.json`).then(a=>{ S.docCache.set(sh,a); S.docLoading.delete(sh);
    if(S.selected>=0 && (S.selected/S.docShardSize|0)===sh) renderDetails(S.selected);
    if(cb) cb();
  }).catch(()=>{ S.docLoading.delete(sh); });
}
// single-file docs (used only when not sharded)
function ensureDocs(cb){ if(S.docs||S.docsLoading){ if(S.docs&&cb) cb(); return; }
  S.docsLoading=true; $('search-hint').textContent='loading factoid text…';
  fetchJSON(DATA+'docs.json',p=>$('search-hint').textContent=`loading text… ${(p*100)|0}%`).then(d=>{
    S.docs=d; S.docsLoading=false; $('search-hint').textContent=`${fmt(d.length)} factoids — search enabled`;
    if(cb) cb();
  }).catch(()=>{S.docsLoading=false; $('search-hint').textContent='text load failed';});
}
// make text searchable: load docs.json (single) or ALL shards (sharded, opt-in)
function ensureSearchable(cb){
  if(!S.docsShards){ ensureDocs(cb); return; }
  if(S.docsAllLoaded){ if(cb) cb(); return; }
  const n=S.docsShards.count; let done=0;
  $('search-hint').textContent=`loading text for search (${n} shards)…`;
  for(let s=0;s<n;s++){
    if(S.docCache.has(s)){ done++; if(done===n){S.docsAllLoaded=true; $('search-hint').textContent=`${fmt(S.N)} factoids — search ready`; if(cb)cb();} continue; }
    loadShard(s, ()=>{ done++; $('search-hint').textContent=`loading text… ${done}/${n}`;
      if(done===n){ S.docsAllLoaded=true; $('search-hint').textContent=`${fmt(S.N)} factoids — search ready`; if(cb) cb(); } });
  }
}

// --------------- legends + sidebar ---------------
function renderFloatLegend(m){
  const el=$('legend-float');
  if(m.kind==='year'){ const [y0,y1]=S.manifest.yearRange;
    el.innerHTML=`<div class="ti">${m.title}</div><div style="height:9px;border-radius:3px;background:linear-gradient(90deg,rgb(58,84,110),rgb(232,23,127))"></div>`+
      `<div style="display:flex;justify-content:space-between;margin-top:3px"><span>${y0}</span><span>${y1}</span></div>`; return; }
  if(m.kind==='combined'){
    el.innerHTML=`<div class="ti">${m.title}</div>`+S.cdims.map(k=>
      `<div class="lr"><span class="dot" style="background:${FAMILY_REP[k]};color:${FAMILY_REP[k]}"></span>${FAMILY_NAME[k]}</div>`).join('')
      +`<div style="font-size:.66rem;color:#93a0b2;margin-top:4px">shade = specific category · grey = none</div>`;
    return; }
  const top=m.labels.map((l,i)=>({l,i,c:(S.catCounts[m.key]||[])[i]||0})).sort((a,b)=>b.c-a.c).slice(0,10);
  el.innerHTML=`<div class="ti">${m.title}</div>`+top.map(o=>
    `<div class="lr"><span class="dot" style="background:${o.l.color};color:${o.l.color}"></span>${esc(o.l.name)}</div>`).join('');
}
function renderLegend(){
  const m=mode(), box=$('legend');
  if(m.kind==='year'){ box.innerHTML='<div class="mini">continuous — use the Year sliders to filter</div>'; return; }
  if(m.kind==='combined'){
    const dc=new Int32Array(4); for(let i=0;i<S.N;i++) dc[S.domDim[i]]++;
    box.innerHTML='';
    S.cdims.forEach((k,di)=>{ const fam=di+1;
      const row=document.createElement('div'); row.className='lrow'+(S.famActive.has(fam)?'':' off');
      row.innerHTML=`<span class="dot" style="background:${FAMILY_REP[k]};color:${FAMILY_REP[k]}"></span>`+
        `<span class="nm">${FAMILY_NAME[k]}</span><span class="cn">${fmt(dc[fam])}</span>`;
      row.onclick=()=>{ S.famActive.has(fam)?S.famActive.delete(fam):S.famActive.add(fam); renderLegend(); requestDraw(); };
      box.appendChild(row);
    });
    const note=document.createElement('div'); note.className='mini'; note.style.marginTop='6px';
    note.innerHTML=`each dot = its most-specific category · shade = which one<br>grey = none on all three (${fmt(dc[0])})`;
    box.appendChild(note);
    return; }
  const counts=S.catCounts[m.key]||[];
  box.innerHTML='';
  m.labels.forEach((l,i)=>{
    const row=document.createElement('div'); row.className='lrow'+(S.activeCat.has(i)?'':' off');
    row.innerHTML=`<span class="dot" style="background:${l.color};color:${l.color}"></span>`+
      `<span class="nm" title="${esc(l.name)}">${esc(l.name)}</span><span class="cn">${fmt(counts[i]||0)}</span>`;
    row.onclick=()=>{ S.activeCat.has(i)?S.activeCat.delete(i):S.activeCat.add(i); renderLegend(); requestDraw(); };
    box.appendChild(row);
  });
}
function resetActiveCat(){ const m=mode(); S.activeCat=new Set(); if(m.kind!=='year') m.labels.forEach((_,i)=>S.activeCat.add(i)); }

function buildSidebar(){
  const mb=$('mode'); mb.innerHTML='';
  S.modes.forEach((m,idx)=>{ const b=document.createElement('button'); b.textContent=m.title;
    b.className=idx===S.modeIdx?'active':''; b.onclick=()=>{ S.modeIdx=idx;
      mb.querySelectorAll('button').forEach((x,k)=>x.classList.toggle('active',k===idx));
      $('src-block').hidden=(m.key==='source'); resetActiveCat(); renderLegend(); requestDraw(); };
    mb.appendChild(b); });

  const sb=$('src-list'); sb.innerHTML='';
  S.manifest.sources.filter(s=>s.count).forEach(s=>{ const row=document.createElement('div'); row.className='lrow';  // ponytail: hide empty sources (ASCO)
    row.innerHTML=`<span class="dot" style="background:${s.color};color:${s.color}"></span><span class="nm">${s.name}</span><span class="cn">${fmt(s.count)}</span>`;
    row.onclick=()=>{ S.srcFilter.has(s.id)?S.srcFilter.delete(s.id):S.srcFilter.add(s.id); row.classList.toggle('off'); requestDraw(); };
    sb.appendChild(row); });

  $('cat-all').onclick=()=>{ const m=mode(); if(m.kind!=='year'){ m.labels.forEach((_,i)=>S.activeCat.add(i)); renderLegend(); requestDraw(); } };
  $('cat-none').onclick=()=>{ S.activeCat.clear(); renderLegend(); requestDraw(); };
  $('src-all').onclick=()=>{ S.manifest.sources.forEach(s=>S.srcFilter.add(s.id)); buildSrcRows(); requestDraw(); };
  $('src-none').onclick=()=>{ S.srcFilter.clear(); buildSrcRows(); requestDraw(); };
  function buildSrcRows(){ const vis=S.manifest.sources.filter(s=>s.count); $('src-list').querySelectorAll('.lrow').forEach((row,k)=>row.classList.toggle('off',!S.srcFilter.has(vis[k].id))); }

  // cross-dimension filter: pick codes from each dimension; a point shows only
  // if it matches EVERY dimension you've narrowed (drug ∧ biomarker ∧ setting).
  // --- cross-dimension query builder ---
  const refresh=()=>{ buildXfilter(); renderQueryBar(); requestDraw(); };
  buildXfilter(); renderQueryBar();
  $('xf-clear').onclick=()=>{ S.query.length=0; S.qops.length=0; refresh(); };

  function qIndex(dim,ci){ return S.query.findIndex(q=>q.dim===dim && q.code===ci); }
  function addQ(dim,ci,name,color){ if(qIndex(dim,ci)>=0) return;
    S.query.push({dim,code:ci,name,color}); if(S.query.length>1) S.qops.push('and'); }
  function removeQ(dim,ci){ const k=qIndex(dim,ci); if(k<0) return;
    S.query.splice(k,1); S.qops.splice(k>0?k-1:0,1); }

  function buildXfilter(){
    const box=$('xfilter'); if(!box) return; box.innerHTML='';
    for(const d of S.manifest.dimensions){
      const h=document.createElement('div'); h.className='xf-head'; h.textContent=d.title; box.appendChild(h);
      d.labels.forEach((l,ci)=>{
        const on=qIndex(d.key,ci)>=0;
        const row=document.createElement('div'); row.className='lrow'+(on?' xsel':'');
        row.innerHTML=`<span class="dot" style="background:${l.color};color:${l.color}"></span><span class="nm">${l.name}</span>`;
        row.onclick=()=>{ on?removeQ(d.key,ci):addQ(d.key,ci,l.name,l.color); refresh(); };
        box.appendChild(row);
      });
    }
  }
  function renderQueryBar(){
    const bar=$('querybar'); if(!bar) return; bar.innerHTML='';
    bar.style.display=S.query.length?'flex':'none';
    S.query.forEach((q,j)=>{
      if(j>0){ const op=document.createElement('button'); op.className='qop '+S.qops[j-1];
        op.textContent=S.qops[j-1].toUpperCase();
        op.onclick=()=>{ S.qops[j-1]=S.qops[j-1]==='and'?'or':'and'; renderQueryBar(); requestDraw(); };
        bar.appendChild(op); }
      const chip=document.createElement('span'); chip.className='qchip';
      chip.innerHTML=`<span class="dot" style="background:${q.color}"></span>${esc(q.name)}<b class="qx">×</b>`;
      chip.querySelector('.qx').onclick=()=>{ removeQ(q.dim,q.code); refresh(); };
      bar.appendChild(chip);
    });
  }

  $('hide-none').onchange=e=>{ S.hideNone=e.target.checked; requestDraw(); };
  $('show-labels').onchange=e=>{ S.showLabels=e.target.checked; requestDraw(); };

  const ymin=$('yr-min'),ymax=$('yr-max'),[y0,y1]=S.manifest.yearRange;
  ymin.min=ymax.min=y0; ymin.max=ymax.max=y1; ymin.value=S.yMin; ymax.value=S.yMax;
  const lbl=()=>$('yr-lbl').textContent=`${ymin.value}–${ymax.value}`; lbl();
  const onYr=()=>{ let lo=+ymin.value,hi=+ymax.value; if(lo>hi)[lo,hi]=[hi,lo]; S.yMin=lo; S.yMax=hi; lbl(); requestDraw(); };
  ymin.oninput=onYr; ymax.oninput=onYr;

  $('psize').oninput=e=>{S.pointSize=+e.target.value; requestDraw();};
  $('alpha').oninput=e=>{S.alpha=+e.target.value; requestDraw();};
  $('reset').onclick=()=>{ fitView(); requestDraw(); };

  const se=$('search'); se.disabled=true; se.placeholder='click to load text & search…';
  se.disabled=false; se.placeholder='search factoid text…';
  se.addEventListener('focus',()=>ensureSearchable());
  let t=null; se.addEventListener('input',e=>{ if(t)clearTimeout(t); t=setTimeout(()=>{applySearch(e.target.value); requestDraw();},200); });

  $('det-close').onclick=()=>{$('details').hidden=true; S.selected=-1; requestDraw();};
  $('meth-btn').onclick=()=>$('meth').dataset.open='1';
  $('meth-close').onclick=()=>$('meth').dataset.open='0';
  $('meth').onclick=e=>{ if(e.target===$('meth')) $('meth').dataset.open='0'; };
  addEventListener('keydown',e=>{ if(e.key==='Escape'){ $('meth').dataset.open='0'; $('details').hidden=true; S.selected=-1; requestDraw(); } });
}
function applySearch(q){ q=(q||'').trim().toLowerCase();
  const ready = S.docsShards ? S.docsAllLoaded : !!S.docs;
  if(!q){ S.searchMask=null; $('search-hint').textContent=''; return; }
  if(!ready){ S.searchMask=null; ensureSearchable(()=>{applySearch(q); requestDraw();}); return; }
  const mask=new Uint8Array(S.N); let hits=0;
  for(let i=0;i<S.N;i++){ const d=docAt(i); if(d&&d.t&&d.t.toLowerCase().includes(q)){ mask[i]=1; hits++; } }
  S.searchMask=mask; $('search-hint').textContent=`${fmt(hits)} match${hits===1?'':'es'}`;
}

// --------------- boot ---------------
async function boot(){
  try{
    await loadData();
    $('stat').textContent=`${fmt(S.N)} factoids · ${S.manifest.sources.filter(s=>s.count).length} sources`;
    resetActiveCat(); buildSidebar(); setupCanvas(); renderLegend(); draw();
    if(!S.docsShards) ensureDocs();   // single-file: load eagerly. sharded: lazy on hover.
    const l=$('loader'); l.classList.add('hide'); setTimeout(()=>l.remove(),400);
  }catch(err){ console.error(err); $('loader').textContent='error: '+err.message; }
}
boot();
})();
