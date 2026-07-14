// Headless harness: mock DOM/canvas/fetch, run app.js's boot(), surface runtime errors.
const fs = require('fs');
const path = require('path');
const WEB = __dirname;

let hadError = false;
const origErr = console.error.bind(console);
console.error = (...a) => { hadError = true; origErr('APP console.error:', ...a); };
process.on('unhandledRejection', (e) => { hadError = true; origErr('UNHANDLED REJECTION:', e && e.stack || e); });

// ---- mock element ----
function mockEl(id) {
  const el = {
    id, style: {}, dataset: {}, _html: '',
    classList: { add(){}, remove(){}, toggle(){} },
    addEventListener(){}, appendChild(){}, removeChild(){}, remove(){},
    querySelectorAll(){ return []; }, closest(){ return null; }, focus(){},
    getBoundingClientRect(){ return {width:1200,height:800,left:0,top:0,right:1200,bottom:800}; },
    set innerHTML(v){ this._html=v; }, get innerHTML(){ return this._html; },
    textContent:'', hidden:false, disabled:false, placeholder:'', value:'2', min:'', max:'',
    width:0, height:0,
  };
  if (id === 'canvas') {
    el.getContext = () => ctxMock();
  }
  return el;
}
function ctxMock() {
  const noop = () => {};
  return new Proxy({
    createImageData: (w,h) => ({ data: new Uint8ClampedArray(w*h*4), width:w, height:h }),
    putImageData: noop, fillRect: noop, strokeRect: noop, clearRect: noop,
    fillText: noop, strokeText: noop, beginPath: noop, arc: noop, stroke: noop, fill: noop,
    moveTo: noop, lineTo: noop, save: noop, restore: noop, scale: noop, translate: noop,
    measureText: () => ({ width: 10 }),
  }, { get(t,p){ return p in t ? t[p] : undefined; }, set(t,p,v){ t[p]=v; return true; } });
}

const els = {};
global.document = {
  getElementById: (id) => (els[id] ||= mockEl(id)),
  createElement: () => mockEl('dyn'),
  addEventListener(){},
};
global.window = global;
global.addEventListener = () => {};
global.devicePixelRatio = 1;
global.requestAnimationFrame = (cb) => { try { cb(); } catch(e){ hadError=true; origErr('rAF cb threw:', e.stack||e);} return 0; };
global.Blob = class { constructor(){} async text(){ return ''; } };

// ---- fetch mock reading files from disk ----
global.fetch = async (u) => {
  let rel = u.replace(/^\.?\//, '').split('?')[0];   // './data/x?_=1' -> 'data/x'
  const p = path.join(WEB, rel);
  const buf = fs.readFileSync(p);
  return {
    ok: true, status: 200,
    headers: { get: () => null },
    body: null,                                  // forces fetchJSON to use json()
    json: async () => JSON.parse(buf.toString('utf8')),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
};

// ---- run app.js ----
const code = fs.readFileSync(path.join(WEB, 'app.js'), 'utf8');
try {
  eval(code);
} catch (e) {
  hadError = true; origErr('SYNC EVAL THREW:', e.stack || e);
}

// boot() is async; wait, then report
setTimeout(() => {
  const loader = els['loader'];
  const stat = els['stat'];
  const info = els['info'];
  console.log('--- result ---');
  console.log('loader text:', loader && loader.textContent);
  console.log('stat text  :', stat && stat.textContent);
  console.log('info text  :', info && info.textContent);
  console.log(hadError ? 'RESULT: ERROR(S) DETECTED ABOVE' : 'RESULT: boot completed with no runtime errors');
  process.exit(hadError ? 1 : 0);
}, 2500);
