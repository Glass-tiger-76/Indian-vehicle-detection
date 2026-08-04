const log = (msg) => document.getElementById('log').textContent += msg + '\n';

let session = null;

async function init() {
  try {
    session = await ort.InferenceSession.create('best.onnx');
    status('Ready — choose an image');
  } catch (e) {
    status('Failed to load model: ' + e.message, true);
  }
}

function status(msg, isError = false) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = isError ? 'error' : '';
}

function showResults(dets) {
  const counts = {};
  for (const d of dets) counts[NAMES[d.cls]] = (counts[NAMES[d.cls]] || 0) + 1;

  const rows = dets.map(d =>
    `<tr><td>${NAMES[d.cls]}</td>
         <td>${(d.score*100).toFixed(1)}%</td>
         <td><div class="bar" style="width:${d.score*100}%"></div></td></tr>`
  ).join('');

  const summary = Object.entries(counts)
    .sort((a,b) => b[1]-a[1])
    .map(([k,v]) => `${v}× ${k}`).join(', ');

  document.getElementById('results').innerHTML =
    `<p style="font-size:14px"><strong>${summary || 'nothing detected'}</strong></p>
     <table><tr><th>Class</th><th>Conf</th><th></th></tr>${rows}</table>`;
}

const SIZE = 640;
let scale, padLeft, padTop;

function letterbox(img) {
  const c = document.createElement('canvas');
  c.width = c.height = SIZE;
  const ctx = c.getContext('2d');

  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, SIZE, SIZE);

  scale = SIZE / Math.max(img.width, img.height);
  const nw = Math.round(img.width * scale);
  const nh = Math.round(img.height * scale);
  padLeft = Math.floor((SIZE - nw) / 2);
  padTop  = Math.floor((SIZE - nh) / 2);

  ctx.drawImage(img, padLeft, padTop, nw, nh);
  return c;
}

const NUM = 8400, NCLS = 14;
const NAMES = ['trak','cyclist','bike','tempo','car','zeep','toto','e-rickshaw',
               'auto-rickshaw','bus','van','cycle-rickshaw','person','taxi'];

function decode(out, conf = 0.25) {
  const d = out.data;
  const dets = [];

  for (let i = 0; i < NUM; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < NCLS; c++) {
      const s = d[(4 + c) * NUM + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < conf) continue;

    const cx = d[i], cy = d[NUM + i], w = d[2*NUM + i], h = d[3*NUM + i];

    dets.push({
      x1: (cx - w/2 - padLeft) / scale,
      y1: (cy - h/2 - padTop)  / scale,
      x2: (cx + w/2 - padLeft) / scale,
      y2: (cy + h/2 - padTop)  / scale,
      score: best, cls
    });
  }
  return dets;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2-x1) * Math.max(0, y2-y1);
  const areaA = (a.x2-a.x1) * (a.y2-a.y1);
  const areaB = (b.x2-b.x1) * (b.y2-b.y1);
  return inter / (areaA + areaB - inter);
}

function nms(dets, thresh = 0.45) {
  dets.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of dets) {
    if (!keep.some(k => k.cls === d.cls && iou(k, d) > thresh)) keep.push(d);
  }
  return keep;
}

async function runInference(canvas,img) {
  const ctx = canvas.getContext('2d');
  const {data} = ctx.getImageData(0, 0, SIZE, SIZE);

  const area = SIZE * SIZE;
  const input = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    input[i]          = data[i*4]     / 255;
    input[i + area]   = data[i*4 + 1] / 255;
    input[i + area*2] = data[i*4 + 2] / 255;
  }

  const tensor = new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]);

  const t0 = performance.now();
  const results = await session.run({[session.inputNames[0]]: tensor});
  const ms = performance.now() - t0;

  const out = results[session.outputNames[0]];
  const dets = nms(decode(out));

  status(`${ms.toFixed(0)}ms · ${dets.length} detections`);
  showResults(dets);
  

  const ot = document.getElementById('out');
  ot.style.display='block';
  ot.width = img.width;
  ot.height = img.height;
  const c = ot.getContext('2d');
  c.drawImage(img, 0, 0);

  c.lineWidth = Math.max(2, img.width / 400);
  c.font = `${Math.max(14, img.width / 60)}px sans-serif`;

  for (const d of dets) {
    c.strokeStyle = 'red';
    c.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    c.fillStyle = 'red';
    c.fillText(`${NAMES[d.cls]} ${d.score.toFixed(2)}`, d.x1, d.y1 - 4);
  }
}

document.getElementById('upload').addEventListener('change', (e) => {
  const file = e.target.files[0];
  const img = new Image();
  img.onload = () => {
    
    const boxed = letterbox(img);
    runInference(boxed,img);

    
  
  };
  img.src = URL.createObjectURL(file);
});

init();

