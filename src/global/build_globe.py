"""MOCRE-GLOBAL — genera globe/index.html: globo 3D (proyección ortográfica en canvas, cero
dependencias) con los micro-puntos de previsión. v3 UI (7-jul):
  - fondo de campo de partículas ondulante (procedural, estética de la textura de referencia de
    Matt: olas de puntos neón azul→rojo sobre negro), animado
  - cursor interactivo: efecto lente (los puntos cercanos crecen y brillan) + tooltip flotante
  - panel de ENTRADAS por punto (los 8 canales del motor con barra fuzzy + término dominante
    calculado en vivo con SETS5) — verificabilidad de que las entradas se consideran
  - paneles glass inmersivos (blur, bordes luminosos, glow, animación de entrada)
La geometría de costas va embebida; los DATOS vivos se cargan de globe_data.json (auto-refresh).
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GLOBE = os.path.join(ROOT, "globe")
os.makedirs(GLOBE, exist_ok=True)


def load_world():
    world = json.load(open(os.path.join(ROOT, "map", "world_land.json")))
    rings = []
    for poly in world:
        for ring in poly:
            r = ring[::2] if len(ring) > 60 else ring
            if len(r) >= 3:
                rings.append([[round(p[0], 2), round(p[1], 2)] for p in r])
    return rings


def load_universes():
    path = os.path.join(ROOT, "out", "validation", "fuzzy_universe_manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOCRE · Globo de previsión sísmica</title>
<style>
  :root{--bg:#04050a;--txt:#e6ecf4;--dim:#8494a8;--line:rgba(90,120,160,.16);
        --glass:rgba(10,14,22,.55);--glass2:rgba(14,19,30,.72);
        --blue:#2e6bff;--red:#ff2d55;--gold:#e8b24a;--green:#37d67a}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--txt);font:13px/1.5 system-ui,-apple-system,sans-serif;
       overflow:hidden}
  #wrap{position:relative;height:100vh;overflow:hidden;perspective:1400px}
  #stage{position:absolute;inset:0;cursor:crosshair}
  #stage.drag{cursor:grabbing}
  canvas{display:block}
  #hdr{position:absolute;top:16px;left:20px;pointer-events:none;max-width:58%}
  #hdr h1{font-size:16px;letter-spacing:.14em;font-weight:600;
          background:linear-gradient(90deg,#9db8ff,#fff 45%,#ff9db0);
          -webkit-background-clip:text;background-clip:text;color:transparent}
  #hdr .sub{font-size:11px;color:var(--dim);margin-top:5px;max-width:600px}
  #live{position:absolute;top:16px;right:16px;font-size:11px;color:var(--dim);text-align:right;
        background:var(--glass);backdrop-filter:blur(10px);padding:6px 12px;border-radius:20px;
        border:1px solid var(--line)}
  #live .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);
    margin-right:6px;box-shadow:0 0 8px var(--green);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  #legend{position:absolute;bottom:16px;left:20px;font-size:10px;color:var(--dim);
          background:var(--glass);backdrop-filter:blur(10px);padding:10px 14px;
          border-radius:12px;border:1px solid var(--line)}
  #legend .bar{width:190px;height:7px;border-radius:4px;margin:5px 0;
    background:linear-gradient(90deg,#12324a,#1e6e5c,#7bb13c,#e0b13a,#e06a2b,#d43333,#ff6060,#ffebeb);
    box-shadow:0 0 12px rgba(224,106,43,.25)}
  #tip{position:absolute;pointer-events:none;display:none;z-index:30;
       background:var(--glass2);backdrop-filter:blur(12px);border:1px solid rgba(140,170,220,.3);
       border-radius:10px;padding:7px 11px;font-size:11px;line-height:1.5;
       box-shadow:0 6px 24px rgba(0,0,0,.5)}
  #tip b{font-size:13px}
  #links{position:absolute;inset:0;z-index:13;pointer-events:none;mix-blend-mode:screen}
  #panel{position:absolute;right:22px;top:64px;bottom:22px;width:min(560px,38vw);z-index:12;
    background:linear-gradient(180deg,rgba(9,13,22,.84),rgba(5,8,15,.94));
    backdrop-filter:blur(18px);border:1px solid rgba(130,170,230,.22);border-radius:8px;
    padding:16px;overflow-y:auto;box-shadow:-34px 0 80px rgba(0,0,0,.38),0 0 48px rgba(70,120,255,.08);
    transform:rotateY(-7deg) translateZ(26px);transform-origin:left center}
  #panel::-webkit-scrollbar{width:5px}
  #panel::-webkit-scrollbar-thumb{background:#26344a;border-radius:3px}
  #panel h2{font-size:14px;letter-spacing:.06em}
  #panel .rg{font-size:11px;color:var(--dim);margin-bottom:12px}
  .axis{position:relative;border:1px solid var(--line);border-radius:12px;padding:11px 13px;
        margin-bottom:10px;background:linear-gradient(135deg,rgba(46,107,255,.05),rgba(255,45,85,.04));
        animation:reveal .45s cubic-bezier(.2,.8,.2,1) both}
  .axis:nth-child(1){animation-delay:.02s}.axis:nth-child(2){animation-delay:.07s}
  .axis:nth-child(3){animation-delay:.12s}.axis:nth-child(4){animation-delay:.17s}
  .axis:nth-child(5){animation-delay:.22s}.axis:nth-child(6){animation-delay:.27s}
  @keyframes reveal{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
  .axis .k{font-size:10px;letter-spacing:.1em;color:#7fb2e8;text-transform:uppercase}
  .axis .big{font-size:22px;font-weight:650;margin-top:2px;text-shadow:0 0 18px rgba(120,160,255,.25)}
  .geo-title{font-size:20px;font-weight:700;letter-spacing:.01em;margin-top:2px}
  .geo-meta{font-size:11px;color:var(--dim);line-height:1.45;margin-top:4px}
  .rows div{display:flex;justify-content:space-between;font-size:12px;padding:2px 0}
  .rows span{color:var(--dim)}
  .note{font-size:10px;color:var(--dim);margin-top:6px;line-height:1.5}
  .in-row{display:grid;grid-template-columns:74px 1fr 92px;gap:8px;align-items:center;
          padding:3px 0;font-size:10.5px}
  .in-name{color:var(--dim);letter-spacing:.02em}
  .in-track{position:relative;height:8px;border-radius:4px;
    background:linear-gradient(90deg,rgba(46,107,255,.5),rgba(90,100,130,.25) 50%,rgba(255,45,85,.5));
    box-shadow:inset 0 1px 3px rgba(0,0,0,.5)}
  .in-track::before{content:'';position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;
    background:rgba(255,255,255,.18)}
  .in-dot{position:absolute;top:50%;width:10px;height:10px;border-radius:50%;
    transform:translate(-50%,-50%);background:#fff;
    box-shadow:0 0 8px 2px rgba(255,255,255,.55)}
  .in-term{text-align:right;color:var(--txt);font-size:10px}
  .in-term small{color:var(--dim)}
  .audit-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:10px}
  .audit-head h2{font-size:15px;letter-spacing:.08em;text-transform:uppercase}
  .audit-mode{font-size:10px;color:#9fc1ff;border:1px solid rgba(120,160,255,.22);border-radius:999px;
    padding:3px 8px;background:rgba(70,120,255,.08);white-space:nowrap}
  .audit-block{border:1px solid rgba(100,130,180,.18);border-radius:8px;padding:10px 11px;margin:9px 0;
    background:linear-gradient(135deg,rgba(28,44,70,.32),rgba(12,16,26,.62))}
  .audit-block h3{font-size:10px;letter-spacing:.12em;color:#7fb2e8;text-transform:uppercase;margin-bottom:7px}
  .audit-row{position:relative;border-top:1px solid rgba(100,130,180,.14);padding:8px 0 7px}
  .audit-row:first-of-type{border-top:0}
  .audit-row .r0{display:grid;grid-template-columns:112px 1fr auto;gap:8px;align-items:baseline}
  .audit-row code{font:600 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;color:#eaf2ff}
  .audit-row .src{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}
  .audit-row .val{font:650 12px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace}
  .audit-row .dom{font-size:9.5px;color:var(--dim);text-align:right}
  .u-track{position:relative;height:13px;border-radius:7px;margin:7px 0 5px;overflow:hidden;
    background:linear-gradient(90deg,rgba(46,107,255,.38),rgba(160,170,190,.18),rgba(255,45,85,.42));
    box-shadow:inset 0 1px 4px rgba(0,0,0,.55)}
  .u-track i{position:absolute;top:0;bottom:0;border-left:1px solid rgba(255,255,255,.24)}
  .u-track b{position:absolute;top:50%;width:10px;height:10px;border-radius:50%;transform:translate(-50%,-50%);
    background:#fff;box-shadow:0 0 10px rgba(255,255,255,.85),0 0 18px rgba(120,170,255,.45)}
  .u-sets{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:3px}
  .u-sets.many{grid-template-columns:repeat(6,minmax(0,1fr))}
  .u-set{min-width:0;border:1px solid rgba(120,150,200,.14);border-radius:5px;padding:3px 4px;
    background:rgba(255,255,255,.025);font-size:8px;color:#8ea1b8;line-height:1.25}
  .u-set strong{display:block;color:#d9e7ff;font-size:8.5px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .u-set.hotset{border-color:rgba(255,255,255,.42);background:rgba(255,255,255,.08);box-shadow:0 0 16px rgba(120,170,255,.16)}
  .raw-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}
  .raw-grid div{border:1px solid rgba(120,150,200,.14);border-radius:6px;padding:6px;background:rgba(255,255,255,.025)}
  .raw-grid span{display:block;color:var(--dim);font-size:9px;text-transform:uppercase;letter-spacing:.08em}
  .raw-grid b{font:650 12px ui-monospace,SFMono-Regular,Menlo,monospace}
  #top h3{font-size:11px;letter-spacing:.1em;color:#7fb2e8;text-transform:uppercase;margin:14px 0 6px}
  .hot{display:flex;justify-content:space-between;padding:6px 10px;border-radius:8px;cursor:pointer;
    font-size:12px;transition:background .15s,transform .15s;border:1px solid transparent}
  .hot:hover{background:rgba(46,107,255,.1);border-color:var(--line);transform:translateX(3px)}
  .hot b{color:var(--gold);text-shadow:0 0 10px rgba(232,178,74,.4)}
  #empty{color:var(--dim);font-size:12px;margin-top:8px}
</style></head><body>
<div id="wrap">
  <div id="stage">
    <canvas id="cv"></canvas>
    <div id="hdr">
      <h1>MOCRE · GLOBO DE PREVISIÓN SÍSMICA</h1>
      <div class="sub" id="subtitle">cargando datos…</div>
    </div>
    <div id="live"><span class="dot"></span><span id="livetxt">—</span></div>
	    <div id="legend">P(M≥5 · 30 días · ~150km)<div class="bar"></div>
	      <span>0.1%</span> · <span>3%</span> · <span>10%</span> · <span>40%</span> · <span>90%+</span><br>
	      ◎ rojo pulsante = señal fuzzy experimental · ○ blanco = sismo real M≥4.5, 7d</div>
	    <div id="tip"></div>
	  </div>
	  <svg id="links"></svg>
	  <div id="panel">
    <div class="audit-head"><h2>Micro-punto</h2><span class="audit-mode">inspector de universos</span></div>
    <div class="rg">click en un punto del globo · pasa el cursor para explorar</div>
    <div id="detail"><div id="empty">Cada punto es una celda de ~111 km con sus PROPIAS entradas
      (8 canales fuzzy medidos de la sismicidad real de su vecindad). Selecciona un punto para ver
      entradas y salidas del motor. Arrastra para rotar · rueda para zoom.</div></div>
    <div id="top"><h3>Más calientes ahora</h3><div id="hotlist"></div></div>
  </div>
</div>
<script>
const COAST = __COAST__;
const UNIVERSES = __UNIVERSES__;
// partición compartida del motor (fallback para calcular EN VIVO el término fuzzy dominante por entrada)
const SETS5 = {muy_bajo:[-1,-1,-0.70,-0.35], bajo:[-0.70,-0.35,0], normal:[-0.35,0,0.35],
               alto:[0,0.35,0.70], muy_alto:[0.35,0.70,1,1]};
const CITY_INDEX = [
  ['Sand Point','Alaska Peninsula','United States',55.34,-160.5],['Anchorage','Alaska','United States',61.22,-149.9],
  ['Vancouver','British Columbia','Canada',49.28,-123.12],['Seattle','Washington','United States',47.61,-122.33],
  ['Portland','Oregon','United States',45.52,-122.68],['San Francisco','California','United States',37.77,-122.42],
  ['Parkfield','Central California','United States',35.9,-120.43],['Los Angeles','Southern California','United States',34.05,-118.24],
  ['Anza','Southern California','United States',33.56,-116.67],['Mexico City','Central Mexico','Mexico',19.43,-99.13],
  ['Oaxaca','Oaxaca','Mexico',17.07,-96.72],['Guatemala City','Guatemala','Guatemala',14.63,-90.51],
  ['San Salvador','San Salvador','El Salvador',13.69,-89.19],['Managua','Managua','Nicaragua',12.11,-86.24],
  ['San Jose','Central Valley','Costa Rica',9.93,-84.08],['Quito','Pichincha','Ecuador',-0.18,-78.47],
  ['Lima','Lima','Peru',-12.05,-77.04],['Arequipa','Arequipa','Peru',-16.4,-71.54],
  ['Santiago','Metropolitan Region','Chile',-33.45,-70.66],['Valparaiso','Central Chile','Chile',-33.05,-71.62],
  ['Punta Arenas','Magallanes','Chile',-53.16,-70.91],['Reykjavik','Capital Region','Iceland',64.15,-21.94],
  ['Ponta Delgada','Azores','Portugal',37.74,-25.67],['Tokyo','Kanto','Japan',35.68,139.76],
  ['Sendai','Tohoku','Japan',38.27,140.87],['Sapporo','Hokkaido','Japan',43.06,141.35],
  ['Kochi','Shikoku','Japan',33.56,133.53],['Taipei','Taiwan','Taiwan',25.04,121.56],
  ['Manila','Luzon','Philippines',14.6,120.98],['Davao','Mindanao','Philippines',7.07,125.61],
  ['Jakarta','Java','Indonesia',-6.21,106.85],['Padang','West Sumatra','Indonesia',-0.95,100.35],
  ['Banda Aceh','Aceh','Indonesia',5.55,95.32],['Port Moresby','National Capital District','Papua New Guinea',-9.44,147.18],
  ['Honiara','Guadalcanal','Solomon Islands',-9.43,159.95],['Port Vila','Efate','Vanuatu',-17.73,168.31],
  ['Suva','Viti Levu','Fiji',-18.14,178.44],["Nuku'alofa",'Tongatapu','Tonga',-21.14,-175.2],
  ['Wellington','North Island','New Zealand',-41.29,174.78],['Christchurch','South Island','New Zealand',-43.53,172.63],
  ['Kathmandu','Bagmati','Nepal',27.71,85.32],['Chengdu','Sichuan','China',30.66,104.06],
  ['Lhasa','Tibet','China',29.65,91.12],['Tehran','Tehran Province','Iran',35.69,51.39],
  ['Istanbul','Marmara','Turkey',41.01,28.98],['Athens','Attica','Greece',37.98,23.73],
  ["L'Aquila",'Abruzzo','Italy',42.35,13.4],['Naples','Campania','Italy',40.85,14.27],
  ['Casablanca','Atlantic Morocco','Morocco',33.57,-7.59],['Addis Ababa','Oromia margin','Ethiopia',8.98,38.79],
  ['Nairobi','Kenya Rift','Kenya',-1.29,36.82],['Antananarivo','Madagascar','Madagascar',-18.88,47.51]
];
function geoDistanceKm(aLat,aLon,bLat,bLon){
  const toRad=Math.PI/180, dLat=(bLat-aLat)*toRad;
  let dLon=(bLon-aLon+540)%360-180; dLon*=toRad;
  const s=Math.sin(dLat/2)**2+Math.cos(aLat*toRad)*Math.cos(bLat*toRad)*Math.sin(dLon/2)**2;
  return 6371*2*Math.atan2(Math.sqrt(s),Math.sqrt(1-s));
}
function oceanRegion(lat, lon){
  const abs=Math.abs(lat);
  if (abs > 60) return lat > 0 ? 'Arctic seismic grid' : 'Southern Ocean seismic grid';
  if (lon > -80 && lon < 20) return 'Atlantic seismic grid';
  if (lon >= 20 && lon < 120) return 'Indian Ocean seismic grid';
  return 'Pacific seismic grid';
}
function nearestPlace(lat, lon){
  let best=null, bd=1e9;
  for (const c of CITY_INDEX){
    const d=geoDistanceKm(lat,lon,c[3],c[4]);
    if (d<bd){bd=d;best=c;}
  }
  const remote = bd > 900;
  return {
    city: best[0],
    region: remote ? oceanRegion(lat, lon) : best[1],
    country: best[2],
    distance_km: Math.round(bd),
    remote
  };
}
function membJS(x,p){
  if(p.length===3){const a=p[0],b=p[1],c=p[2];
    const L=b>a?(x-a)/(b-a):1, R=c>b?(c-x)/(c-b):1;
    return Math.max(0,Math.min(1,Math.min(L,R)));}
  const a=p[0],b=p[1],c=p[2],d=p[3];
  const L=b>a?(x-a)/(b-a):1, R=d>c?(d-x)/(d-c):1;
  return Math.max(0,Math.min(1,Math.min(L,1,R)));
}
function domTerm(v, sets=SETS5){
  let best='normal', bm=-1;
  for(const t in sets){const m=membJS(v,sets[t]); if(m>bm){bm=m;best=t;}}
  return [best,bm];
}
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const stage = document.getElementById('stage'), tip = document.getElementById('tip');
const links = document.getElementById('links');
const panel = document.getElementById('panel');
let W,H,R,GX,GY,AX, lon0 = -20*Math.PI/180, lat0 = 15*Math.PI/180, zoom = 1;
let DATA = null, PTS = [], RECENT = [];
let spin = true;
let mx=-1e4, my=-1e4;                       // cursor (efecto lente)
const LENS = 80;

function resize(){
  W = stage.clientWidth; H = stage.clientHeight;
  cv.width = W*devicePixelRatio; cv.height = H*devicePixelRatio;
  cv.style.width = W+'px'; cv.style.height = H+'px';
  links.setAttribute('viewBox','0 0 '+W+' '+H);
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  const pw = panel ? panel.getBoundingClientRect().width + 46 : 0;
  GX = Math.max(W*0.30, (W-pw)*0.47);
  GY = H*0.52;
  R = Math.max(160, Math.min(W-pw*0.75,H)*0.38);
  AX = GX + Math.max(R*0.82, (W - pw - GX) * 0.56);
}
window.addEventListener('resize', resize);

function proj(sinf, cosf, lam){
  const dl = lam - lon0;
  const cosc = Math.sin(lat0)*sinf + Math.cos(lat0)*cosf*Math.cos(dl);
  if (cosc < 0) return null;
  const x = R*zoom*cosf*Math.sin(dl);
  const y = R*zoom*(Math.cos(lat0)*sinf - Math.sin(lat0)*cosf*Math.cos(dl));
  return [GX + x, GY - y, cosc];
}
function projLL(lat, lon){
  const f = lat*Math.PI/180, l = lon*Math.PI/180;
  return proj(Math.sin(f), Math.cos(f), l);
}
function pColor(p){
  const stops = [[0.001,[18,50,74]],[0.01,[30,110,92]],[0.03,[123,177,60]],
                 [0.10,[224,177,58]],[0.25,[224,106,43]],[0.45,[212,51,51]],
                 [0.70,[255,96,96]],[0.92,[255,235,235]]];
  let c = stops[0][1];
  for (let i=0;i<stops.length-1;i++){
    const p0=stops[i][0], c0=stops[i][1], p1=stops[i+1][0], c1=stops[i+1][1];
    if (p >= p1) { c = c1; continue; }
    if (p >= p0){
      const t = (Math.log(p)-Math.log(p0))/(Math.log(p1)-Math.log(p0));
      c = c0.map((v,k)=>Math.round(v+(c1[k]-v)*t));
    }
  }
  return c;
}

// ---- FONDO: campo de partículas ondulante (estética de la textura de referencia) --------------
const WNX = 96, WNY = 54;
function drawWave(t){
  const gx = W/(WNX-1), gy = H/(WNY-1);
  for(let j=0;j<WNY;j++){
    for(let i=0;i<WNX;i++){
      const u=i/WNX, v=j/WNY;
      const arg1 = u*7 + v*4 + t*0.5;
      const ph = Math.sin(arg1) + 0.6*Math.sin(u*13 - v*9 - t*0.32)
               + 0.35*Math.sin((u+v)*21 + t*0.75);
      const x = i*gx, y = j*gy + ph*14;
      const crest = Math.pow(0.5+0.5*Math.sin(arg1 + 1.3), 3);
      const mixv = Math.min(1, Math.max(0, u + 0.25*Math.sin(v*6+t*0.2)));
      const r = Math.round(30 + mixv*225);
      const g = Math.round(40 + (1-Math.abs(mixv-0.5)*2)*30);
      const b = Math.round(255 - mixv*170);
      ctx.fillStyle = 'rgba('+r+','+g+','+b+','+(0.05 + crest*0.30).toFixed(3)+')';
      const s = 1 + crest*1.3;
      ctx.fillRect(x, y, s, s);
    }
  }
}

let dragging=false, SEL=null;
function draw(){
  const t = Date.now()/1000;
  ctx.clearRect(0,0,W,H);
  drawWave(t);
  // esfera/océano (disco opaco sobre el campo de ondas)
  const g = ctx.createRadialGradient(GX-R*.3, GY-R*.35, R*.2, GX, GY, R*zoom);
  g.addColorStop(0,'#0c1320'); g.addColorStop(1,'#05080f');
  ctx.beginPath(); ctx.arc(GX, GY, R*zoom, 0, 7); ctx.fillStyle = g; ctx.fill();
  ctx.strokeStyle = 'rgba(80,120,180,.35)'; ctx.lineWidth = 1.2; ctx.stroke();
  // halo atmosférico
  const ha = ctx.createRadialGradient(GX,GY,R*zoom*0.98, GX,GY,R*zoom*1.06);
  ha.addColorStop(0,'rgba(70,120,255,.14)'); ha.addColorStop(1,'rgba(70,120,255,0)');
  ctx.beginPath(); ctx.arc(GX,GY,R*zoom*1.06,0,7); ctx.fillStyle=ha; ctx.fill();
  // graticule + costas
  ctx.strokeStyle = 'rgba(50,70,90,.25)'; ctx.lineWidth = .5;
  for (let lat=-60; lat<=60; lat+=30) strokePath(gratRing(lat, true));
  for (let lon=-180; lon<180; lon+=30) strokePath(gratRing(lon, false));
  ctx.strokeStyle = 'rgba(130,160,185,.55)'; ctx.lineWidth = .8;
  for (const ring of COAST) strokePath(ring.map(p=>projLL(p[1],p[0])));
  // micro-puntos con efecto LENTE
  let hover=null, hd=196;
  if (PTS.length){
    const base = Math.max(1.4, 2.2*zoom);
    for (const q of PTS){
      const pr = proj(q.sinf, q.cosf, q.lam);
      if (!pr){ q.sx=undefined; continue; }
      q.sx = pr[0]; q.sy = pr[1];
      const dx=pr[0]-mx, dy=pr[1]-my;
      const d2=dx*dx+dy*dy;
      let sz=base, boost=0;
      if (d2 < LENS*LENS){
        const d=Math.sqrt(d2), k=1-d/LENS;
        sz = base*(1+2.1*k*k); boost=k;
        if (d2<hd){ hd=d2; hover=q; }
      }
      const c=q.col;
      ctx.fillStyle = 'rgb('+c[0]+','+c[1]+','+c[2]+')';
      ctx.globalAlpha = Math.min(1, 0.45 + 0.55*pr[2] + boost*0.5);
      ctx.fillRect(pr[0]-sz/2, pr[1]-sz/2, sz, sz);
      if (boost>0.25){
        ctx.globalAlpha = boost*0.35;
        ctx.fillStyle = '#fff';
        ctx.fillRect(pr[0]-sz*0.75, pr[1]-sz*0.75, sz*1.5, sz*1.5);
      }
      if (q.alert){
        const ph = (t*1.4 + q.la*0.31) % 1;
        ctx.globalAlpha = 1;
        ctx.strokeStyle = 'rgba(255,60,60,'+(0.9*(1-ph)).toFixed(2)+')';
        ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.arc(pr[0], pr[1], 4+ph*10, 0, 7); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }
  // RADAR: sismos reales recientes
  for (const e of RECENT){
    const pr = proj(e.sinf, e.cosf, e.lam);
    if (!pr) continue;
    const phase = ((t*0.9 + e.seed) % 1);
    const rr = 3 + phase*13*Math.min(2,zoom);
    const fresh = Math.max(0, 1 - e.hours/168);
    ctx.strokeStyle = 'rgba(255,255,255,'+(0.75*(1-phase)*(.3+.7*fresh)).toFixed(3)+')';
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(pr[0], pr[1], rr, 0, 7); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,'+(.35+.5*fresh).toFixed(2)+')';
    ctx.beginPath(); ctx.arc(pr[0], pr[1], 1.6+e.mag*0.35, 0, 7); ctx.fill();
  }
  drawAuditBridge(t);
  syncSvgLinks(t);
  // seleccionado: anillo doble animado
  if (SEL && SEL.sx!==undefined){
    ctx.strokeStyle='#fff'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(SEL.sx,SEL.sy,7,0,7); ctx.stroke();
    ctx.strokeStyle='rgba(255,255,255,.35)';
    ctx.beginPath(); ctx.arc(SEL.sx,SEL.sy,11+2*Math.sin(t*3),0,7); ctx.stroke();
  }
  // tooltip flotante
  if (hover && !dragging){
    tip.style.display='block';
    tip.style.left=(hover.sx+16)+'px'; tip.style.top=(hover.sy-14)+'px';
    tip.innerHTML = '<b>'+(hover.p*100).toFixed(1)+'%</b> · M~'+hover.mag.toFixed(1)+'<br>'+
      '<span style="color:var(--dim)">'+hover.place.city+', '+hover.place.country+'<br>'+
      (hover.la+0.5)+'°, '+(hover.lo+0.5)+'°'+
      (hover.alert?' · <span style="color:#ff6b6b">señal fuzzy experimental</span>':'')+'</span>';
  } else tip.style.display='none';
}
function gratRing(v, isLat){
  const pts=[];
  if (isLat){ for(let lon=-180;lon<=180;lon+=4) pts.push(projLL(v,lon)); }
  else { for(let lat=-88;lat<=88;lat+=4) pts.push(projLL(lat,v)); }
  return pts;
}
function strokePath(pts){
  ctx.beginPath(); let pen=false;
  for (const p of pts){
    if (!p){ pen=false; continue; }
    if (pen) ctx.lineTo(p[0],p[1]); else ctx.moveTo(p[0],p[1]);
    pen=true;
  }
  ctx.stroke();
}
function stageXYFromRect(r){
  const sr = stage.getBoundingClientRect();
  return {x:r.left-sr.left, y:r.top-sr.top, w:r.width, h:r.height};
}
function auditAnchors(){
  if (!panel) return [];
  const rows = Array.from(panel.querySelectorAll('.audit-row[data-var]')).filter(el=>{
    const r = el.getBoundingClientRect(), pr = panel.getBoundingClientRect();
    return r.bottom > pr.top + 20 && r.top < pr.bottom - 20;
  }).slice(0, 18);
  return rows.map(el=>{
    const r = stageXYFromRect(el.getBoundingClientRect());
    const v = SEL && SEL.in ? SEL.in[el.dataset.var] : 0;
    return {x:r.x+2, y:r.y+r.h/2, key:el.dataset.var, value:v};
  });
}
function drawAuditBridge(t){
  if (!SEL || SEL.sx===undefined || !panel) return;
  const pr = stageXYFromRect(panel.getBoundingClientRect());
  const axisX = Math.min(pr.x-22, AX || pr.x-38);
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  ctx.setLineDash([8,10]);
  ctx.strokeStyle = 'rgba(145,180,255,.20)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(axisX, 82); ctx.lineTo(axisX, H-42); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(150,185,255,.42)';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText('eje de rotación visual', axisX-58, 72);
  const anchors = auditAnchors();
  for (const a of anchors){
    const hot = Math.max(0, Math.min(1, (Number(a.value)||0) * .5 + .5));
    const r = Math.round(70 + hot*185), g = Math.round(118 + hot*72), b = Math.round(255 - hot*115);
    const pulse = .34 + .18*Math.sin(t*2.0 + a.y*.017);
    ctx.strokeStyle = 'rgba('+r+','+g+','+b+','+pulse.toFixed(3)+')';
    ctx.lineWidth = 1.05;
    ctx.beginPath();
    const c1x = SEL.sx + (axisX-SEL.sx)*0.55, c1y = SEL.sy + Math.sin(a.y*.021+t)*22;
    const c2x = axisX + (a.x-axisX)*0.44, c2y = a.y + Math.cos(a.y*.017+t)*16;
    ctx.moveTo(SEL.sx, SEL.sy);
    ctx.bezierCurveTo(c1x, c1y, c2x, c2y, a.x, a.y);
    ctx.stroke();
    ctx.fillStyle = 'rgba('+r+','+g+','+b+',.55)';
    ctx.beginPath(); ctx.arc(a.x, a.y, 2.2, 0, 7); ctx.fill();
  }
  ctx.restore();
}
function syncSvgLinks(t){
  if (!links) return;
  if (!SEL || SEL.sx===undefined || !panel){ links.innerHTML=''; return; }
  const pr = stageXYFromRect(panel.getBoundingClientRect());
  const axisX = Math.min(pr.x-22, AX || pr.x-38);
  const anchors = auditAnchors().slice(0, 12);
  let svg = '<path d="M '+axisX+' 72 L '+axisX+' '+(H-36)+'" stroke="rgba(150,190,255,.42)" stroke-width="1" stroke-dasharray="8 10" fill="none"/>';
  svg += '<text x="'+(axisX-60)+'" y="61" fill="rgba(185,210,255,.62)" font-size="10" font-family="ui-monospace,monospace">eje visual</text>';
  for (const a of anchors){
    const hot = Math.max(0, Math.min(1, (Number(a.value)||0) * .5 + .5));
    const r = Math.round(80 + hot*175), g = Math.round(128 + hot*82), b = Math.round(255 - hot*105);
    const op = .46 + .18*Math.sin(t*2.1+a.y*.012);
    const c1x = SEL.sx + (axisX-SEL.sx)*0.58, c1y = SEL.sy + Math.sin(a.y*.018+t)*20;
    const c2x = axisX + (a.x-axisX)*0.50, c2y = a.y + Math.cos(a.y*.015+t)*14;
    svg += '<path d="M '+SEL.sx.toFixed(1)+' '+SEL.sy.toFixed(1)+' C '+c1x.toFixed(1)+' '+c1y.toFixed(1)+' '+c2x.toFixed(1)+' '+c2y.toFixed(1)+' '+a.x.toFixed(1)+' '+a.y.toFixed(1)+'" '+
      'stroke="rgba('+r+','+g+','+b+','+op.toFixed(3)+')" stroke-width="1.55" fill="none"/>';
    svg += '<circle cx="'+a.x.toFixed(1)+'" cy="'+a.y.toFixed(1)+'" r="2.6" fill="rgba('+r+','+g+','+b+',.78)"/>';
  }
  links.innerHTML = svg;
}

// ---- interacción ----
let lx=0, ly=0, moved=0;
stage.addEventListener('pointerdown', e=>{ dragging=true; moved=0; lx=e.clientX; ly=e.clientY;
  stage.classList.add('drag'); spin=false; });
window.addEventListener('pointermove', e=>{
  const r = stage.getBoundingClientRect();
  mx = e.clientX - r.left; my = e.clientY - r.top;
  if(!dragging) return;
  const dx=e.clientX-lx, dy=e.clientY-ly; moved += Math.abs(dx)+Math.abs(dy);
  lon0 -= dx/(R*zoom)*0.9; lat0 += dy/(R*zoom)*0.9;
  lat0 = Math.max(-1.45, Math.min(1.45, lat0));
  lx=e.clientX; ly=e.clientY;
});
window.addEventListener('pointerup', e=>{
  stage.classList.remove('drag');
  if (dragging && moved < 6) pick(e.clientX, e.clientY);
  dragging=false;
});
stage.addEventListener('wheel', e=>{ e.preventDefault();
  zoom = Math.max(.7, Math.min(6, zoom * (e.deltaY<0 ? 1.12 : 0.89))); }, {passive:false});

function pick(cx, cy){
  const r = stage.getBoundingClientRect();
  const x = cx - r.left, y = cy - r.top;
  let best=null, bd=196;
  for (const q of PTS){
    if (q.sx===undefined) continue;
    const d=(q.sx-x)*(q.sx-x)+(q.sy-y)*(q.sy-y);
    if (d<bd){ bd=d; best=q; }
  }
  if (best){ SEL=best; showDetail(best); }
}

function fmtDays(d){
  if (d==null) return '—';
  if (d<45) return Math.round(d)+' días';
  if (d<720) return (d/30.4).toFixed(1)+' meses';
  return (d/365.25).toFixed(1)+' años';
}
// ---- inspector de universos: valores vivos + particiones completas por variable -------------------
const V2_IN = (UNIVERSES.global_v2_mamdani && UNIVERSES.global_v2_mamdani.input_universes) || {};
const V3_IN = (UNIVERSES.global_v3_ia2_ia3 && UNIVERSES.global_v3_ia2_ia3.input_universes) || {};
const V2_OUT = (UNIVERSES.global_v2_mamdani && UNIVERSES.global_v2_mamdani.output_universes) || {};
const V3_OUT = (UNIVERSES.global_v3_ia2_ia3 && UNIVERSES.global_v3_ia2_ia3.output_universes) || {};
const AUDIT_INPUTS = [
  {k:'a_seis', label:'sismicidad', src:'global v2 · Mamdani'},
  {k:'a_swarm', label:'enjambre', src:'global v2 · Mamdani'},
  {k:'a_fore', label:'foreshock', src:'global v2 · Mamdani'},
  {k:'A_state', label:'actividad persistente', src:'global v2 · Mamdani'},
  {k:'C_state', label:'carga/recurrencia', src:'global v2 · Mamdani'},
  {k:'mag_scale', label:'escala tectónica', src:'global v2 · Mamdani'},
  {k:'a_tect', label:'slip GEM', src:'global v2 · Mamdani'},
  {k:'ia1', label:'núcleo estadístico IA1', src:'global v3 · IA1→IA2→IA3'}
];
function fmtNum(v, n=3){
  if(v===undefined || v===null || Number.isNaN(Number(v))) return '—';
  const x = Number(v);
  if(Math.abs(x)>=1000) return x.toExponential(2);
  return x.toFixed(n).replace(/\.?0+$/,'');
}
function universeFor(k){
  if(k==='ia1' && V3_IN.ia1) return V3_IN.ia1;
  return V2_IN[k] || {domain:[-1,1], sets:SETS5};
}
function clamp01(x){ return Number.isFinite(x) ? Math.max(0, Math.min(1, x)) : 0; }
function posInDomain(v, domain){
  const lo=Number(domain[0]), hi=Number(domain[1]);
  return clamp01((Number(v)-lo)/(hi-lo || 1))*100;
}
function termList(sets){ return Object.keys(sets || {}); }
function setParams(p){ return '['+p.map(x=>fmtNum(x,3)).join(', ')+']'; }
function universeRow(q, item){
  const k=item.k, v=q.in[k], u=universeFor(k), sets=u.sets || SETS5;
  const dt = domTerm(Number(v), sets), pos = posInDomain(v, u.domain).toFixed(2);
  const terms = termList(sets);
  const many = terms.length > 6 ? ' many' : '';
  const ticks = [25,50,75].map(x=>'<i style="left:'+x+'%"></i>').join('');
  const setHtml = terms.map(t=>{
    const mu = membJS(Number(v), sets[t]);
    return '<div class="u-set '+(t===dt[0]?'hotset':'')+'"><strong>'+t.replace('_',' ')+'</strong>'+
      setParams(sets[t])+'<br>μ='+mu.toFixed(2)+'</div>';
  }).join('');
  return '<div class="audit-row" data-var="'+k+'">'+
    '<div class="r0"><div><code>'+k+'</code><div class="src">'+item.src+'</div></div>'+
    '<div><strong>'+item.label+'</strong></div>'+
    '<div class="dom">U=['+fmtNum(u.domain[0],3)+', '+fmtNum(u.domain[1],3)+']</div></div>'+
    '<div class="u-track">'+ticks+'<b style="left:'+pos+'%"></b></div>'+
    '<div class="r0"><div class="src">valor vivo</div><div class="val">'+fmtNum(v,4)+
    '</div><div class="dom">'+dt[0].replace('_',' ')+' · μ='+dt[1].toFixed(2)+'</div></div>'+
    '<div class="u-sets'+many+'">'+setHtml+'</div></div>';
}
function outputUniverseRow(name, value, domain, note){
  const pos = posInDomain(value, domain).toFixed(2);
  return '<div class="audit-row">'+
    '<div class="r0"><div><code>'+name+'</code><div class="src">salida</div></div>'+
    '<div class="val">'+fmtNum(value,4)+'</div><div class="dom">U=['+fmtNum(domain[0],3)+', '+fmtNum(domain[1],3)+']</div></div>'+
    '<div class="u-track">'+[25,50,75].map(x=>'<i style="left:'+x+'%"></i>').join('')+
    '<b style="left:'+pos+'%"></b></div><div class="note">'+note+'</div></div>';
}
function auditRows(q){
  return AUDIT_INPUTS.map(x=>universeRow(q,x)).join('');
}
function outputRows(q){
  const occU = (V2_OUT.ocurrencia && V2_OUT.ocurrencia.domain) || [-1,1];
  const magU = (V2_OUT.magnitud && V2_OUT.magnitud.domain) || [3,8.5];
  return outputUniverseRow('p30_calibrada', q.p, [0,1], 'probabilidad calibrada final usada en el globo')+
    outputUniverseRow('mom_mamdani', q.mom, occU, 'salida Mamdani bipolar antes de calibración')+
    outputUniverseRow('p3_ia2_ia3', q.p3, [0,1], 'señal fuzzy experimental en sombra')+
    outputUniverseRow('mag_climatológica', q.mag, magU, 'valor típico histórico; no es una predicción de magnitud');
}
function rawRows(q){
  return '<div class="raw-grid">'+
    '<div><span>omori</span><b>'+fmtNum(q.in.omori,3)+'</b></div>'+
    '<div><span>rate_bg/año</span><b>'+fmtNum(q.in.rate_bg,3)+'</b></div>'+
    '<div><span>n30</span><b>'+fmtNum(q.in.n30,0)+'</b></div>'+
    '<div><span>n365</span><b>'+fmtNum(q.n365,0)+'</b></div>'+
    '<div><span>t_last_d</span><b>'+(q.tlast>=99999?'—':fmtNum(q.tlast,0))+'</b></div>'+
    '<div><span>alerta</span><b>'+(q.alert?'sí':'no')+'</b></div>'+
    '</div>';
}
function showDetail(q){
  const el = document.getElementById('detail');
  const cc = 'rgb('+q.col[0]+','+q.col[1]+','+q.col[2]+')';
  const ccs = 'rgba('+q.col[0]+','+q.col[1]+','+q.col[2]+',.5)';
  const pl = q.place || nearestPlace(q.la+0.5, q.lo+0.5);
  el.innerHTML =
    '<div class="axis"><div class="k">Ubicación</div>'+
    '<div class="geo-title">'+pl.city+', '+pl.country+'</div>'+
    '<div class="geo-meta">'+pl.region+' · ciudad de referencia a '+pl.distance_km+' km'+
    (pl.remote?' · celda oceánica/remota':'')+'</div>'+
    '<div class="big" style="font-size:16px;margin-top:10px">'+(q.la+0.5)+'°, '+(q.lo+0.5)+'°</div>'+
    '<div class="note">celda 1°×1° (~111 km) · '+q.n365+' sismos M≥4.5 en su vecindad el último año'+
    ' · último hace '+(q.tlast>=99999?'—':fmtDays(q.tlast))+'</div></div>'+
    (q.alert?'<div class="axis" style="border-color:rgba(255,60,60,.5);box-shadow:0 0 24px rgba(255,45,85,.12)">'+
      '<div class="k" style="color:#ff6b6b">Señal fuzzy experimental</div>'+
      '<div class="note">Indicador retrospectivo correlacionado y mantenido en sombra; no es una alerta ni una probabilidad operativa. Valor: '+
      (q.p3*100).toFixed(0)+'%</div></div>':'')+
    '<div class="audit-block"><h3>Entradas fuzzy · valor vivo, universo y particiones</h3>'+
    auditRows(q)+'</div>'+
    '<div class="audit-block"><h3>Salidas calculadas · universo de discurso</h3>'+
    outputRows(q)+'</div>'+
    '<div class="audit-block"><h3>Variables crudas de soporte</h3>'+rawRows(q)+'</div>'+
    '<div class="axis"><div class="k">Ocurrencia · P(M≥5 en 30 días)</div>'+
    '<div class="big" style="color:'+cc+';text-shadow:0 0 20px '+ccs+'">'+(q.p*100).toFixed(1)+'%</div>'+
    '<div class="note">modelo estadístico Omori+tasa; la capa fuzzy se publica solo como diagnóstico en sombra. MOM='+(q.mom!==undefined?q.mom.toFixed(2):'—')+'</div></div>'+
    '<div class="axis"><div class="k">Magnitud histórica de referencia</div>'+
    '<div class="big">M típica '+q.mag.toFixed(1)+'</div>'+
    '<div class="note">climatología de los M≥5 de esta zona, no magnitud prevista · máximo histórico '+
    (q.magmax&&q.magmax!=='—'?'M'+q.magmax:'—')+'</div></div>'+
    '<div class="axis"><div class="k">Espera media Poisson derivada</div>'+
    '<div class="big">'+fmtDays(q.mdays)+'</div>'+
    '<div class="note">1/λ calculado desde P30. No es intervalo previsto, fecha objetivo ni cuenta atrás.</div></div>'+
    '<div class="note">Vista de investigación. El registro prospectivo comenzó ahora; las pruebas históricas previas son exploratorias.</div>';
}

function kmBetween(a,b){
  const r=Math.PI/180, p1=(a.la+.5)*r, p2=(b.la+.5)*r;
  const dp=p2-p1, dl=((((b.lo-a.lo)+540)%360)-180)*r;
  const h=Math.sin(dp/2)**2+Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)**2;
  return 12742*Math.asin(Math.sqrt(Math.min(1,h)));
}
function diversifiedHot(points,n=8,minKm=300){
  const chosen=[];
  for(const q of points.slice().sort((a,b)=>b.p-a.p)){
    if(chosen.every(x=>kmBetween(q,x)>=minKm)) chosen.push(q);
    if(chosen.length===n) break;
  }
  return chosen;
}

// ---- datos vivos ----
async function loadData(){
  try{
    const r = await fetch('globe_data.json?t='+Date.now());
    DATA = await r.json();
    const F = {}; DATA.fields.forEach((n,i)=>F[n]=i);
    PTS = DATA.cells.map(c=>{
      const la=c[F.la], lo=c[F.lo];
      const f=(la+0.5)*Math.PI/180, l=(lo+0.5)*Math.PI/180;
      return {la, lo, p:c[F.p30], mag:c[F.mag_typical]||c[F.mag_exp]||5.2, magmax:c[F.mag_max]||'—',
              n365:c[F.n365], tlast:c[F.t_last_d], mdays:c[F.poisson_mean_wait_days]||c[F.mean_days], mom:c[F.mom],
              alert:c[F.alert]||0, p3:c[F.p3]||0,
              place:nearestPlace(la+0.5, lo+0.5),
              in:{a_seis:c[F.a_seis], a_swarm:c[F.a_swarm], a_fore:c[F.a_fore],
                  A_state:c[F.A_state], C_state:c[F.C_state], mag_scale:c[F.mag_scale],
                  a_tect:c[F.a_tect], ia1:c[F.ia1], omori:c[F.omori],
                  rate_bg:c[F.rate_bg], n30:c[F.n30]},
              sinf:Math.sin(f), cosf:Math.cos(f), lam:l, col:pColor(c[F.p30])};
    });
    RECENT = (DATA.recent||[]).map((e,i)=>{
      const f=e[0]*Math.PI/180, l=e[1]*Math.PI/180;
      return {sinf:Math.sin(f), cosf:Math.cos(f), lam:l, mag:e[2], hours:e[3], seed:i*0.37%1};
    });
    document.getElementById('subtitle').textContent =
      DATA.n_cells+' micro-puntos · '+DATA.target+' · '+(DATA.engine||'')+
      ' · pulsos blancos = sismos reales M≥4.5 (radar, distinto del objetivo M≥5)'+
      (DATA.n_alerts?' · '+DATA.n_alerts+' señales fuzzy experimentales':'');
    updLive(new Date(DATA.generated));
    const hot = diversifiedHot(PTS,8,300);
    document.getElementById('hotlist').innerHTML = hot.map((q,i)=>
      '<div class="hot" onclick="goTo('+i+')"><span>'+q.place.city+', '+q.place.country+'</span>'+
      '<b>'+(q.p*100).toFixed(1)+'%</b></div>').join('');
    window._hot = hot;
    if (!SEL && hot[0]) {
      lon0 = hot[0].lam; lat0 = Math.asin(hot[0].sinf);
      SEL = hot[0]; showDetail(SEL); spin=false;
    }
  }catch(e){ document.getElementById('livetxt').textContent = 'error de datos'; }
}
function goTo(i){
  const q = window._hot[i];
  lon0 = q.lam; lat0 = Math.asin(q.sinf);
  SEL = q; showDetail(q); spin=false;
}
function updLive(gen){
  const mins = Math.round((Date.now()-gen.getTime())/60000);
  document.getElementById('livetxt').textContent = mins<=20
    ? 'actualizado · datos de hace '+(mins<1?'<1':mins)+' min'
    : 'DESACTUALIZADO · última salida hace '+mins+' min';
}
setInterval(()=>{ if(DATA) updLive(new Date(DATA.generated)); }, 30000);
setInterval(loadData, 5*60*1000);

function tick(){
  if (spin) lon0 += 0.0012;
  draw();                                   // animación continua (ondas + pulsos + lente)
  requestAnimationFrame(tick);
}
resize(); loadData(); tick();
</script></body></html>
"""


def main():
    coast = load_world()
    universes = load_universes()
    html = (TEMPLATE
            .replace("__COAST__", json.dumps(coast))
            .replace("__UNIVERSES__", json.dumps(universes)))
    out = os.path.join(GLOBE, "index.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"-> {out} ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
