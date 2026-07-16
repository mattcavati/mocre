"""MOCRE-1 world forecast map generator. Reads out/forecast.json + map/world_land.json and emits
a SINGLE self-contained map/index.html (world geometry + forecast embedded inline, no external
requests) -- servible por HTTP o abrible directo. Regenerar tras cada corrida del modelo:
    python3 src/forecast.py && python3 src/build_map.py

The 3 output axes the spec defines are the map's organizing principle, explicit per segment:
  UBICACIÓN  -> geographic coordinates (marker position on the world map + trace + lat/lon label)
  VALOR      -> Richter magnitude (marker size + M>= label)
  DELTA TIEMPO -> when (color + P(1yr) + mean-years-to-next + honest temporal bucket)
Honesty carried through: each marker shows corrected per-event CSEP and N-test diagnostics;
none is presented as confirmatory before the prospective ledger accumulates outcomes.
"""
import json
import os

from common import ROOT

OUT = os.path.join(ROOT, "out")
MAP_DIR = os.path.join(ROOT, "map")


def risk_color(p_1yr):
    """Green (low near-term chance) -> amber -> red (high). On P(1yr) of an M>=target event."""
    stops = [(0.0, (46, 160, 120)), (0.15, (120, 170, 70)), (0.35, (210, 170, 60),),
             (0.6, (220, 120, 50)), (1.0, (210, 60, 60))]
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]
        x1, c1 = stops[i + 1]
        if p_1yr <= x1:
            t = 0 if x1 == x0 else (p_1yr - x0) / (x1 - x0)
            return tuple(round(c0[k] + t * (c1[k] - c0[k])) for k in range(3))
    return stops[-1][1]


def main():
    forecast = json.load(open(os.path.join(OUT, "forecast.json")))
    world = json.load(open(os.path.join(MAP_DIR, "world_land.json")))

    # enrich each segment with a precomputed color (kept out of the JS to keep it simple)
    for s in forecast["segments"]:
        c = risk_color(s["p_1yr"])
        s["color"] = f"rgb({c[0]},{c[1]},{c[2]})"

    html = TEMPLATE.replace("__WORLD__", json.dumps(world)) \
                   .replace("__FORECAST__", json.dumps(forecast))
    out_path = os.path.join(MAP_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({os.path.getsize(out_path)//1024} KB, self-contained)")
    print("servir:  cd map && python3 -m http.server 7896")


TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MOCRE-1 · Mapa de investigación</title>
<style>
  :root{
    --bg:#0a0d12; --panel:rgba(20,26,34,.82); --line:#1e2831; --sea:#0c1119;
    --land:#161d26; --land-stroke:#232e3a; --txt:#dfe7ee; --dim:#8797a6; --accent:#4ea3c9;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  #app{display:flex;height:100vh;overflow:hidden}
  #mapwrap{flex:1;position:relative;overflow:hidden;background:
    radial-gradient(1200px 700px at 60% 30%,#0e141c,#080b10)}
  svg{width:100%;height:100%;display:block;cursor:grab}
  svg.drag{cursor:grabbing}
  .land{fill:var(--land);stroke:var(--land-stroke);stroke-width:.3;vector-effect:non-scaling-stroke;
    pointer-events:none}
  .graticule{stroke:#141b23;stroke-width:.4;fill:none;vector-effect:non-scaling-stroke;
    pointer-events:none}
  .trace{stroke:#e8eef4;stroke-width:1.4;fill:none;opacity:.55;vector-effect:non-scaling-stroke;
    pointer-events:none}
  .hit{fill:transparent}
  .seg{cursor:pointer}
  .seg .halo{opacity:.22}
  .seg .ring{fill:none;stroke:#fff;stroke-width:1;opacity:.35;vector-effect:non-scaling-stroke}
  .seg.active .ring{opacity:.95;stroke-width:1.6}
  .seg .lbl{fill:var(--txt);font-size:11px;font-weight:600;paint-order:stroke;
    stroke:#0a0d12;stroke-width:3px;pointer-events:none}
  header{position:absolute;top:0;left:0;right:0;padding:14px 18px;pointer-events:none;
    background:linear-gradient(#0a0d12cc,#0a0d1200)}
  header h1{margin:0;font-size:15px;letter-spacing:.08em;font-weight:700}
  header .sub{font-size:11px;color:var(--dim);margin-top:3px;max-width:70%}
  #panel{width:340px;flex:none;background:var(--panel);backdrop-filter:blur(14px);
    border-left:1px solid var(--line);padding:18px;overflow-y:auto}
  #panel h2{margin:0 0 2px;font-size:14px;letter-spacing:.03em}
  #panel .rg{font-size:11px;color:var(--dim);margin-bottom:16px}
  .place{border:1px solid rgba(78,163,201,.26);border-radius:12px;padding:12px 13px;margin-bottom:12px;
    background:linear-gradient(135deg,rgba(78,163,201,.08),rgba(255,255,255,.012))}
  .place .city{font-size:20px;font-weight:740;letter-spacing:.01em}
  .place .meta{font-size:12px;color:var(--dim);margin-top:3px;line-height:1.45}
  .place .zone{font-size:10px;color:#9bd6ee;letter-spacing:.12em;text-transform:uppercase;margin-top:10px}
  .axis{border:1px solid var(--line);border-radius:10px;padding:12px 13px;margin-bottom:11px;
    background:rgba(255,255,255,.015)}
  .axis .k{font-size:10px;letter-spacing:.14em;color:var(--accent);text-transform:uppercase;font-weight:700}
  .axis .big{font-size:22px;font-weight:700;margin:5px 0 2px;font-variant-numeric:tabular-nums}
  .axis .note{font-size:11px;color:var(--dim);line-height:1.45}
  .bar{height:6px;border-radius:3px;background:#1a232c;margin:8px 0 4px;overflow:hidden}
  .bar>i{display:block;height:100%;border-radius:3px}
  .rows{font-size:11px;color:var(--dim)}
  .rows div{display:flex;justify-content:space-between;padding:2px 0}
  .rows b{color:var(--txt);font-weight:600;font-variant-numeric:tabular-nums}
  .verdict{font-size:11px;padding:8px 10px;border-radius:8px;line-height:1.4;margin-top:4px}
  .v-better{background:rgba(46,160,120,.14);border:1px solid rgba(46,160,120,.4);color:#a9e6cf}
  .v-same{background:rgba(150,150,150,.08);border:1px solid #2a343e;color:var(--dim)}
  .v-skip{background:rgba(80,110,140,.1);border:1px solid #263748;color:#9fb6ca}
  #legend{position:absolute;bottom:14px;left:18px;font-size:10px;color:var(--dim);
    background:#0a0d12aa;padding:9px 11px;border-radius:8px;border:1px solid var(--line)}
  #legend .grad{width:150px;height:8px;border-radius:4px;margin:5px 0 3px;
    background:linear-gradient(90deg,rgb(46,160,120),rgb(210,170,60),rgb(210,60,60))}
  #legend .lr{display:flex;justify-content:space-between}
  .hint{position:absolute;bottom:14px;right:18px;font-size:10px;color:#4a5764}
  .empty{color:var(--dim);font-size:12px;line-height:1.5;margin-top:40px;text-align:center}
  @media (max-width:720px){
    #app{flex-direction:column}
    #mapwrap{height:58vh;flex:none}
    #panel{width:100%;height:42vh;border-left:0;border-top:1px solid var(--line);padding:14px}
    header{padding:12px 14px}
    header h1{font-size:13px}
    header .sub{display:none}
    #legend{left:12px;bottom:12px;transform:scale(.84);transform-origin:left bottom}
    .hint{display:none}
    .seg .lbl{font-size:10px}
    .axis .big{font-size:20px}
    .empty{margin-top:20px}
  }
</style>
</head>
<body>
<div id="app">
  <div id="mapwrap">
    <svg id="map" viewBox="0 0 1000 500" preserveAspectRatio="xMidYMid meet"></svg>
    <header>
      <h1>MOCRE-1 · MAPA DE PREDICCIÓN</h1>
      <div class="sub" id="hdrsub"></div>
    </header>
    <div id="legend">
      <div>Riesgo — P(evento ≥ M en 1 año)</div>
      <div class="grad"></div>
      <div class="lr"><span>bajo</span><span>alto</span></div>
      <div style="margin-top:6px">Tamaño ∝ magnitud objetivo (Richter)</div>
    </div>
    <div class="hint">arrastra para mover · rueda para zoom · doble-clic centra</div>
  </div>
  <div id="panel">
    <div class="empty" id="empty">Selecciona un segmento en el mapa<br>para ver sus 3 salidas:
      <br><br>ubicación · umbral objetivo · horizonte probabilístico</div>
    <div id="detail" style="display:none"></div>
  </div>
</div>
<script>
const WORLD = __WORLD__;
const FC = __FORECAST__;
const svg = document.getElementById('map');
const NS = "http://www.w3.org/2000/svg";
const W = 1000, H = 500;
// equirectangular projection
function px(lon){ return (lon + 180) / 360 * W; }
function py(lat){ return (90 - lat) / 180 * H; }

function el(tag, attrs){ const e = document.createElementNS(NS, tag);
  for(const k in attrs) e.setAttribute(k, attrs[k]); return e; }

// gworld = geographic layer (graticule + land + fault traces): scales with zoom via a transform.
// gseg   = markers: positioned by the SAME projection+transform but drawn at CONSTANT screen size
//          (circles re-anchored on every zoom/pan; only their position follows the map, not size).
const gworld = el('g', {});
const gseg = el('g', {});

// graticule
for(let lon=-180; lon<=180; lon+=30){ gworld.appendChild(el('line',
  {x1:px(lon),y1:0,x2:px(lon),y2:H,class:'graticule'})); }
for(let lat=-60; lat<=60; lat+=30){ gworld.appendChild(el('line',
  {x1:0,y1:py(lat),x2:W,y2:py(lat),class:'graticule'})); }
// land
for(const poly of WORLD){ for(const ring of poly){
  let d = "M" + ring.map(p => px(p[0]).toFixed(1)+" "+py(p[1]).toFixed(1)).join(" L") + "Z";
  gworld.appendChild(el('path', {d, class:'land'}));
}}
// fault traces (geographic -> in gworld, scale with the map)
FC.segments.forEach(s => {
  if(s.trace && s.trace.length>1){
    const d = "M" + s.trace.map(p => px(p[0]).toFixed(1)+" "+py(p[1]).toFixed(1)).join(" L");
    gworld.appendChild(el('path', {d, class:'trace'}));
  }
});
svg.appendChild(gworld);

// markers -- each is a <g class=seg> centered at (0,0) with fixed-radius shapes; a translate
// positions it, recomputed on zoom/pan so the marker tracks its lon/lat but never changes size.
const bySlug = {};
const LABEL_PLACEMENT = {
  "saf-parkfield": {x:-12, y:-14, anchor:"end"},
  "saf-mojave": {x:-12, y:16, anchor:"end"},
  "sjc-anza": {x:13, y:22, anchor:"start"},
  "csz-central-or": {x:13, y:-10, anchor:"start"},
  "ok-pawnee-prague": {x:13, y:3, anchor:"start"}
};
FC.segments.forEach(s => {
  const r = 4 + (s.target_mag - 3) * 2.4;         // VALOR: size by Richter magnitude (screen px)
  const grp = el('g', {class:'seg'});
  grp.appendChild(el('circle', {cx:0, cy:0, r:Math.max(r*1.7,14), class:'hit'}));  // click target
  grp.appendChild(el('circle', {cx:0, cy:0, r:r*1.7, fill:s.color, class:'halo'}));
  grp.appendChild(el('circle', {cx:0, cy:0, r, fill:s.color}));    // DELTA TIEMPO: color by P(1yr)
  grp.appendChild(el('circle', {cx:0, cy:0, r:r+3, class:'ring'}));
  const lp = LABEL_PLACEMENT[s.slug] || {x:r+5, y:3, anchor:"start"};
  const t = el('text', {x:lp.x, y:lp.y, class:'lbl', "text-anchor":lp.anchor});
  t.textContent = s.short_label || s.segment_id.split('-')[0];
  grp.appendChild(t);
  grp.addEventListener('click', (e)=>{ e.stopPropagation(); select(s.slug); });
  gseg.appendChild(grp);
  bySlug[s.slug] = {group:grp, data:s, mapx:px(s.lon), mapy:py(s.lat)};
});
svg.appendChild(gseg);

document.getElementById('hdrsub').textContent = FC.model_note.split('.')[1] ?
  ("Salidas: ubicación · umbral objetivo · horizontes Poisson. " + FC.model_note) :
  FC.model_note;

function verdictClass(v){ if(!v) return 'v-skip';
  if(v.indexOf('IG/evento=+')>=0 && v.indexOf('N-test=pasa')>=0) return 'v-better';
  if(v.indexOf('IG/evento=')>=0) return 'v-same';
  return 'v-skip'; }
function verdictText(v){
  return v+' · no confirmatorio'; }

let ACTIVE_SLUG=null;
function select(slug){
  ACTIVE_SLUG=slug;
  Object.values(bySlug).forEach(o => o.group.classList.remove('active'));
  const o = bySlug[slug]; o.group.classList.add('active');
  const s = o.data;
  document.getElementById('empty').style.display='none';
  const d = document.getElementById('detail'); d.style.display='block';
  const p1 = (s.p_1yr*100);
  const yrs = s.mean_years_to_next;
  d.innerHTML = `
    <h2>${s.segment_id}</h2>
    <div class="rg">${s.name} · régimen ${s.regime}</div>

    <div class="place">
      <div class="city">${s.place ? s.place.city : s.short_label}</div>
      <div class="meta">${s.place ? `${s.place.region} · ${s.place.country}` : s.name}</div>
      <div class="zone">${s.place ? s.place.zone : 'zona sísmica modelada'}</div>
    </div>

    <div class="axis">
      <div class="k">Ubicación · coordenadas geográficas</div>
      <div class="big">${s.lat.toFixed(2)}°, ${s.lon.toFixed(2)}°</div>
      <div class="note">Centroide del segmento sobre el universo de discurso geográfico. La traza de falla se dibuja sobre el mapa; el nombre regional evita que la coordenada quede muda.</div>
    </div>

    <div class="axis">
      <div class="k">Umbral objetivo del modelo</div>
      <div class="big">M ≥ ${s.target_mag}</div>
      <div class="note">Umbral de magnitud objetivo del segmento.</div>
    </div>

    <div class="axis">
      <div class="k">Horizonte probabilístico</div>
      <div class="big">${p1.toFixed(1)}% <span style="font-size:12px;color:var(--dim)">en 1 año</span></div>
      <div class="bar"><i style="width:${Math.min(100,p1).toFixed(1)}%;background:${s.color}"></i></div>
      <div class="rows">
        <div><span>P(30 días)</span><b>${(s.p_30d*100).toFixed(2)}%</b></div>
        <div><span>P(1 año)</span><b>${p1.toFixed(1)}%</b></div>
        <div><span>P(10 años)</span><b>${(s.p_10yr*100).toFixed(1)}%</b></div>
        <div><span>Espera media Poisson 1/λ</span><b>${yrs ? yrs.toFixed(1)+' años' : '—'}</b></div>
      </div>
      <div class="note" style="margin-top:6px">Climatología Poisson, no una fecha, ventana ni cuenta atrás. Puede permanecer estable si la tasa cambia poco.</div>
    </div>

    <div class="axis">
      <div class="k">Ganancia vs ETAS (hoy)</div>
      <div class="rows">
        <div><span>λ MOCRE-1 / día</span><b>${s.lam_mocre_daily.toExponential(2)}</b></div>
        <div><span>λ ETAS-solo / día</span><b>${s.lam_etas_daily.toExponential(2)}</b></div>
        <div><span>G (refinador)</span><b>${s.gain_now}×</b></div>
      </div>
      <div class="verdict ${verdictClass(s.csep_verdict)}">${verdictText(s.csep_verdict)}</div>
    </div>
    ${fuzzyBlock(s.fuzzy)}
    ${criticalBlock(s.critical_shadow)}
    <div class="note" style="color:#556; font-size:10px">datos hasta ${s.last_data} · ${s.n_target_events_hist} eventos M≥${s.target_mag} históricos</div>
  `;
}

function criticalBlock(c){
  if(!c) return '';
  const top=(c.top_rules||[]).slice(0,3).map(r=>`${r.rule}: ${r.weighted.toFixed(3)}`).join(' · ');
  return `<div class="axis" style="border-left:2px solid #9d72d8;padding-left:8px;margin-top:14px">
    <div class="k" style="color:#b494e6">MOCRE-2 · estado crítico inferido (sombra)</div>
    <div class="big">${(c.critical_state*100).toFixed(1)}%</div>
    <div class="note">Índice fuzzy recurrente, no probabilidad de terremoto. Umbral experimental ${(c.threshold*100).toFixed(1)}%; modelo no promocionado.</div>
    <div class="note">Reglas dominantes: ${top||'sin evidencia suficiente'}.</div>
  </div>`;
}

async function refreshCriticalShadow(){
  try{
    const r=await fetch('../critical-shadow.json?t='+Date.now());
    if(!r.ok) return;
    const live=await r.json();
    const by=new Map((live.segments||[]).map(x=>[x.segment,x]));
    FC.segments.forEach(s=>{ if(by.has(s.slug)) s.critical_shadow=by.get(s.slug); });
    if(ACTIVE_SLUG) select(ACTIVE_SLUG);
  }catch(_e){}
}
refreshCriticalShadow();
setInterval(refreshCriticalShadow,10*60*1000);

// Salidas fuzzy experimentales: solo se muestran valores que superan un baseline
// condicionado en el holdout; las demás quedan explícitamente no publicadas.
function fuzzyBlock(f){
  if(!f) return '';
  const oc = f.ocurrencia, mg = f.magnitud, dn = f.donde;
  const mgTxt = (mg.value==null) ? 'no publicada' : `M IA3 ${mg.value.toFixed(1)}`;
  const ocTxt = (oc.prob==null) ? 'no publicada' : `${(oc.prob*100).toFixed(0)}%`;
  return `
    <div class="axis" style="border-left:2px solid #4a6fa5;padding-left:8px;margin-top:14px">
      <div class="k" style="color:#7fa8d8">Motor fuzzy experimental · IA2+IA3</div>
      <div class="rows" style="margin-top:6px">
        <div><span>Magnitud</span><b>${mgTxt}</b></div>
        <div><span>Ocurrencia 30d</span><b>${ocTxt}</b></div>
        <div><span>Cuándo</span><b style="color:var(--dim)">sin señal</b></div>
        <div><span>Dónde</span><b style="color:var(--dim)">sin señal</b></div>
      </div>
      <div class="note" style="margin-top:6px;font-size:10px;line-height:1.5">
        <b>magnitud</b>: ${mg.nota}.
        <b>ocurrencia</b>: ${oc.nota}.
        <b>cuándo/dónde</b>: sin señal medida (IA3 converge a la media, no la inventa).
      </div>
    </div>`;
}

// pan + zoom via a transform on gworld (map scales); markers re-anchored at fixed screen size.
// state maps map-coords (0..W,0..H) to screen: screenX = mapX*s + tx.
let s = 1, tx = 0, ty = 0;
function render(){
  gworld.setAttribute('transform', `translate(${tx},${ty}) scale(${s})`);
  for(const o of Object.values(bySlug)){
    o.group.setAttribute('transform', `translate(${o.mapx*s+tx},${o.mapy*s+ty})`);
  }
}
// fit the map so a given map-bbox fills the svg viewport
function fitTo(x0,x1,y0,y1){
  const rect = svg.getBoundingClientRect();
  const vbW = W, vbH = W*(rect.height/rect.width);   // viewBox is 0 0 W (W*aspect)
  s = Math.min(vbW/(x1-x0), vbH/(y1-y0));
  tx = vbW/2 - (x0+x1)/2*s; ty = vbH/2 - (y0+y1)/2*s;
  render();
}
function activeBBox(pad){
  const xs = FC.segments.map(o=>px(o.lon)), ys = FC.segments.map(o=>py(o.lat));
  return [Math.min(...xs)-pad, Math.max(...xs)+pad, Math.min(...ys)-pad, Math.max(...ys)+pad];
}
// keep the svg viewBox height matched to the element aspect so scale math stays square
function syncViewBox(){ const rect = svg.getBoundingClientRect();
  svg.setAttribute('viewBox', `0 0 ${W} ${W*(rect.height/rect.width)}`); }
syncViewBox();
window.addEventListener('resize', ()=>{ syncViewBox(); render(); });

let drag=null;
svg.addEventListener('mousedown', e=>{ drag={x:e.clientX,y:e.clientY,tx,ty};
  svg.classList.add('drag'); });
window.addEventListener('mouseup', ()=>{ drag=null; svg.classList.remove('drag'); });
window.addEventListener('mousemove', e=>{ if(!drag) return;
  const rect = svg.getBoundingClientRect();
  const k = W / rect.width;                 // px(screen) -> viewBox units
  tx = drag.tx + (e.clientX-drag.x)*k; ty = drag.ty + (e.clientY-drag.y)*k; render(); });
svg.addEventListener('wheel', e=>{ e.preventDefault();
  const rect = svg.getBoundingClientRect();
  const k = W / rect.width;
  const mvx = (e.clientX-rect.left)*k, mvy = (e.clientY-rect.top)*k; // cursor in viewBox units
  const f = e.deltaY>0 ? 0.88 : 1.14;
  const ns = Math.max(0.6, Math.min(60, s*f));
  // keep the point under the cursor fixed: mv = mapX*s+t  ->  solve t for new s
  tx = mvx - (mvx - tx)*(ns/s); ty = mvy - (mvy - ty)*(ns/s); s = ns; render();
}, {passive:false});
svg.addEventListener('dblclick', ()=>{ const b=activeBBox(60); fitTo(b[0],b[1],b[2],b[3]); });
svg.addEventListener('click', ()=>{ Object.values(bySlug).forEach(o=>o.group.classList.remove('active'));
  document.getElementById('detail').style.display='none';
  document.getElementById('empty').style.display='block'; });

// initial view: fit all active regions with enough label breathing room.
{ const b=activeBBox(145); fitTo(b[0],b[1],b[2],b[3]); }
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
