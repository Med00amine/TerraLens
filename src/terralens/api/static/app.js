// TerraLens web interface: a thin client over the REST API.

const $ = (selector, root = document) => root.querySelector(selector);
const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
const fmtArea = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const state = { classes: [], sceneFile: null, sceneUrl: null };

// ---------------------------------------------------------------------------- helpers
let toastTimer;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 5000);
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return res.json();
}

function upload(file) {
  const form = new FormData();
  form.append("file", file);
  return { method: "POST", body: form };
}

const loading = (text) => `<div class="loading"><div><div class="spinner"></div>${escapeHtml(text)}</div></div>`;
const failed = (message) =>
  `<div class="placeholder"><p><strong>Something went wrong.</strong><br>${escapeHtml(message)}</p></div>`;

function prettyName(name) {
  const cls = state.classes.find((c) => c.name === name);
  return cls ? cls.label : name.replace(/[_-]+/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

// ------------------------------------------------------------------------- tabs & input
function showTab(btn) {
  for (const b of document.querySelectorAll('[role="tab"]')) {
    const selected = b === btn;
    b.setAttribute("aria-selected", String(selected));
    document.getElementById(b.getAttribute("aria-controls")).hidden = !selected;
  }
}

function setupTabs() {
  const fromHash = () => showTab(location.hash === "#map" ? $("#tab-btn-map") : $("#tab-btn-classify"));
  $("#tab-btn-classify").addEventListener("click", () => {
    history.replaceState(null, "", location.pathname); // does not fire hashchange
    fromHash();
  });
  $("#tab-btn-map").addEventListener("click", () => (location.hash = "map"));
  window.addEventListener("hashchange", fromHash);
  fromHash();
}

function setupDropzone(zone, onFile) {
  const input = $("input", zone);
  input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
    });
  }
  zone.addEventListener("drop", (e) => e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]));
}

function showPreview(zone, file) {
  const img = $(".preview", zone);
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  const url = URL.createObjectURL(file);
  img.src = url;
  img.dataset.url = url;
  img.hidden = false;
  zone.classList.add("has-image");
  showCompressionTip(zone, file);
  return url;
}

// Heavy JPEG compression measurably lowers accuracy (see README, "Robustness").
function showCompressionTip(zone, file) {
  let tip = zone.nextElementSibling?.classList.contains("tip") ? zone.nextElementSibling : null;
  if (!tip) {
    tip = document.createElement("p");
    tip.className = "tip";
    tip.textContent = "Tip: JPEG compression can blur the details the model relies on. Lossless PNG or TIFF exports give the most reliable results.";
    zone.after(tip);
  }
  tip.hidden = file.type !== "image/jpeg" || Boolean(file.isSample);
}

// ------------------------------------------------------------------------ model & samples
async function loadModel() {
  const badge = $("#model-badge");
  try {
    const info = await api("/api/model");
    state.classes = info.classes;
    const acc = info.metrics?.test_acc;
    $("#model-text").textContent =
      `${info.architecture} · ${info.img_size} px` + (acc ? ` · ${pct(acc)} test accuracy` : "");
    badge.classList.add("ok");
  } catch {
    $("#model-text").textContent = "Model not loaded";
    badge.classList.add("error");
  }
}

async function loadSamples() {
  let samples = [];
  try {
    samples = await api("/api/samples");
  } catch {
    return;
  }
  for (const sample of samples) {
    const btn = document.createElement("button");
    const label = prettyName(sample.name);
    btn.type = "button";
    btn.className = "sample";
    btn.title = sample.kind === "tile" ? `Ground truth: ${label}` : label;
    btn.innerHTML =
      `<img src="${sample.url}" alt="${escapeHtml(label)}" loading="lazy"` +
      `${sample.kind === "tile" ? ' class="pixelated"' : ""}><span>${escapeHtml(label)}</span>`;
    btn.addEventListener("click", async () => {
      for (const b of btn.parentElement.children) b.classList.toggle("active", b === btn);
      const blob = await (await fetch(sample.url)).blob();
      const file = new File([blob], sample.url.split("/").pop(), { type: blob.type });
      file.isSample = true; // EuroSAT originals are JPEGs; no compression tip for them
      if (sample.kind === "tile") classifyTile(file);
      else selectScene(file);
    });
    $(sample.kind === "tile" ? "#tile-samples" : "#scene-samples").append(btn);
  }
}

// -------------------------------------------------------------------------- classify
const barRow = (p) => `
  <div class="bar-row">
    <span class="name"><i style="background:${p.color}"></i>${escapeHtml(p.label)}</span>
    <span class="pct">${pct(p.probability)}</span>
    <div class="bar"><div style="width:0;background:${p.color}" data-width="${(p.probability * 100).toFixed(2)}%"></div></div>
  </div>`;

async function classifyTile(file) {
  const url = showPreview($("#tile-drop"), file);
  const out = $("#tile-result");
  out.innerHTML = loading("Analysing tile…");
  try {
    // alpha=1 returns the pure heatmap; blending happens client-side with the slider.
    const res = await api("/api/explain?alpha=1", upload(file));
    renderTile(out, res, url);
  } catch (err) {
    out.innerHTML = failed(err.message);
    toast(err.message);
  }
}

function renderTile(out, res, url) {
  const top = res.explained;
  out.innerHTML = `
    <div class="verdict">
      <div class="swatch" style="background:${top.color}"></div>
      <div>
        <div class="kicker">Predicted land cover</div>
        <div class="label">${escapeHtml(top.label)}</div>
      </div>
      <div class="confidence"><b>${pct(top.probability)}</b><small>confidence</small></div>
    </div>
    <div class="columns">
      <div>
        <p class="section-title"><span>Class probabilities</span><span>top ${res.predictions.length}</span></p>
        <div class="bars">${res.predictions.map(barRow).join("")}</div>
        <div class="chips">
          <span class="chip">Inference + Grad-CAM <b>${res.elapsed_ms.toFixed(0)} ms</b></span>
          <span class="chip">Tile footprint <b>640 m × 640 m</b></span>
        </div>
      </div>
      <div>
        <p class="section-title"><span>Why? Grad-CAM evidence for “${escapeHtml(top.label)}”</span></p>
        <div class="cam-viewer">
          <img class="pixelated" src="${url}" alt="Input tile">
          <img class="heat" src="${res.overlay}" alt="Grad-CAM heatmap" style="opacity:0.5">
        </div>
        <label class="slider">Tile <input type="range" min="0" max="100" value="50" aria-label="Heatmap opacity"> Heatmap</label>
        <div class="cam-legend"><span>low</span><div class="ramp"></div><span>high influence</span></div>
        <p class="explain-note">Warm areas pushed the model towards this class the most.</p>
      </div>
    </div>`;
  const heat = $(".heat", out);
  $(".slider input", out).addEventListener("input", (e) => (heat.style.opacity = e.target.value / 100));
  requestAnimationFrame(() => {
    for (const bar of out.querySelectorAll(".bar > div")) bar.style.width = bar.dataset.width;
  });
}

// ------------------------------------------------------------------------------- map
function selectScene(file) {
  state.sceneFile = file;
  state.sceneUrl = showPreview($("#scene-drop"), file);
  $("#run-map").disabled = false;
  runMap();
}

async function runMap() {
  if (!state.sceneFile) return;
  const tile = Number($("#gsd").value);
  const stride = $("#smooth").checked ? tile / 2 : tile;
  const out = $("#map-result");
  const button = $("#run-map");
  button.disabled = true;
  out.innerHTML = loading("Classifying every 640 m cell of the scene…");
  try {
    const res = await api(`/api/map?tile_px=${tile}&stride_px=${stride}`, upload(state.sceneFile));
    renderMap(out, res);
  } catch (err) {
    out.innerHTML = failed(err.message);
    toast(err.message);
  } finally {
    button.disabled = false;
  }
}

function renderMap(out, res) {
  const s = res.summary;
  out.innerHTML = `
    <div class="viewer-toolbar">
      <div class="segmented" role="group" aria-label="Layer">
        <button data-view="overlay" class="active">Overlay</button>
        <button data-view="map">Land cover</button>
        <button data-view="confidence">Confidence</button>
        <button data-view="image">Satellite</button>
      </div>
      <label class="slider">Opacity <input type="range" min="0" max="100" value="60" aria-label="Map opacity"></label>
    </div>
    <div class="map-viewer">
      <img src="${state.sceneUrl}" alt="Satellite scene">
      <img class="layer pixelated" src="${res.map}" alt="Land-cover map" style="opacity:0.6">
    </div>
    <div class="chips">
      <span class="chip">Area <b>${fmtArea(s.area_km2)} km²</b></span>
      <span class="chip">Tiles classified <b>${s.tiles.toLocaleString()}</b></span>
      <span class="chip">Mean confidence <b>${pct(s.mean_confidence)}</b></span>
      <span class="chip">Uncertain area <b>${pct(s.uncertain_fraction)}</b></span>
      <span class="chip">Processed in <b>${(res.elapsed_ms / 1000).toFixed(2)} s</b></span>
    </div>
    <div class="stacked">${res.classes
      .map((c) => `<div style="width:${c.fraction * 100}%;background:${c.color}" title="${escapeHtml(c.label)}: ${pct(c.fraction)}"></div>`)
      .join("")}</div>
    <table>
      <thead><tr><th>Land cover</th><th class="num">Share</th><th class="num">Area (km²)</th><th class="num">Confidence</th></tr></thead>
      <tbody>${res.classes
        .map(
          (c) => `<tr>
            <td><span class="name"><i style="background:${c.color}"></i>${escapeHtml(c.label)}</span></td>
            <td class="num">${pct(c.fraction)}</td>
            <td class="num">${fmtArea(c.area_km2)}</td>
            <td class="num">${pct(c.mean_confidence)}</td>
          </tr>`,
        )
        .join("")}</tbody>
    </table>`;

  const layer = $(".layer", out);
  const slider = $(".slider input", out);
  const views = { overlay: [res.map, true], map: [res.map, false], confidence: [res.confidence, false], image: [null, false] };
  const setView = (name) => {
    for (const b of out.querySelectorAll(".segmented button")) b.classList.toggle("active", b.dataset.view === name);
    const [src, blended] = views[name];
    layer.hidden = !src;
    if (src) layer.src = src;
    layer.style.opacity = blended ? slider.value / 100 : 1;
    slider.disabled = !blended;
  };
  for (const b of out.querySelectorAll(".segmented button")) b.addEventListener("click", () => setView(b.dataset.view));
  slider.addEventListener("input", () => (layer.style.opacity = slider.value / 100));
}

// ------------------------------------------------------------------------------ init
setupTabs();
setupDropzone($("#tile-drop"), classifyTile);
setupDropzone($("#scene-drop"), selectScene);
$("#run-map").addEventListener("click", runMap);
document.addEventListener("paste", (e) => {
  const file = [...e.clipboardData.files].find((f) => f.type.startsWith("image/"));
  if (!file) return;
  if ($("#tab-map").hidden) classifyTile(file);
  else selectScene(file);
});
await loadModel();
loadSamples();
