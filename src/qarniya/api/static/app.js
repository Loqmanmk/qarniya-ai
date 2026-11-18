// Qarniya AI — interface du médecin.
// Sécurité : tout texte venant du serveur est inséré avec textContent (jamais innerHTML).
"use strict";

const MAPS = [
  { key: "axial", label: "Courbure axiale", tokens: ["axial"] },
  { key: "anterior", label: "Élévation antérieure", tokens: ["anterior", "anterieure", "antérieure"] },
  { key: "posterior", label: "Élévation postérieure", tokens: ["posterior", "posterieure", "postérieure"] },
  { key: "pachymetry", label: "Pachymétrie", tokens: ["pachy"] },
];
const LEVELS = {
  faible: { css: "low", icon: "✓", text: "Risque faible" },
  a_surveiller: { css: "monitor", icon: "◐", text: "À surveiller" },
  eleve: { css: "high", icon: "▲", text: "Risque élevé" },
};
const files = { exam: {}, before: {}, after: {} };

// ---------- petits outils DOM ----------
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}
const pct = (p) => `${Math.round(p * 100)} %`;

function guessMap(fileName) {
  const name = fileName.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
  return MAPS.find((m) => m.tokens.some((t) => name.includes(t.normalize("NFD").replace(/[̀-ͯ]/g, ""))));
}

// ---------- zones de dépôt ----------
function setSlot(group, mapKey, file) {
  files[group][mapKey] = file;
  const slot = document.querySelector(`.dropzone[data-group="${group}"] .slot[data-map="${mapKey}"]`);
  slot.classList.add("filled");
  slot.querySelector("img")?.remove();
  slot.querySelector(".name")?.remove();
  const img = el("img", { alt: "" });
  img.src = URL.createObjectURL(file);
  slot.append(img, el("span", { class: "name", text: file.name }));
}

function buildDropzone(zone) {
  const group = zone.dataset.group;
  for (const map of MAPS) {
    const input = el("input", { type: "file", accept: "image/png,image/jpeg", "aria-label": map.label });
    input.addEventListener("change", () => input.files[0] && setSlot(group, map.key, input.files[0]));
    const slot = el("label", { class: "slot", "data-map": map.key },
      el("span", { class: "slot-label", text: map.label }),
      el("span", { text: "cliquer ou déposer" }), input);
    slot.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length === 1) {
        e.preventDefault(); e.stopPropagation();
        zone.classList.remove("drag");
        setSlot(group, map.key, e.dataTransfer.files[0]);
      }
    });
    zone.append(slot);
  }
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    for (const file of e.dataTransfer.files) {
      const map = guessMap(file.name);
      if (map) setSlot(group, map.key, file);
    }
  });
}

// ---------- état du service ----------
async function loadHealth() {
  const box = document.getElementById("health");
  try {
    const h = await (await fetch("/api/health")).json();
    box.append(
      el("span", { class: "chip on", text: `Modèle ${h.model}` }),
      el("span", { class: `chip ${h.novelty_detector ? "on" : "off"}`, text: "Détecteur d'images inhabituelles" }),
      el("span", { class: `chip ${h.numbers_opinion ? "on" : "off"}`, text: "Avis des chiffres" }),
    );
  } catch {
    box.append(el("span", { class: "chip off", text: "Service indisponible" }));
  }
}

// ---------- envoi ----------
function errorText(payload) {
  if (!payload || !payload.detail) return "Erreur inattendue.";
  if (typeof payload.detail === "string") return payload.detail;
  return payload.detail.map((d) => `${(d.loc || []).slice(-1)[0] || ""} : ${d.msg}`).join(" ; ");
}

async function submit(form, url, buildData, render) {
  const error = form.querySelector(".error");
  const busy = form.querySelector(".busy");
  const button = form.querySelector("button[type=submit]");
  error.hidden = true;
  let data;
  try { data = buildData(); } catch (e) { error.textContent = e.message; error.hidden = false; return; }
  button.disabled = true; busy.hidden = false;
  try {
    const response = await fetch(url, { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(errorText(payload));
    render(payload);
  } catch (e) {
    error.textContent = e.message; error.hidden = false;
  } finally {
    button.disabled = false; busy.hidden = true;
  }
}

function addMaps(data, group, prefix) {
  for (const map of MAPS) {
    const file = files[group][map.key];
    if (!file) throw new Error(`Carte manquante : ${map.label}.`);
    data.append(prefix + map.key, file);
  }
}

// ---------- affichage d'une analyse ----------
function levelChip(code) {
  const level = LEVELS[code];
  return el("span", { class: `level ${level.css}` }, el("span", { "aria-hidden": "true", text: level.icon }),
    el("span", { text: level.text }));
}

function gauge(probability) {
  const g = el("div", { class: "gauge", role: "img", "aria-label": `Score ${pct(probability)}` });
  for (const [from, to, color] of [[0, 30, "var(--good)"], [30, 70, "var(--warning)"], [70, 100, "var(--critical)"]]) {
    const z = el("div", { class: "zone" });
    z.style.left = `${from}%`; z.style.width = `${to - from}%`; z.style.background = color; z.style.opacity = "0.25";
    g.append(z);
  }
  const m = el("div", { class: "marker" });
  m.style.left = `${probability * 100}%`;
  g.append(m);
  const scale = el("div", { class: "gauge-scale" },
    el("span", { text: "0 %" }), el("span", { text: "30 %" }), el("span", { text: "70 %" }), el("span", { text: "100 %" }));
  return el("div", {}, g, scale);
}

function banner(kind, title, lines) {
  const b = el("div", { class: `banner ${kind}`, role: "status" }, el("strong", { text: title }));
  for (const line of lines) b.append(el("div", { text: line }));
  return b;
}

function mapsGrid(maps) {
  const grid = el("div", { class: "maps" });
  for (const m of maps) {
    const img = el("img", { alt: `${m.label} avec zones qui ont fait monter le score` });
    img.src = `data:image/png;base64,${m.overlay_png}`;
    const bar = el("div", { class: "bar" }, el("span"));
    bar.firstChild.style.width = `${Math.round(m.importance * 100)}%`;
    grid.append(el("div", { class: "map-card" }, el("figure", {}, img,
      el("figcaption", {}, el("div", { text: `${m.label} — ${pct(m.importance)} de la décision` }), bar))));
  }
  return grid;
}

function analysisCard(r, title) {
  const card = el("div", { class: "card" });
  if (title) card.append(el("h2", { class: "card-title", text: title }));
  if (r.status === "rejected") {
    card.append(banner("rejected", "✕ Analyse impossible", r.reasons));
    return card;
  }
  if (r.status === "unusual") {
    card.append(banner("unusual", "⚠ Score non fiable", r.reasons));
  }
  if (r.disagreement) {
    card.append(banner("disagree", "◐ Désaccord entre les images et les chiffres",
      ["Le niveau est relevé à « à surveiller » : examen attentif recommandé."]));
  }
  const unreliable = r.status === "unusual";
  const score = el("div", { class: unreliable ? "score unreliable" : "score" },
    el("div", { class: "value", text: pct(r.probability) }),
    el("div", { class: "caption", text: unreliable ? "score images — non fiable" : "score images" }),
    levelChip(r.final_level));
  card.append(el("div", { class: "result-head" }, score,
    el("div", {}, el("p", { class: "summary", text: r.summary }), gauge(r.probability))));
  if (r.numbers_probability !== null && r.numbers_probability !== undefined) {
    card.append(el("div", { class: "opinions" },
      el("div", { class: "opinion" }, el("div", { class: "k", text: "Avis des images (4 cartes)" }),
        el("div", { class: "v", text: `${pct(r.probability)}` })),
      el("div", { class: "opinion" }, el("div", { class: "k", text: "Avis des chiffres (Kmax, épaisseur)" }),
        el("div", { class: "v", text: `${pct(r.numbers_probability)} — ${r.numbers_level_label}` }))));
  }
  card.append(mapsGrid(r.maps),
    el("p", { class: "legend-heat", text: "Zones qui ont fait monter le score. Aide à la lecture, pas une preuve médicale." }));
  return card;
}

function renderAnalysis(r) {
  const target = document.getElementById("result-analyse");
  target.replaceChildren(analysisCard(r));
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderProgression(p) {
  const target = document.getElementById("result-progression");
  const head = el("div", { class: "card" }, el("p", { class: "progress-verdict", text: p.verdict }));
  if (p.comparable) {
    head.append(el("div", { class: "compare" },
      el("div", {}, el("div", { class: "k", text: "Examen ancien" }), el("div", { class: "score" },
        el("div", { class: "value", text: pct(p.before.probability) }))),
      el("div", { class: "arrow", "aria-hidden": "true", text: "→" }),
      el("div", {}, el("div", { class: "k", text: "Examen récent" }), el("div", { class: "score" },
        el("div", { class: "value", text: pct(p.after.probability) })))));
  }
  target.replaceChildren(head, analysisCard(p.after, "Examen récent — détail"), analysisCard(p.before, "Examen ancien — détail"));
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- démarrage ----------
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".dropzone").forEach(buildDropzone);
  loadHealth();

  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => { t.classList.toggle("active", t === tab); t.setAttribute("aria-selected", t === tab); });
    document.querySelectorAll(".panel").forEach((p) => { p.hidden = p.id !== `tab-${tab.dataset.tab}`; });
  }));

  const formA = document.getElementById("form-analyse");
  formA.addEventListener("submit", (e) => {
    e.preventDefault();
    submit(formA, "/api/analyses", () => {
      const data = new FormData();
      data.append("eye", formA.querySelector("input[name=eye]:checked").value);
      addMaps(data, "exam", "");
      for (const name of ["kmax", "pachymetry_thinnest"]) {
        const value = formA.querySelector(`input[name=${name}]`).value;
        if (value !== "") data.append(name, value);
      }
      return data;
    }, renderAnalysis);
  });

  const formP = document.getElementById("form-progression");
  formP.addEventListener("submit", (e) => {
    e.preventDefault();
    submit(formP, "/api/progressions", () => {
      const data = new FormData();
      data.append("eye", formP.querySelector("input[name=eye]:checked").value);
      addMaps(data, "before", "before_");
      addMaps(data, "after", "after_");
      return data;
    }, renderProgression);
  });
});
