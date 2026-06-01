/* ==================================================================
   Enerji Tüketimi Tahmini - Arayüz Mantığı (script.js)
   Backend (Flask) ile fetch üzerinden konuşur. TR/EN dil desteği.
   ================================================================== */

const API = "";

const NUMERIC_IDS = [
  "Temperature", "Humidity", "SquareFootage",
  "Occupancy", "RenewableEnergy", "Hour",
];

const PRED_MIN = 53;
const PRED_MAX = 99;

const I18N = {
  tr: {
    tag: "OLASILIK DERSİ · FİNAL PROJESİ",
    title1: "Enerji Tüketimi ", title2: "Tahmini",
    subtitle: "Regresyon modeli ile bir binanın saatlik enerji tüketimini (kWh) tahmin edin. Aşağıdaki değerleri ayarlayın veya rastgele bir senaryo üretin.",
    loading: "Model yükleniyor…",
    inputParams: "Girdi Parametreleri",
    randomBtn: "⚡ Rastgele Senaryo",
    temp: "Sıcaklık", humidity: "Nem", area: "Bina Alanı",
    occupancy: "Doluluk", people: "kişi",
    renewable: "Yenilenebilir Enerji", hour: "Saat",
    hvac: "HVAC (Isıtma/Soğutma)", lighting: "Aydınlatma",
    dayofweek: "Haftanın Günü", holiday: "Tatil mi?", month: "Ay",
    predictBtn: "Tahmin Et →",
    resultLabel: "TAHMİNİ TÜKETİM",
    lowHigh: "düşük → yüksek",
    resultHint: 'Parametreleri ayarlayıp "Tahmin Et" butonuna basın.',
    modelCompare: "Model Karşılaştırması",
    modelCompareSub: "Test verisi üzerinde R² skoru (yüksek = daha iyi)",
    dataSource: "Veri Kaynağı: Kaggle — Energy Consumption Prediction Dataset",
    bestModel: "En iyi model: ",
    calculating: "Hesaplanıyor…",
    noBackend: "Backend'e bağlanılamadı.",
    hintLow: "Düşük tüketim — verimli bir senaryo.",
    hintMid: "Orta düzey tüketim — tipik bir durum.",
    hintHigh: "Yüksek tüketim — enerji tasarrufu için fırsat var.",
    whyThisResult: "Tahmin Neden Böyle Çıktı?", // YENİ
  },
  en: {
    tag: "PROBABILITY COURSE · FINAL PROJECT",
    title1: "Energy Consumption ", title2: "Prediction",
    subtitle: "Predict a building's hourly energy consumption (kWh) with a regression model. Adjust the values below or generate a random scenario.",
    loading: "Loading model…",
    inputParams: "Input Parameters",
    randomBtn: "⚡ Random Scenario",
    temp: "Temperature", humidity: "Humidity", area: "Floor Area",
    occupancy: "Occupancy", people: "people",
    renewable: "Renewable Energy", hour: "Hour",
    hvac: "HVAC (Heating/Cooling)", lighting: "Lighting",
    dayofweek: "Day of Week", holiday: "Holiday?", month: "Month",
    predictBtn: "Predict →",
    resultLabel: "PREDICTED CONSUMPTION",
    lowHigh: "low → high",
    resultHint: 'Adjust the parameters and click "Predict".',
    modelCompare: "Model Comparison",
    modelCompareSub: "R² score on the test set (higher = better)",
    dataSource: "Data Source: Kaggle — Energy Consumption Prediction Dataset",
    bestModel: "Best model: ",
    calculating: "Calculating…",
    noBackend: "Could not connect to backend.",
    hintLow: "Low consumption — an efficient scenario.",
    hintMid: "Moderate consumption — a typical case.",
    hintHigh: "High consumption — room for energy savings.",
    whyThisResult: "Why did the prediction turn out this way?", // YENİ
  },
};

const LABELS = {
  tr: {
    days: { Monday: "Pazartesi", Tuesday: "Salı", Wednesday: "Çarşamba",
      Thursday: "Perşembe", Friday: "Cuma", Saturday: "Cumartesi", Sunday: "Pazar" },
    onoff: { On: "Açık", Off: "Kapalı" },
    yesno: { Yes: "Evet", No: "Hayır" },
    months: ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
      "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"],
  },
  en: {
    days: { Monday: "Monday", Tuesday: "Tuesday", Wednesday: "Wednesday",
      Thursday: "Thursday", Friday: "Friday", Saturday: "Saturday", Sunday: "Sunday" },
    onoff: { On: "On", Off: "Off" },
    yesno: { Yes: "Yes", No: "No" },
    months: ["January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"],
  },
};

let lang = "tr";
let META = null;
let lastPrediction = null;

function $(id) { return document.getElementById(id); }
function t(key) { return I18N[lang][key]; }

function bindSlider(id) {
  const el = $(id);
  const out = $(id + "_out");
  const update = function() { out.textContent = el.value; };
  el.addEventListener("input", update);
  update();
}

function fillSelect(id, values, labelMap, keepValue) {
  const sel = $(id);
  const prev = keepValue ? sel.value : null;
  sel.innerHTML = "";
  values.forEach(function(v) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = labelMap ? (labelMap[v] ?? v) : v;
    sel.appendChild(opt);
  });
  if (prev !== null) sel.value = prev;
}

function applyTranslations() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach(function(el) {
    const key = el.getAttribute("data-i18n");
    if (I18N[lang][key] !== undefined) el.textContent = I18N[lang][key];
  });
}

function collectInputs() {
  const payload = {};
  NUMERIC_IDS.forEach(function(id) { payload[id] = parseFloat($(id).value); });
  
  // JSHint hatalarını önlemek için Dot Notation kullanıldı
  payload.Occupancy = parseInt($("Occupancy").value, 10);
  payload.Hour = parseInt($("Hour").value, 10);
  payload.Month = parseInt($("Month").value, 10);
  payload.HVACUsage = $("HVACUsage").value;
  payload.LightingUsage = $("LightingUsage").value;
  payload.DayOfWeek = $("DayOfWeek").value;
  payload.Holiday = $("Holiday").value;
  
  // YENİ: Model Seçimi
  const selModel = $("SelectedModel");
  if (selModel && selModel.value) {
      payload.selected_model = selModel.value;
  }
  return payload;
}

function renderSelects(keepValues) {
  if (!META) return;
  const L = LABELS[lang];
  fillSelect("HVACUsage", META.categorical_options.HVACUsage, L.onoff, keepValues);
  fillSelect("LightingUsage", META.categorical_options.LightingUsage, L.onoff, keepValues);
  fillSelect("DayOfWeek", META.categorical_options.DayOfWeek, L.days, keepValues);
  fillSelect("Holiday", META.categorical_options.Holiday, L.yesno, keepValues);
  const months = Array.from({ length: 12 }, function(_, i) { return i + 1; });
  fillSelect("Month", months,
    Object.fromEntries(months.map(function(m) { return [m, L.months[m - 1]]; })), keepValues);
  if (!keepValues) $("Month").value = 6;
}

function setLang(newLang) {
  lang = newLang;
  $("langTR").classList.toggle("active", lang === "tr");
  $("langEN").classList.toggle("active", lang === "en");
  applyTranslations();
  renderSelects(true);
  if (META) {
    $("modelName").textContent = t("bestModel") + META.best_model;
  }
  if (lastPrediction !== null) showResult(lastPrediction);
}

async function loadMetadata() {
  try {
    const res = await fetch(`${API}/api/metadata`);
    META = await res.json();
    $("modelName").textContent = t("bestModel") + META.best_model;
    
    // YENİ: Modelleri açılır menüye ekle
    const modelOptions = [META.best_model, ...Object.keys(META.metrics)];
    const uniqueModels = [...new Set(modelOptions)]; // Tekrar edenleri siler
    fillSelect("SelectedModel", uniqueModels);
    
    renderSelects(false);
    renderMetrics(META.metrics, META.best_model);
  } catch (e) {
    $("modelName").textContent = t("noBackend");
    console.error(e);
  }
}

function renderMetrics(results, bestName) {
  const container = $("metricsBars");
  container.innerHTML = "";
  const entries = Object.entries(results).sort(function(a, b) { return b[1].R2 - a[1].R2; });
  entries.forEach(function(item) {
    const name = item[0];
    const m = item[1];
    const div = document.createElement("div");
    div.className = "metric-bar" + (name === bestName ? " best" : "");
    div.innerHTML = `
      <div class="row">
        <span class="name">${name}</span>
        <span class="val">R² ${m.R2.toFixed(3)}</span>
      </div>
      <div class="track"><div class="bar"></div></div>`;
    container.appendChild(div);
    requestAnimationFrame(function() {
      div.querySelector(".bar").style.width = `${Math.max(0, m.R2) * 100}%`;
    });
  });
}

async function predict() {
  const payload = collectInputs();
  const btn = $("predictBtn");
  btn.textContent = t("calculating");
  btn.disabled = true;
  try {
    const res = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.success) {
      lastPrediction = data.prediction;
      showResult(data.prediction);
      
      // YENİ: SHAP AÇIKLAMALARI (JSHint hatasız)
      const shapContainer = $("shapContainer");
      const shapList = $("shapList");
      if (shapList) shapList.innerHTML = "";
      
      if (data.aciklamalar && data.aciklamalar.length > 0) {
        if (shapContainer) shapContainer.style.display = "block";
        data.aciklamalar.forEach(function(item) {
          const li = document.createElement("li");
          const isPositive = item.etki > 0;
          const sign = isPositive ? "+" : "";
          const colorClass = isPositive ? "effect-positive" : "effect-negative";
          
          li.innerHTML = `
            <span class="feature-name">${item.ozellik}</span>
            <span class="${colorClass}">${sign}${item.etki.toFixed(2)} kWh</span>
          `;
          shapList.appendChild(li);
        });
      } else {
        if (shapContainer) shapContainer.style.display = "none";
      }
      
    } else {
      $("resultHint").textContent = "Hata / Error: " + data.error;
    }
  } catch (e) {
    $("resultHint").textContent = t("noBackend");
    console.error(e);
  } finally {
    btn.textContent = t("predictBtn");
    btn.disabled = false;
  }
}

function showResult(value) {
  const valEl = $("predValue");
  valEl.textContent = value.toFixed(1);
  valEl.parentElement.classList.remove("flash");
  void valEl.parentElement.offsetWidth;
  valEl.parentElement.classList.add("flash");

  let pct = ((value - PRED_MIN) / (PRED_MAX - PRED_MIN)) * 100;
  pct = Math.min(100, Math.max(0, pct));
  $("gaugeFill").style.width = pct + "%";

  let hint;
  if (value < 65) hint = t("hintLow");
  else if (value < 82) hint = t("hintMid");
  else hint = t("hintHigh");
  $("resultHint").textContent = hint;
}

async function randomScenario() {
  try {
    const res = await fetch(`${API}/api/random`);
    const s = await res.json();
    NUMERIC_IDS.forEach(function(id) {
      if (s[id] !== undefined) {
        $(id).value = s[id];
        $(id + "_out").textContent = s[id];
      }
    });
    $("HVACUsage").value = s.HVACUsage;
    $("LightingUsage").value = s.LightingUsage;
    $("DayOfWeek").value = s.DayOfWeek;
    $("Holiday").value = s.Holiday;
    $("Month").value = s.Month;
    predict();
  } catch (e) {
    console.error(e);
  }
}

function init() {
  NUMERIC_IDS.forEach(bindSlider);
  applyTranslations();
  loadMetadata();
  $("predictBtn").addEventListener("click", predict);
  $("randomBtn").addEventListener("click", randomScenario);
  $("langTR").addEventListener("click", function() { setLang("tr"); });
  $("langEN").addEventListener("click", function() { setLang("en"); });
}

document.addEventListener("DOMContentLoaded", init);