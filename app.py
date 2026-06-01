import os
import sys
import subprocess
import random
import traceback
import numpy as np
import pandas as pd
import joblib
import shap
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask import send_file

# =====================================================================
# 1. DOSYA YOLLARI VE KLASÖR AYARLARI (Mutlak Yollar)
# =====================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(APP_DIR, "frontend")

PREPROCESS_SCRIPT = os.path.join(APP_DIR, "preProcess", "preProcess.py")
MODELTRAIN_SCRIPT = os.path.join(APP_DIR, "modelTrain", "modelTrain.py")
MODEL_BUNDLE_PATH = os.path.join(APP_DIR, "trainedModel", "trainedModel.joblib")
SCALER_BUNDLE_PATH = os.path.join(APP_DIR, "trainedModel", "scaler.joblib")

# Global değişkenler (Sunucu kalktıktan sonra dolacak)
GLOBAL_MODELLER = {} # Tek bir model yerine sözlük tutacağız
GLOBAL_SCALER = None
GLOBAL_MODEL = None
GLOBAL_BEST_NAME = "Bilinmiyor"
GLOBAL_SHAP_MODEL = None
GLOBAL_FEATURE_COLS = []
GLOBAL_SKORLAR = {}

# =====================================================================
# 2. OTONOM BORU HATTI (PIPELINE) TETİKLEYİCİSİ
# =====================================================================
def otomatik_boru_hatti_calistir():
    """Flask ayağa kalkmadan önce işlemleri sırayla ve güvenle çalıştırır."""
    try:
        print("\n" + "="*50)
        print("⚙️ ADIM 1/2: Veri Ön İşleme (preProcess.py) başlatılıyor...")
        subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True)
        print("✔ Veri ön işleme başarıyla tamamlandı!")

        print("\n⚙️ ADIM 2/2: Model Eğitimi (modeltrain.py) başlatılıyor...")
        subprocess.run([sys.executable, MODELTRAIN_SCRIPT], check=True)
        print("✔ En iyi model eğitildi ve paketlendi!")
        print("="*50 + "\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ KRİTİK HATA: Boru hattında bir sorun oluştu!")
        print(f"Lütfen yukarıdaki hataları kontrol edin. İşlem durduruluyor.")
        sys.exit(1)

def modeli_yukle():
    global GLOBAL_MODELLER, GLOBAL_BEST_NAME, GLOBAL_FEATURE_COLS, GLOBAL_SCALER, GLOBAL_SKORLAR
    try:
        print(f"📦 Model paketi yükleniyor...")
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        
        GLOBAL_MODELLER = bundle["modeller"] # TÜM MODELLERİ YÜKLE
        GLOBAL_BEST_NAME = bundle["best_name"]
        GLOBAL_FEATURE_COLS = bundle["feature_cols"]
        GLOBAL_SKORLAR = bundle.get("skorlar", {})
        
        GLOBAL_SCALER = joblib.load(SCALER_BUNDLE_PATH)
        print(f"🚀 TOPLAM {len(GLOBAL_MODELLER)} MODEL HAZIR!")
    except Exception as e:
        print(f"❌ HATA: Model yüklenemedi! {e}")
        sys.exit(1)
# =====================================================================
# 3. FLASK UYGULAMASI VE VERİ İŞLEME
# =====================================================================
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

def veriyi_modele_hazirla(ham_veri):
    """Arayüzden gelen ham JSON verisini, modelin eğitimde gördüğü formata çevirir."""
    
    # 1. Ham değerler
    temp = float(ham_veri["Temperature"])
    hum = float(ham_veri["Humidity"])
    sqft = float(ham_veri["SquareFootage"])
    occ = int(ham_veri["Occupancy"])
    ren = float(ham_veri["RenewableEnergy"])
    hour = int(ham_veri["Hour"])
    month = int(ham_veri["Month"])
    
    # 2. Kategorik Dönüşümler
    hvac = 1 if ham_veri["HVACUsage"] == "On" else 0
    lighting = 1 if ham_veri["LightingUsage"] == "On" else 0
    holiday = 1 if ham_veri["Holiday"] == "Yes" else 0
    
    gunler = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    gun_index = gunler.get(ham_veri["DayOfWeek"], 0)
    is_weekend = 1 if gun_index >= 5 else 0

    # 3. Yeni Özellikler (Feature Engineering)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    comfort_index = temp * (1 + (hum / 100))
    hvac_occ_interaction = hvac * occ
    temp_lag1 = temp 

    # DataFrame'i Oluşturma
    row = {
        "Temperature": temp, "Humidity": hum, "SquareFootage": sqft,
        "Occupancy": occ, "RenewableEnergy": ren, "HVACUsage": hvac,
        "LightingUsage": lighting, "Holiday": holiday, "Is_Weekend": is_weekend,
        "Hour_Sin": hour_sin, "Hour_Cos": hour_cos, "Month_Sin": month_sin,
        "Month_Cos": month_cos, "Comfort_Index": comfort_index,
        "HVAC_Occupancy_Interaction": hvac_occ_interaction, "Temp_Lag1": temp_lag1
    }
    
    df = pd.DataFrame([row])
    
    # One-Hot Encoding (Günler ve Günün Bölümleri)
    for gun in gunler.keys():
        col_name = f"DayOfWeek_{gun}"
        df[col_name] = 1 if ham_veri["DayOfWeek"] == gun else 0
        
    if hour <= 5: time_of_day = "Gece"
    elif hour <= 11: time_of_day = "Sabah"
    elif hour <= 17: time_of_day = "Ogle"
    else: time_of_day = "Aksam"
    
    for zaman in ["Gece", "Sabah", "Ogle", "Aksam"]:
        col_name = f"Time_Of_Day_{zaman}"
        df[col_name] = 1 if time_of_day == zaman else 0

    # Sütunları Güvence Altına Alma
    for col in GLOBAL_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0 
            

    sayisal_sutunlar = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy', 'Comfort_Index', 'Temp_Lag1']
    df[sayisal_sutunlar] = GLOBAL_SCALER.transform(df[sayisal_sutunlar])
            
    # Modelin eğitimde gördüğü sıraya diz
    return df[GLOBAL_FEATURE_COLS]


# =====================================================================
# 4. API UÇ NOKTALARI (ENDPOINTS)
# =====================================================================
@app.route("/")
def index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return "<h1>HATA: index.html bulunamadı!</h1>", 404
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/api/metadata")
def metadata():
    formatted_metrics = {isim: {"R2": skor} for isim, skor in GLOBAL_SKORLAR.items()}

    return jsonify({
        "best_model": GLOBAL_BEST_NAME,
        "metrics": formatted_metrics, # Sabit veri yerine formatladığımız sözlüğü koyduk
        "numeric_ranges": {
            "Temperature": {"min": 15, "max": 40},
            "Humidity": {"min": 20, "max": 80},
            "SquareFootage": {"min": 800, "max": 5000},
            "Occupancy": {"min": 0, "max": 20},
            "RenewableEnergy": {"min": 0, "max": 100}
        },
        "categorical_options": {
            "HVACUsage": ["On", "Off"],
            "LightingUsage": ["On", "Off"],
            "DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "Holiday": ["No", "Yes"]
        },
        "hour_range": [0, 23],
        "month_range": [1, 12]
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        ham_veri = request.get_json(force=True)
        
        # 1. Hangi modelin seçildiğini al (Gelmezse varsayılan olarak Süper Modeli kullan)
        istenen_model_ismi = ham_veri.get("selected_model", GLOBAL_BEST_NAME)
        aktif_model = GLOBAL_MODELLER.get(istenen_model_ismi, GLOBAL_MODELLER[GLOBAL_BEST_NAME])
        
        # 2. Veriyi modele hazırla ve tahmin yap
        X_hazir = veriyi_modele_hazirla(ham_veri)
        tahmin = float(aktif_model.predict(X_hazir)[0])
        
        # 3. SHAP Açıklamaları (Sadece o modele ait açıklamalar)
        aciklamalar = []
        try:
            explainer = shap.Explainer(aktif_model, X_hazir)
            shap_values = explainer(X_hazir).values
            for i, col in enumerate(GLOBAL_FEATURE_COLS):
                etki = float(shap_values[0][i])
                if abs(etki) > 0.1: 
                    aciklamalar.append({"ozellik": col, "etki": round(etki, 2)})
            aciklamalar = sorted(aciklamalar, key=lambda x: abs(x["etki"]), reverse=True)[:4]
        except Exception:
            pass # Bazı modeller (KNN gibi) SHAP desteklemez, hata verirse es geç.

        return jsonify({
            "success": True,
            "prediction": round(tahmin, 2),
            "unit": "kWh",
            "model": istenen_model_ismi, # Ekrana seçilen modeli yazdır
            "aciklamalar": aciklamalar 
        })
    except Exception as e:
        print("--- TAHMİN HATASI ---")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/random")
def random_sample():
    return jsonify({
        "Temperature": round(random.uniform(20.0, 32.0), 1),
        "Humidity": round(random.uniform(30.0, 60.0), 1),
        "SquareFootage": random.randint(1000, 3000),
        "Occupancy": random.randint(0, 15),
        "RenewableEnergy": round(random.uniform(0.0, 25.0), 1),
        "Hour": random.randint(0, 23),
        "Month": random.randint(1, 12),
        "HVACUsage": random.choice(["On", "Off"]),
        "LightingUsage": random.choice(["On", "Off"]),
        "DayOfWeek": random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
        "Holiday": random.choice(["Yes", "No"])
    })

@app.route("/api/heatmap")
def get_heatmap():
    """Uygulama klasöründeki HEATMAP.png dosyasını frontend'e gönderir."""
    heatmap_path = os.path.join(APP_DIR, "HEATMAP.png")
    if os.path.exists(heatmap_path):
        return send_file(heatmap_path, mimetype='image/png')
    else:
        return "Isı haritası bulunamadı", 404

# =====================================================================
# 5. UYGULAMAYI BAŞLATMA
# =====================================================================
if __name__ == "__main__":
    # 1. İşleme ve eğitim scriptlerini çalıştır
    otomatik_boru_hatti_calistir()
    
    # 2. Yeni eğitilen model paketini yükle
    modeli_yukle()
    
    # 3. Web sitesini ayağa kaldır
    print("🌐 Web sunucusu başlatılıyor... http://127.0.0.1:5000 adresine gidin.")
    app.run(host="127.0.0.1", port=5000, debug=False) 
    # Not: debug=True kullanıldığında Flask otonom işlemleri 2 kere çalıştırabilir, bu yüzden kapalı.