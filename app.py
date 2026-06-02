import os
import sys
import subprocess
import random
import traceback
import numpy as np
import pandas as pd
import joblib
import shap
import warnings
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# SHAP kütüphanesinin gereksiz uyarılarını terminalde gizler
warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(APP_DIR, "frontend")

PREPROCESS_SCRIPT = os.path.join(APP_DIR, "preProcess", "preProcess.py")
MODELTRAIN_SCRIPT = os.path.join(APP_DIR, "modelTrain", "modelTrain.py")
RESIDUAL_SCRIPT = os.path.join(APP_DIR, "residualAnalysis","residualAnalysis.py")
MODEL_BUNDLE_PATH = os.path.join(APP_DIR, "trainedModel", "trainedModel.joblib")
SCALER_BUNDLE_PATH = os.path.join(APP_DIR, "trainedModel", "scaler.joblib")


GLOBAL_SCALER = None
GLOBAL_MODELLER = {}
GLOBAL_BEST_NAME = "Bilinmiyor"
GLOBAL_FEATURE_COLS = []
GLOBAL_SKORLAR = {}

def otomatik_boru_hatti_calistir():
    try:
        print("\n" + "="*50)
        print("⚙️ ADIM 1/3: Veri Ön İşleme başlatılıyor...")
        subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True)
        
        print("\n⚙️ ADIM 2/3: Model Eğitimi başlatılıyor...")
        subprocess.run([sys.executable, MODELTRAIN_SCRIPT], check=True)

        print("\n⚙️ ADIM 3/3: Kalıntı analizi başlatılıyor...")
        subprocess.run([sys.executable, RESIDUAL_SCRIPT], check=True)

        
        print("="*50 + "\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ KRİTİK HATA: Boru hattında bir sorun oluştu! {e}")
        sys.exit(1)

def modeli_yukle():
    global GLOBAL_MODELLER, GLOBAL_BEST_NAME, GLOBAL_FEATURE_COLS, GLOBAL_SCALER, GLOBAL_SKORLAR
    try:
        print(f"📦 Model paketi yükleniyor...")
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        
        if "modeller" in bundle:
            GLOBAL_MODELLER = bundle["modeller"]
        else:
            GLOBAL_MODELLER = {bundle["best_name"]: bundle["model"]}
            
        GLOBAL_BEST_NAME = bundle["best_name"]
        GLOBAL_FEATURE_COLS = bundle["feature_cols"]
        GLOBAL_SKORLAR = bundle.get("skorlar", {GLOBAL_BEST_NAME: 1.0})
        
        GLOBAL_SCALER = joblib.load(SCALER_BUNDLE_PATH)
        print(f"🚀 TOPLAM {len(GLOBAL_MODELLER)} MODEL VE SHAP HAZIR!")
    except Exception as e:
        print(f"❌ HATA: Model yüklenemedi! {e}")
        sys.exit(1)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

def veriyi_modele_hazirla(ham_veri):
    temp = float(ham_veri["Temperature"])
    hum = float(ham_veri["Humidity"])
    sqft = float(ham_veri["SquareFootage"])
    occ = int(ham_veri["Occupancy"])
    ren = float(ham_veri["RenewableEnergy"])
    hour = int(ham_veri["Hour"])
    month = int(ham_veri["Month"])
    
    hvac = 1 if ham_veri["HVACUsage"] == "On" else 0
    lighting = 1 if ham_veri["LightingUsage"] == "On" else 0
    holiday = 1 if ham_veri["Holiday"] == "Yes" else 0
    
    gunler = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    gun_index = gunler.get(ham_veri["DayOfWeek"], 0)
    is_weekend = 1 if gun_index >= 5 else 0

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    comfort_index = temp * (1 + (hum / 100))
    hvac_occ_interaction = hvac * occ
    temp_lag1 = temp 

    row = {
        "Temperature": temp, "Humidity": hum, "SquareFootage": sqft,
        "Occupancy": occ, "RenewableEnergy": ren, "HVACUsage": hvac,
        "LightingUsage": lighting, "Holiday": holiday, "Is_Weekend": is_weekend,
        "Hour_Sin": hour_sin, "Hour_Cos": hour_cos, "Month_Sin": month_sin,
        "Month_Cos": month_cos, "Comfort_Index": comfort_index,
        "HVAC_Occupancy_Interaction": hvac_occ_interaction, "Temp_Lag1": temp_lag1
    }
    df = pd.DataFrame([row])
    
    for gun in gunler.keys():
        df[f"DayOfWeek_{gun}"] = 1 if ham_veri["DayOfWeek"] == gun else 0
        
    time_of_day = "Gece" if hour <= 5 else "Sabah" if hour <= 11 else "Ogle" if hour <= 17 else "Aksam"
    for zaman in ["Gece", "Sabah", "Ogle", "Aksam"]:
        df[f"Time_Of_Day_{zaman}"] = 1 if time_of_day == zaman else 0

    for col in GLOBAL_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0 
            
    sayisal_sutunlar = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy', 'Comfort_Index', 'Temp_Lag1']
    df[sayisal_sutunlar] = GLOBAL_SCALER.transform(df[sayisal_sutunlar])
            
    return df[GLOBAL_FEATURE_COLS]

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/api/metadata")
def metadata():
    formatted_metrics = {isim: {"R2": skor} for isim, skor in GLOBAL_SKORLAR.items()}
    return jsonify({
        "best_model": GLOBAL_BEST_NAME,
        "metrics": formatted_metrics,
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
        istenen_model_ismi = ham_veri.get("selected_model", GLOBAL_BEST_NAME)
        aktif_model = GLOBAL_MODELLER.get(istenen_model_ismi, list(GLOBAL_MODELLER.values())[0])
        
        X_hazir = veriyi_modele_hazirla(ham_veri)
        tahmin = float(aktif_model.predict(X_hazir)[0])
        
    # --- ŞEFFAF AÇIKLAMALAR (HİBRİT YÖNTEM: LİNEER + AĞAÇ) ---
        aciklamalar = []
        try:
            # 1. DURUM: EĞER MODEL LİNEER İSE (Doğrudan Matematik Kullan)
            # Stacking modellerinde de coef_ vardır ama özelliklere değil, alt modellere aittir. Onu hariç tutuyoruz.
            if hasattr(aktif_model, "coef_") and "Stacking" not in istenen_model_ismi:
                katsayilar = aktif_model.coef_
                # Bazı modellerde katsayılar matris olarak döner, onu düzeltelim
                if len(katsayilar.shape) > 1: 
                    katsayilar = katsayilar[0]
                
                for i, col in enumerate(GLOBAL_FEATURE_COLS):
                    deger = float(X_hazir.iloc[0, i])
                    katsayi = float(katsayilar[i])
                    etki = katsayi * deger # StandardScaler sayesinde SHAP değerine tam eşittir!
                    
                    if abs(etki) > 0.1:
                        aciklamalar.append({"ozellik": col, "etki": round(etki, 2)})
                        
            # 2. DURUM: EĞER MODEL AĞAÇ İSE (SHAP TreeExplainer Kullan)
            elif hasattr(aktif_model, "feature_importances_"):
                explainer = shap.TreeExplainer(aktif_model)
                shap_values = explainer.shap_values(X_hazir)
                
                for i, col in enumerate(GLOBAL_FEATURE_COLS):
                    etki = float(shap_values[0][i])
                    if abs(etki) > 0.1: 
                        aciklamalar.append({"ozellik": col, "etki": round(etki, 2)})
            
            # 3. DURUM: EĞER SÜPER MODEL (STACKING) İSE (Yedek Ağaç Kullan)
            else:
                yedek_model = GLOBAL_MODELLER.get("Random Forest")
                # İsim eşleşmezse diye listedeki ilk ağacı bul
                if yedek_model is None:
                    for m in GLOBAL_MODELLER.values():
                        if hasattr(m, "feature_importances_"):
                            yedek_model = m
                            break
                            
                explainer = shap.TreeExplainer(yedek_model)
                shap_values = explainer.shap_values(X_hazir)
                
                for i, col in enumerate(GLOBAL_FEATURE_COLS):
                    etki = float(shap_values[0][i])
                    if abs(etki) > 0.1: 
                        aciklamalar.append({"ozellik": col, "etki": round(etki, 2)})

            # Bulunan tüm etkileri büyükten küçüğe sırala
            aciklamalar = sorted(aciklamalar, key=lambda x: abs(x["etki"]), reverse=True)[:4]
            
        except Exception as e:
            print(f"Açıklama Üretilirken Hata: {e}")
            print(traceback.format_exc())
        # ---------------------------------------------------------

        return jsonify({
            "success": True,
            "prediction": round(tahmin, 2),
            "unit": "kWh",
            "model": istenen_model_ismi,
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
    heatmap_path = os.path.join(APP_DIR, "HEATMAP.png")
    if os.path.exists(heatmap_path):
        return send_file(heatmap_path, mimetype='image/png')
    return "Isı haritası bulunamadı", 404

@app.route("/api/residual")
def get_residual():
    residual_path = os.path.join(APP_DIR, "residualAnalysis", "RESIDUALS.png")
    if os.path.exists(residual_path):
        return send_file(residual_path, mimetype='image/png')
    return "Kalıntı haritası bulunamadı", 404

if __name__ == "__main__":
    otomatik_boru_hatti_calistir()
    modeli_yukle()
    print("🌐 Web sunucusu başlatılıyor... http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)