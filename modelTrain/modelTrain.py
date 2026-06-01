import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import os
import joblib

# Doğrusal, Uzaklık ve Vektör Modelleri (GERİ EKLENDİ)
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Ağaç Tabanlı ve Ensemble Modeller
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# --- OTOMATİK DOSYA YOLU BULUCU ---
MEVCUT_KLASOR = os.path.dirname(os.path.abspath(__file__))
ANA_KLASOR = os.path.dirname(MEVCUT_KLASOR) 

DATA_KLASORU = os.path.join(ANA_KLASOR, "data")
DOSYA_YOLU = os.path.join(DATA_KLASORU, "Prepared_Data.csv") 

TRAINED_MODEL_KLASORU = os.path.join(ANA_KLASOR, "trainedModel")
MODEL_CIKTI_YOLU = os.path.join(TRAINED_MODEL_KLASORU, "trainedModel.joblib")

def trainTest(dosya_yolu):
    print("1. Veri yükleniyor...")
    try:
        df = pd.read_csv(dosya_yolu)
    except FileNotFoundError:
        print(f"❌ HATA: {dosya_yolu} bulunamadı!")
        import sys
        sys.exit(1)
    
    X = df.drop('EnergyConsumption', axis=1)
    y = df['EnergyConsumption']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Tüm modeller sahnede
    modeller = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=42),
        "Lasso Regression": Lasso(random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Support Vector Regressor": SVR(kernel='rbf'),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, objective='reg:squarederror', verbosity=0),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0)
    }
    
    skorlar = {}
    print("2. Çapraz Doğrulama (CV) Sınavları Başlıyor...\n")
    
    for isim, model in modeller.items():
        try:
            cv_sonuclari = cross_val_score(model, X, y, cv=kf, scoring='r2')
            ortalama_r2 = cv_sonuclari.mean()
            skorlar[isim] = ortalama_r2
            print(f"{isim:25s} | Ortalama R2: {ortalama_r2:.4f}")
        except Exception as e:
            pass
            
    print("-" * 55)
    
    # 🌟 İLERİ SEVİYE 1: EN İYİ 3 MODELİ SEÇME
    top_3_isimler = sorted(skorlar, key=skorlar.get, reverse=True)[:3]
    print(f"🥇 Şampiyonlar Ligi (Top 3): {top_3_isimler}")
    
    # 🌟 İLERİ SEVİYE 2: TOP 3 İÇİN DİNAMİK OPTİMİZASYON (Grid)
    param_grids = {
        "Random Forest": {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        "Extra Trees": {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 7]},
        "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 7]},
        "LightGBM": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [-1, 5, 10]},
        "CatBoost": {'iterations': [100, 200], 'learning_rate': [0.05, 0.1], 'depth': [4, 6, 8]},
        "Ridge Regression": {'alpha': [0.1, 1.0, 10.0]},
        "Lasso Regression": {'alpha': [0.01, 0.1, 1.0]},
        "K-Nearest Neighbors": {'n_neighbors': [3, 5, 7]},
        "Support Vector Regressor": {'C': [0.1, 1, 10]},
        "AdaBoost": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1, 1.0]}
    }
    
    optimize_modeller = []
    
    print("\n⚙️ İlk 3 model için hiperparametre optimizasyonu yapılıyor...")
    for model_ismi in top_3_isimler:
        ham_model = modeller[model_ismi]
        izgara = param_grids.get(model_ismi, {}) # Modele özel parametreleri al
        
        # Eğer modelin optimize edilecek parametresi yoksa (Örn: Linear Regression) direkt listeye ekle
        if not izgara:
            print(f"   ► {model_ismi} optimize edilecek parametreye sahip değil, doğrudan alınıyor.")
            ham_model.fit(X, y)
            optimize_modeller.append((model_ismi, ham_model))
            continue
            
        print(f"   ► {model_ismi} optimize ediliyor...")
        # Süreyi kısa tutmak için n_iter=5 kullanıyoruz
        optimizasyon = RandomizedSearchCV(ham_model, izgara, n_iter=5, cv=3, scoring='r2', random_state=42)
        optimizasyon.fit(X, y)
        
        en_iyi_versiyon = optimizasyon.best_estimator_
        optimize_modeller.append((model_ismi, en_iyi_versiyon))
        print(f"     ✔ En iyi parametreler: {optimizasyon.best_params_}")

    # ... (Yukarıdaki optimizasyon kodları aynı kalacak) ...
    
    # 🌟 İLERİ SEVİYE 3: OPTİMİZE EDİLMİŞ MODELLERLE STACKING REGRESSOR
    print("\n🏗️ Optimize edilmiş 3 model birleştirilerek 'Süper Model' (Stacking) oluşturuluyor...")
    super_model = StackingRegressor(estimators=optimize_modeller, final_estimator=Ridge())
    super_model.fit(X, y)
    print("✔ Süper Model başarıyla eğitildi!")
    
    # 🌟 YENİ: Arayüzden seçilebilmesi için TÜM modelleri tam veriyle eğitip sözlükte topluyoruz
    kaydedilecek_modeller = {}
    print("\n📦 Tüm modeller canlı kullanım için paketleniyor...")
    for isim, ham_model in modeller.items():
        ham_model.fit(X, y)
        kaydedilecek_modeller[isim] = ham_model
        
    # Süper Modeli de başköşeye ekliyoruz
    kaydedilecek_modeller["Süper Model (Stacking)"] = super_model
    
    return kaydedilecek_modeller, list(X.columns), skorlar

if __name__ == "__main__":
    tum_modeller, feature_cols, skorlar = trainTest(DOSYA_YOLU)
    
    os.makedirs(TRAINED_MODEL_KLASORU, exist_ok=True)
    
    bundle = {
        "modeller": tum_modeller, # Artık tek bir model değil, tüm modellerin sözlüğü gidiyor
        "best_name": "Süper Model (Stacking)",
        "feature_cols": feature_cols,
        "skorlar": skorlar
    }
    
    joblib.dump(bundle, MODEL_CIKTI_YOLU)
    print(f"\n💾 12 Modelin tamamı {MODEL_CIKTI_YOLU} konumuna başarıyla kaydedildi!")