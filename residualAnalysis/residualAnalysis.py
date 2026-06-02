import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import warnings
import joblib
warnings.filterwarnings('ignore')

# --- YOLLAR ---
MEVCUT_KLASOR = os.path.dirname(os.path.abspath(__file__))
DATA_KLASORU = os.path.join(MEVCUT_KLASOR,"..", "data")
DOSYA_YOLU = os.path.join(DATA_KLASORU, "Prepared_Data.csv")
CIKTI_YOLU = os.path.join(MEVCUT_KLASOR, "RESIDUALS.png")
MODEL_DOSYA_YOLU = os.path.join(MEVCUT_KLASOR,"..", "trainedModel","trainedModel.joblib")

def kalinti_analizi_yap():
    print("\n" + "="*60)
    print(" 📊 KALINTI (RESIDUAL) ANALİZİ BAŞLATILIYOR")
    print("="*60)
    
    try:
        # 1. Veriyi Yükle ve Böl
        print("1. İşlenmiş Veri Yükleniyor...")
        df = pd.read_csv(DOSYA_YOLU)
        
        X = df.drop('EnergyConsumption', axis=1)
        y = df['EnergyConsumption']
        
        # Test için verinin %20'sini ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Modeli Eğit ve Tahmin Yap
        print("2. Model Eğitiliyor...")
        bundle = joblib.load(MODEL_DOSYA_YOLU)
        en_iyi_isim = bundle["best_name"]
        en_iyi_model = bundle["modeller"][en_iyi_isim]
        y_pred = en_iyi_model.predict(X_test)
        
      
        
        # 3. KALINTILARI (HATALARI) HESAPLA: Gerçek - Tahmin
        kalintilar = y_test - y_pred
        
        # 4. GÖRSELLEŞTİRME (Kalıntı Grafiği)
        print("3. Kalıntı Grafikleri Çiziliyor...")
        plt.figure(figsize=(14, 6))
        
        # Sol Grafik: Tahminler vs Kalıntılar (Scatter Plot)
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_pred, y=kalintilar, alpha=0.6, color="#4CAF50")
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.title('Tahmin Edilen Değerler vs. Hatalar (Kalıntılar)', fontsize=12)
        plt.xlabel('Modelin Tahmini (kWh)', fontsize=10)
        plt.ylabel('Hata Miktarı (Gerçek - Tahmin)', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Sağ Grafik: Kalıntıların Dağılımı (Histogram)
        plt.subplot(1, 2, 2)
        sns.histplot(kalintilar, kde=True, color="#2196F3", bins=30)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title('Hataların Dağılımı (Normal Dağılım Testi)', fontsize=12)
        plt.xlabel('Hata Miktarı', fontsize=10)
        plt.ylabel('Frekans (Görülme Sıklığı)', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(CIKTI_YOLU, dpi=300)
        plt.close()
        
        print(f"\n✔ İŞLEM BAŞARILI! Grafik şuraya kaydedildi: {CIKTI_YOLU}")
  
        
    except FileNotFoundError:
        print(f"❌ HATA: {DOSYA_YOLU} bulunamadı!")

if __name__ == "__main__":
    kalinti_analizi_yap()