import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

import joblib

import matplotlib.pyplot as plt
import seaborn as sns

# --- OTOMATİK DOSYA YOLU BULUCU ---
# Bu dosyanın (preProcess.py) bulunduğu klasörün tam yolunu alır
MEVCUT_KLASOR = os.path.dirname(os.path.abspath(__file__))
# Bir üst klasöre (ECon) çıkar ve 'data' klasörüne girer
ANA_KLASOR = os.path.dirname(MEVCUT_KLASOR)
DATA_KLASORU = os.path.join(ANA_KLASOR, "data")

GIRDI_YOLU = os.path.join(DATA_KLASORU, "Energy_consumption.csv")
CIKTI_YOLU = os.path.join(DATA_KLASORU, "Prepared_Data.csv")

def veri_yukle(dosya_yolu):
    print(f"[{dosya_yolu}] yükleniyor...")
    return pd.read_csv(dosya_yolu)

def zaman_ozellikleri_uret(df):
    df_new = df.copy()
    df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])
    df_new['Hour'] = df_new['Timestamp'].dt.hour
    df_new['Month'] = df_new['Timestamp'].dt.month
    
    df_new['Hour_Sin'] = np.sin(2 * np.pi * df_new['Hour'] / 24)
    df_new['Hour_Cos'] = np.cos(2 * np.pi * df_new['Hour'] / 24)
    df_new['Month_Sin'] = np.sin(2 * np.pi * df_new['Month'] / 12)
    df_new['Month_Cos'] = np.cos(2 * np.pi * df_new['Month'] / 12)

    df_new['Time_Of_Day'] = pd.cut(df_new['Hour'], bins=[-1, 5, 11, 17, 23], labels=['Gece', 'Sabah', 'Ogle', 'Aksam'])
    df_new['Is_Weekend'] = df_new['Timestamp'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    return df_new

def kategorik_verileri_kodla(df):
    df_new = df.copy()
    df_new['HVACUsage'] = df_new['HVACUsage'].map({'On': 1, 'Off': 0})
    df_new['LightingUsage'] = df_new['LightingUsage'].map({'On': 1, 'Off': 0})
    df_new['Holiday'] = df_new['Holiday'].map({'Yes': 1, 'No': 0})
    
    df_new = pd.get_dummies(df_new, columns=['DayOfWeek', 'Time_Of_Day'], drop_first=False)
    
    for col in df_new.columns:
        if df_new[col].dtype == bool:
            df_new[col] = df_new[col].astype(int)
    return df_new

def etkilesim_ozellikleri_uret(df):
    df_new = df.copy()
    df_new['HVAC_Occupancy_Interaction'] = df_new['HVACUsage'] * df_new['Occupancy']
    df_new['Comfort_Index'] = df_new['Temperature'] * (1 + (df_new['Humidity'] / 100))
    return df_new

def gecikme_ozellikleri_ekle(df):
    df_new = df.copy()
    df_new = df_new.sort_values('Timestamp') 
    df_new['Temp_Lag1'] = df_new['Temperature'].shift(1)
    df_new['Temp_Lag1'] = df_new['Temp_Lag1'].fillna(df_new['Temperature']) 
    return df_new

def sayisal_verileri_olceklendir(df):
    df_new = df.copy()
    df_new = df_new.drop(['Timestamp', 'Hour', 'Month'], axis=1)
    
    scaler = StandardScaler()
    sayisal_sutunlar = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy', 'Comfort_Index', 'Temp_Lag1']
    
    # Veriyi ölçeklendir
    df_new[sayisal_sutunlar] = scaler.fit_transform(df_new[sayisal_sutunlar])
    
    # SCALER'I KAYDET (app.py'nin kullanması için)
    # trainedModel klasörü yoksa hata vermemesi için oluştur
    trained_model_yolu = os.path.join(ANA_KLASOR, "trainedModel")
    os.makedirs(trained_model_yolu, exist_ok=True)
    joblib.dump(scaler, os.path.join(trained_model_yolu, "scaler.joblib"))
    
    return df_new

def heatmap_olustur_ve_kaydet(df, cikti_yolu):
    """
    İşlenmiş verinin korelasyon matrisini hesaplar ve 
    görseli şık bir ısı haritası (heatmap) olarak kaydeder.
    """
    print("🎨 Korelasyon Isı Haritası çiziliyor...")
    
    # Görsel boyutunu ayarlayalım (Sütun sayısı fazla olduğu için büyük tutuyoruz)
    plt.figure(figsize=(14, 12))
    
    # Korelasyon matrisini hesapla
    korelasyon_matrisi = df.corr()
    
    # Seaborn ile ısı haritasını çizdir
    # cmap='coolwarm': Negatifler mavi, pozitifler kırmızı olur
    # annot=True: Kutuların içine sayısal değerleri yazar (Çok kalabalıksa False yapabilirsiniz)
    sns.heatmap(korelasyon_matrisi, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                cbar=True, 
                square=True,
                linewidths=.5)
    
    plt.title("Özellikler Arası Korelasyon Isı Haritası (İşlenmiş Veri)", fontsize=16)
    plt.tight_layout() # Yazıların kesilmesini engeller
    
    # Görseli kaydet
    plt.savefig(cikti_yolu, dpi=300) # dpi=300 yüksek kalite sağlar
    plt.close()
    print(f"✔ Isı haritası kaydedildi: {cikti_yolu}")


def veri_isleme_boru_hatti():
    try:
        df = veri_yukle(GIRDI_YOLU)
        df = zaman_ozellikleri_uret(df)
        df = kategorik_verileri_kodla(df)
        df = etkilesim_ozellikleri_uret(df)
        df = gecikme_ozellikleri_ekle(df)
        df = sayisal_verileri_olceklendir(df)
        
        # Eğer 'data' klasörü yoksa oluştur
        os.makedirs(DATA_KLASORU, exist_ok=True)
        df.to_csv(CIKTI_YOLU, index=False)
        print(f"✔ İşlem başarılı! Dosya şuraya kaydedildi: {CIKTI_YOLU}")
        heatmap_yolu = os.path.join(ANA_KLASOR, "HEATMAP.png")
        heatmap_olustur_ve_kaydet(df, heatmap_yolu)
    except Exception as e:
        print(f"❌ Ön işleme sırasında hata: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    veri_isleme_boru_hatti()