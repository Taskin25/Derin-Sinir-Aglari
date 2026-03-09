import numpy as np
import pickle
import os
import warnings

warnings.filterwarnings("ignore", message=".*align should be passed.*")


def load_batch(file):
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        data = d[b'data']
        labels = d[b'labels']
    return data, labels


print("--- CIFAR-10 k-NN Sınıflandırma Ödevi ---")

data_path = "cifar-10-batches-py"

X_train, y_train = load_batch(os.path.join(data_path, "data_batch_1"))
X_test, y_test = load_batch(os.path.join(data_path, "test_batch"))

X_train = np.array(X_train[:5000], dtype=np.float64)
y_train = np.array(y_train[:5000])

X_test = np.array(X_test[:20], dtype=np.float64)
y_test = np.array(y_test[:20])

print(f"Sistem hazır: {len(X_train)} eğitim örneği yüklendi.\n")

print("Mesafe Hesaplama Türünü Seçiniz:")
print("1 - L1 (Manhattan)")
print("2 - L2 (Öklid)")
choice = input("Seçiminiz (1 veya 2): ")

k = int(input("k komşu sayısını giriniz: "))

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

dogru_tahmin = 0

print("\nSonuçlar Hesaplanıyor...\n" + "=" * 40)

for i in range(len(X_test)):
    test_resmi = X_test[i]

    if choice == '1':
        mesafeler = np.sum(np.abs(X_train - test_resmi), axis=1)
    else:
        mesafeler = np.sqrt(np.sum((X_train - test_resmi) ** 2, axis=1))

    en_yakin_indeksler = np.argsort(mesafeler)[:k]
    en_yakin_etiketler = y_train[en_yakin_indeksler]

    oylar = np.bincount(en_yakin_etiketler, minlength=10)
    tahmin_edilen_sinif = np.argmax(oylar)

    gercek_ad = classes[y_test[i]]
    tahmin_ad = classes[tahmin_edilen_sinif]
    durum = "BAŞARILI" if y_test[i] == tahmin_edilen_sinif else "HATALI"

    if durum == "BAŞARILI":
        dogru_tahmin += 1

    print(f"Resim {i + 1:2}: Gerçek: {gercek_ad:12} | Tahmin: {tahmin_ad:12} -> {durum}")

basari_orani = (dogru_tahmin / len(X_test)) * 100
print("=" * 40)
print(f"Ödev Tamamlandı. Toplam Başarı: %{basari_orani}")
