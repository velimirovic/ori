# 🎓 KOMPLETAN REZIME - MAŠINSKO UČENJE (ORI)

## 📚 **ŠTA STE SVE NAUČILI:**

---

## 🎯 **1. NADGLEDANO UČENJE (Supervised Learning)**
*Učimo iz podataka sa unapred definisanim oznakama (X → y)*

### 🏷️ **KLASIFIKACIJA** (predviđamo kategorije)

#### **🔢 K-Nearest Neighbors (KNN)**
- **Šta radi:** Pronađi K najbližih suseda i glasaj koji je najčešći
- **Kada koristiti:** Mali dataset, jednostavan problem
- **Primer:** Prepoznavanje cifara (0-9) iz slika
- **Prednosti:** Jednostavan, radi za male podatke
- **Mane:** Spor za velike podatke, loš za visoke dimenzije

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

#### **🧠 Naivni Bayes (Naive Bayes)**  
- **Šta radi:** Računa verovatnoće na osnovu Bayesove teoreme
- **Kada koristiti:** Text klasifikacija, spam detekcija
- **Primer:** Određivanje ko govori rečenicu (Cartman, Stan, Kyle...)
- **Prednosti:** Brz, dobar za tekst, manje podataka potrebno
- **Mane:** "Naivan" - pretpostavlja nezavisnost features

```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_bow, y_train)  # BoW = Bag of Words
y_pred = nb.predict(X_test_bow)
```

#### **🌐 Neuronske mreže (MLP - Multi-Layer Perceptron)**
- **Šta radi:** Imitira mozak - slojevi neurona koji uče obrasce
- **Kada koristiti:** Složeni problemi, veliko podataka
- **Primer:** Predviđanje zvanja profesora, klasifikacija teksta
- **Prednosti:** Vrlo moćan, uči složene obrasce
- **Mane:** Zahteva više podataka, sporiji trening

```python
import torch.nn as nn
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        # Definiši slojove...
    def forward(self, x):
        # Prosledi podatke kroz slojevi...
```

---

## 🔍 **2. NENADGLEDANO UČENJE (Unsupervised Learning)**
*Tražimo skrivene obrasce u podacima BEZ oznaka (samo X, nema y)*

### 📊 **KLASTEROVANJE** (grupisanje sličnih podataka)

#### **⭕ K-Means**
- **Šta radi:** Deli podatke u K grupa (klastera)
- **Kada koristiri:** Eksplorativna analiza, grupisanje kupaca, segmentacija
- **Primer:** Grupisanje cvetova Iris po sličnosti
- **Prednosti:** Jednostavan, brz
- **Mane:** Mora unapred znati broj grupa (K), radi samo za sferne oblike

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_  # kom klasteru pripada svaki element
```

#### **🎯 DBSCAN**
- **Šta radi:** Grupiše na osnovu gustine, automatski detektuje šum
- **Kada koristiti:** Kada ne znaš broj grupa, neregularni oblici klastera
- **Primer:** Detekcija anomalija, grupisanje geografskih podataka
- **Prednosti:** Ne treba unapred K, detektuje šum, bilo koji oblik
- **Mane:** Teško podesiti parametre (eps, min_samples)

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan.fit(X)
labels = dbscan.labels_  # -1 = šum/outlier
```

---

## 🛠️ **3. PREPROCESSING TEHNIKE**

### **📊 Za numeričke podatke:**
- **StandardScaler:** (vrednost - srednja) / std_devijacija
- **One-hot encoding:** kategorije → binarne kolone
- **Ordinal encoding:** kategorije → brojevi sa redosledom

### **📝 Za tekst (NLP):**
- **Bag of Words (BoW):** prebroji reči u rečenici
- **CountVectorizer:** automatski BoW sa filtriranjem
- **Stop words:** ukloni česte reči ("the", "and", "is")

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words="english")
X_bow = vectorizer.fit_transform(texts)
```

---

## 📋 **4. ZADACI IZ LEKCIJA:**

### **🔢 Zadatak 1 (Tvoj kolokvijum):**
- **Problem:** Predvideti zvanje profesora (AsstProf, Prof, AssocProf)
- **Algoritam:** MLP Neuronska mreža
- **Features:** Numerički (plata, godine, pol, oblast)
- **Preprocessing:** StandardScaler + One-hot encoding
- **Metrika:** F1-score (87.78%)

### **📝 Zadatak 2 (Tvoj kolokvijum):**
- **Problem:** Ko govori rečenicu u South Park-u?
- **Algoritmi:** Naivni Bayes + MLP
- **Features:** Tekst (rečenice)
- **Preprocessing:** Bag of Words (CountVectorizer)
- **Metrika:** Accuracy (41% NB, 38% MLP)

### **🌸 Iris Classification (iz lekcija):**
- **Problem:** Klasifikacija vrsta cvetova
- **Algoritmi:** KNN, K-Means (klasterovanje)
- **Features:** petal length, petal width, sepal length, sepal width
- **Cilj:** Učenje osnovnih algoritama

### **🔢 Digit Recognition (iz lekcija):**
- **Problem:** Prepoznavanje cifara (0-9) iz slika 8x8 piksela
- **Algoritam:** KNN
- **Features:** 64 piksela (8x8 slike)
- **Cilj:** Primer computer vision-a

### **🏦 Bank Dataset (zadatak iz lekcije):**
- **Problem:** Eksplorativna analiza bankovnih podataka
- **Algoritam:** K-Means klasterovanje
- **Cilj:** Segmentacija kupaca, otkrivanje grupa

### **🎨 Image Compression (bonus zadatak):**
- **Problem:** Kompresija slike redukcijom boja
- **Algoritam:** K-Means
- **Cilj:** Praktična primena - od 16M boja → K boja

---

## 🎯 **5. KADA KORISTITI KOJI ALGORITAM:**

### **📈 Klasifikacija:**
- **KNN:** Mali dataset, jednostavan problem
- **Naivni Bayes:** Tekst, spam detekcija, brza potreba
- **MLP:** Složeni problemi, dovoljno podataka, visoka preciznost

### **📊 Klasterovanje:**
- **K-Means:** Znaš broj grupa, sferni klasteri, brza analiza
- **DBSCAN:** Ne znaš broj grupa, neregularni oblici, detekcija anomalija

### **🔤 Tekst:**
- **Bag of Words + Naivni Bayes:** Standardna kombinacija
- **Bag of Words + MLP:** Kad treba veća preciznost

---

## 🏆 **6. METRIKE EVALUACIJE:**

- **Accuracy:** % tačnih predviđanja (za balansiraane dataset)
- **F1-score:** Kombinuje precision i recall (za nebalansiraane)
- **Confusion Matrix:** Detaljna analiza grešaka po klasama

---

## 💡 **7. PRAKTIČNI SAVETI:**

### **🔄 Uvek:**
1. **Podeli podatke:** train_test_split
2. **Preprocess:** fit_transform(train), transform(test)
3. **Treniraj:** model.fit(X_train, y_train)
4. **Evaluiraj:** accuracy_score(y_test, y_pred)

### **⚠️ Nikad:**
- Ne fit-uj preprocessing na test podacima
- Ne testiraj na train podacima
- Ne zaboravi random_state za reproducibilnost

---

## 🚀 **8. ŠTA NISTE PROŠLI (ali postoji):**

- **Support Vector Machines (SVM)**
- **Random Forest / Decision Trees**
- **Logistic Regression**
- **Deep Learning (CNN, RNN)**
- **Reinforcement Learning**
- **Principal Component Analysis (PCA)**

---

## 📝 **QUICK REFERENCE:**

```python
# OSNOVNI TEMPLATE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Podeli podatke
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Preprocess
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit na train
X_test = scaler.transform(X_test)        # transform na test

# 3. Treniraj model
model.fit(X_train, y_train)

# 4. Evaluiraj
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

🎉 **GOTOV SI ZA KOLOKVIJUM!** 💪