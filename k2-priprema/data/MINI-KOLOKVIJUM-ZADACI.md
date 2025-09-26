# 🎓 MINI-KOLOKVIJUM - MAŠINSKO UČENJE

## 📝 **INSTRUKCIJE:**
Za svaki zadatak određi:
1. **Koji algoritam** bi koristio i zašto
2. **Koje su karakteristike (X)** 
3. **Šta je cilj (y)**
4. **Kakav preprocessing** treba
5. **Koju metriku** za evaluaciju

---

## 🎯 **ZADATAK 1: E-mail Spam Detekcija**

**Dataset:** 5000 e-mail poruka sa sledećim podacima:
- Tekst e-mail poruke
- Broj reči u poruci  
- Broj velikih slova
- Da li sadrži reči: "free", "winner", "urgent"
- Da li je pošaljeno noću (22-06h)
- **LABEL:** spam ili ne-spam

**❓ Tvoj odgovor:**
- Algoritam: Naivni Bayes ✅ (odlično za tekst + brojevi!)
- Karakteristike (X): Tekst poruke, broj reči, broj velikih slova, da/ne "free/winner/urgent", noćno slanje ✅
- Cilj (y): Spam (0/1 ili True/False) ✅ 
- Preprocessing: CountVectorizer za tekst, standardizacija za brojčane ✅
- Metrika: Accuracy ili F1-score ✅

---

## 🏠 **ZADATAK 2: Cene Nekretnina**

**Dataset:** 2000 stanova sa podacima:
- Kvadratura (m²)
- Broj soba
- Sprat 
- Godina izgradnje
- Deo grada (Centar, Novi Beograd, Zemun...)
- Blizina stanice (da/ne)
- **LABEL:** Cena u EUR

**❓ Tvoj odgovor:**
- Algoritam: KNN ili MLP ✅ (za numeričke podatke)
- Karakteristike (X): Kvadratura, br.soba, sprat, godina, deo grada (one-hot), blizina stanice (0/1)   
- Cilj (y): Cena u EUR (numerička - REGRESIJA!)
- Preprocessing: StandardScaler za brojeve, One-hot encoding za grad
- Metrika: MAE (Mean Absolute Error) ili RMSE

---

## 🛒 **ZADATAK 3: Segmentacija Kupaca**

**Dataset:** 10000 kupaca e-commerce sajta:
- Ukupna potrošnja (EUR)
- Broj kupovina mesečno
- Prosečna vrednost korpe
- Broj vraćenih artikala
- Vreme provedeno na sajtu (minuti)
- **NEMA LABELA!**

**Cilj:** Grupišemo kupace u kategorije za marketing kampanje

**❓ Tvoj odgovor:**
- Algoritam: K-Means ✅ (klasterovanje - NEMA LABELA!)
- Karakteristike (X): Potrošnja, br.kupovina, vrednost korpe, vraćeni artikli, vreme na sajtu  
- Cilj: Grupe kupaca (VIP, povremeni, problematični...)
- Preprocessing: StandardScaler (sve su brojevi!)
- Kako validiraš rezultate: Elbow method, Silhouette score, business logika

---

## 🎬 **ZADATAK 4: Prepoznavanje Žanra Filma**

**Dataset:** 8000 opisa filmova sa IMDb:
- Kratak opis filma (tekst)
- Trajanje filma (minuti)
- Godina izdavanja  
- Broj glumaca
- Budžet filma
- **LABEL:** žanr (Akcija, Komedija, Drama, Horror, Triler)

**❓ Tvoj odgovor:**
- Algoritam: Naive Bayes ili MLP ✅ (tekst + numerički)
- Karakteristike (X): Opis filma (CountVectorizer), trajanje, godina, br.glumaca, budžet  
- Cilj (y): Žanr (5 klasa - multiclass klasifikacija)
- Preprocessing: CountVectorizer za tekst, StandardScaler za brojeve
- Metrika: Accuracy, F1-score (macro average)

---

## 🏥 **ZADATAK 5: Dijagnostika**

**Dataset:** 1500 pacijenata sa podacima:
- Godine starosti
- Pol (M/F)
- BMI indeks
- Krvni pritisak (sistolni/dijastolni)
- Nivo holesterola  
- Da li puši (da/ne)
- Fizička aktivnost (h/nedeljno)
- **LABEL:** rizik od srčanog udara (nizak, srednji, visok)

**❓ Tvoj odgovor:**
- Algoritam: KNN ili MLP ✅ (medicinski podaci - važna preciznost!)
- Karakteristike (X): Godine, pol (0/1), BMI, pritisak, holesterol, pušenje (0/1), aktivnost  
- Cilj (y): Rizik (3 klase: nizak=0, srednji=1, visok=2)
- Preprocessing: StandardScaler za brojeve, Label encoding za pol/pušenje
- Metrika: F1-score (važno za medicinu!), Confusion matrix

---

## 📱 **ZADATAK 6: Detekcija Anomalija u Saobraćaju**

**Dataset:** GPS podaci vozila tokom dana:
- Brzina (km/h)
- Ubrzanje/usporenje  
- Vreme dana
- GPS koordinate (lat, lon)
- Tip dana (radni/vikend)
- **NEMA JASNIH LABELA** - treba detektovati čudno ponašanje

**❓ Tvoj odgovor:**
- Algoritam: DBSCAN ✅ (anomaly detection - nema labela!)
- Karakteristike (X): Brzina, ubrzanje, vreme, lat, lon, tip dana (0/1)  
- Cilj: Pronaći outlier vozila (brza vožnja, čudne rute...)
- Preprocessing: StandardScaler za sve numeričke
- Kako validiraš: Vizualizacija na mapi, domain knowledge, % anomalija

---

## 🎮 **ZADATAK 7: Prepoznavanje Emocija u Komentarima**

**Dataset:** 15000 komentara sa YouTube videa:
- Tekst komentara
- Broj lajkova komentara
- Broj odgovora
- Dužina komentara (karakteri)
- **LABEL:** emocija (pozitivna, negativna, neutralna)

**❓ Tvoj odgovor:**
- Algoritam: Naive Bayes ✅ (sentiment analysis = klassični NLP!)
- Karakteristike (X): Tekst komentara (CountVectorizer), lajkovi, odgovori, dužina  
- Cilj (y): Emocija (3 klase: pozitivna, negativna, neutralna)
- Preprocessing: CountVectorizer za tekst, StandardScaler za brojeve
- Metrika: F1-score (balanced dataset?), Accuracy

---

## 🏆 **BONUS ZADATAK: Preporučivanje Muzike**

**Dataset:** Spotify listening history:
- Žanr pesme
- Tempo (BPM)
- Danceability score
- Valence (pozitivnost)
- Vreme slušanja (jutro/veče)
- Da li je preskočena pesma
- **CILJ:** Da li će korisnik lajkovati pesmu?

**❓ Tvoj odgovor:**
- Algoritam: MLP ili KNN ✅ (recommendation system!)
- Karakteristike (X): Žanr (one-hot), tempo, danceability, valence, vreme (0/1), preskočena (0/1)  
- Cilj (y): Lajk (0/1 - binary klasifikacija)
- Preprocessing: One-hot za žanr, StandardScaler za numeričke
- Metrika: Precision/Recall (važno za preporuke!), AUC-ROC

---

## 🎯 **KAKO OCENITI SEBE:**

**Za svaki zadatak dobijašь:**
- ✅ **2 boda** - tačan algoritam + objašnjenje
- ✅ **1 bod** - ispravke u X, y, preprocessing  
- ✅ **1 bod** - odgovarajuća metrika

**UKUPNO: 32 boda**
- **28+ bodova** = Spremna/spreman! 🏆
- **20-27 bodova** = Dobro, još malo vežbe 📚  
- **<20 bodova** = Ponovi algoritme 🔄

---

## 💡 **KADA ZAVRŠIŠ, POŠALJI MI ODGOVORE DA PROVERIM!** 

Srećno! 🚀