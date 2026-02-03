# ⚽ Transfer Predictor – ML-driven transfermodell

Ett proof-of-concept-projekt som använder maskininlärning för att förutsäga om en fotbollstransfer blir gratis eller betald, samt att estimera transfersumman.

---

## 🚀 Kom igång

### Steg 1 – Klona repot
```bash
git clone https://github.com/SebbeHN/transfer-predictor.git
cd transfer-predictor
```

### Steg 2 – Installera beroenden
```bash
pip install -r requirements.txt
```

### Steg 3 – (Om modeller saknas) Träna modeller
Alla `.pkl`-filer finns normalt i `models/`.  
Om de saknas, kör hela notebooken:

```bash
jupyter notebook notebooks/transfers.ipynb
```

När alla celler körs sparas modellerna automatiskt i `models/`.

### Steg 4 – Starta appen
```bash
streamlit run app/app_ml.py
```

Appen laddar modeller från `models/` och erbjuder interaktiv input för att göra prediktioner.

---

## 🗂️ Projektstruktur

```
transfer-predictor/
├─ app/
│  └─ app_ml.py              # Streamlit-applikationen
├─ notebooks/
│  └─ transfers.ipynb        # Notebook med träning & analys
├─ models/                   # Sparade modeller (.pkl, via joblib/LFS)
├─ data/                     # Dataset (CSV-filer)
├─ requirements.txt
└─ README.md
```

---

## 📊 Funktioner & metodik

- **Klassificering** – gratis vs betald transfer  
  - Random Forest Classifier, Logistic Regression  
  - Utvärderat med ROC AUC & Accuracy  

- **Regression** – prediktion av transferbelopp  
  - Random Forest Regressor, Ridge Regression  
  - Utvärderat med R², MAE och RMSE  

- **Feature Engineering**  
  - Position mapping  
  - Klubb-tier (elite/top/mid/lower/unknown)  
  - Ligastyrka (proxy via snittfees)  
  - Ålder (inkl. icke-linjära termer)  
  - År & transferfönster (sommar/vinter)  
  - Läckagevariabler borttagna (t.ex. fee, transfer_type)

- **Deployment**  
  - Streamlit-app som laddar tränade `.pkl`-filer  
  - Transparens kring features & pipeline  

---

## 📦 Data

Alla dataset finns i `data/` som **CSV-filer**.  
Det är dessa som används i notebooken för att träna modellerna.

---

## ⚠ Begränsningar

- Modellen underskattar ofta extrema transfers (“supertransfers”).  
- Kontraktslängd, marknadsvärde och spelarstatistik saknas.  
- Historiska mönster kan snabbt bli inaktuella när marknaden ändras.  

---

## 📈 Framtida utveckling

- Mer avancerade modeller (XGBoost, LightGBM).  
- Inkludera kontraktslängd och spelarprestation.  
- Automatisk datahämtning + retraining.  
- Utvärdera tidsseriemodeller för marknadsförändringar.  

---

## 📝 Licens & bidrag

Detta är ett studentprojekt / proof-of-concept.  
Bidrag välkomnas via Pull Requests.
