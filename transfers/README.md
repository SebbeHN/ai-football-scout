# Transfer Predictor - ML Transfer Model

A proof-of-concept project using machine learning to predict whether a football transfer will be free or paid, and to estimate the transfer fee.

---

## Getting Started

### Step 1 - Clone the repo
```bash
git clone https://github.com/SebbeHN/ai-football-scout.git
cd ai-football-scout/transfers
```

### Step 2 - Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 - (If models are missing) Train models
All `.pkl` files are normally in `models/`.  
If missing, run the full notebook:

```bash
jupyter notebook notebook/transfers.ipynb
```

When all cells run, models are saved automatically to `models/`.

### Step 4 - Start the app
```bash
streamlit run app/app_ml.py
```

The app loads models from `models/` and offers interactive input for predictions.

---

## Project Structure

```
transfers/
├─ app/
│  └─ app_ml.py              # Streamlit application
├─ notebook/
│  └─ transfers.ipynb        # Notebook with training & analysis
├─ models/                   # Saved models (.pkl via joblib)
├─ data/                     # Dataset (CSV files)
├─ requirements.txt
└─ README.md
```

---

## Features & Methods

- **Classification** - free vs paid transfer  
  - Random Forest Classifier  
  - Evaluated with ROC AUC & Accuracy  

- **Regression** - transfer fee prediction  
  - Random Forest Regressor  
  - Evaluated with R2, MAE and RMSE  

- **Feature Engineering**  
  - Position mapping  
  - Club tier (elite/top/mid/lower/unknown)  
  - League strength  
  - Age factors  
  - Transfer window (summer/winter)  

---

## Data

All datasets are in `data/` as CSV files from major European leagues.

---

## Limitations

- Model may underestimate extreme transfers  
- Contract length and player statistics not included  
- Historical patterns can become outdated  

---

## License

Student project / proof-of-concept.
