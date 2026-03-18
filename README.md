# 🛡️ CTGAN Studio: Synthetic Data Generation & Evaluation

A comprehensive, full-stack application for generating high-fidelity synthetic data using Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). Built for security-conscious data science workflows where privacy and data utility are paramount.

---

## 🎓 Project Context
- **Project:** Synthetic Data Generation and Evaluation using CTGAN
- **Developer:** Prajval T Rathod (1MS24MC074)
- **Institution:** MSRIT, Department of MCA
- **Academic Year:** 2025-26
- **Credits:** 10 (Final Year Solo Project)

---

## 🚀 Key Features

### 1. Dual-Model Architecture
- **CTGAN (Conditional GAN):** Optimized for capturing complex relationships between categorical and continuous columns using adversarial training.
- **TVAE (Tabular VAE):** A variational autoencoder approach that often provides faster training and superior preservation of statistical distributions.

### 2. Comprehensive Results Dashboard
- **Synthetic Preview:** Real-time exploration of generated records.
- **Statistical Fidelity:** Automated side-by-side comparison of Means, Standard Deviations, and Correlation Heatmaps.
- **Visual Analytics:** Distribution plots for every column to verify data alignment.

### 3. Academic "Boosters" (Advanced Evaluation)
- **🛡️ Privacy & Security Audit:** DCR (Distance to Closest Record) analysis to detect and prevent "memorization" or data leakage.
- **🎯 Feature Importance Alignment:** Uses Random Forest models to prove that the synthetic data has preserved the underlying logic of the original features.
- **🤖 ML Utility Scoring:** Benchmarks synthetic data against Real data using Logistic Regression, Decision Trees, and Random Forests.

---

## 🛠️ Tech Stack
- **Backend:** Flask (Python)
- **AI Core:** SDV (Synthetic Data Vault), CTGAN, TVAE
- **Computation:** Pandas, NumPy, Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **UI:** Minimalist Monochrome (Vercel/Apple Style)

---

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.9 or higher installed.

### 2. Clone the Repository
```bash
git clone <repository-url>
cd synthetic-data-ctgan
```

### 3. Setup Virtual Environment (Recommended)
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚦 Usage Instructions

1. **Start the Server:**
   ```bash
   python app.py
   ```
2. **Access the App:** Open `http://127.0.0.1:5000` in your browser.
3. **The Workflow:**
   - **Upload:** Select a raw CSV dataset (e.g., `adult_income.csv`).
   - **Generate:** Choose your Model (CTGAN or TVAE), set Epochs (Recommended 300+ for high fidelity), and number of rows.
   - **Evaluate:** Review the Privacy Score and ML Utility on the dashboard.
   - **Export:** Download your synthetic data in CSV, Excel, JSON, or DOCX formats.

---

## 📂 Project Structure
```text
├── app.py                # Main Flask Application
├── modules/              # Core Logic Modules
│   ├── ctgan_trainer.py  # Model training (CTGAN/TVAE)
│   ├── privacy_evaluator.py # DCR & Privacy Scoring
│   ├── ml_evaluator.py    # Feature Importance & ML Metrics
│   └── preprocessing.py   # Data cleaning & encoding
├── static/               # Visual assets & Generated plots
├── templates/            # HTML Dashboards
└── datasets/             # Input & Cleaned datasets
```

---

## 📜 License
This project is developed for educational purposes as part of the MCA curriculum at MSRIT.
