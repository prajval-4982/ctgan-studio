# 🛡️ Privacy-Preserving Synthetic Data Generation using DP-CTGAN

## 📖 About the Project
In the modern era of machine learning, organizations face a critical dilemma: the need for massive amounts of data to train AI models versus stringent data privacy laws (such as India's DPDPA or Europe's GDPR). Sharing raw, real-world data—especially in healthcare or finance—exposes individuals to severe privacy risks. 

This research project aims to solve this **Privacy-Utility Trade-off** by developing a robust framework for generating highly realistic synthetic tabular data using Deep Learning. Instead of just replicating statistical correlations, this project guarantees privacy by integrating **Differential Privacy (DP-CTGAN)**. This restricts the framework such that the generated data cannot be reverse-engineered by malicious adversaries to identify the original humans in the dataset.

This repository serves as a fully functional and end-to-end framework capable of:
1. Sourcing and encoding raw datasets.
2. Generating privacy-preserving rows with mathematical $\varepsilon$ constraints.
3. Stress-testing the output data using state-of-the-art **Membership Inference Attacks (MIA)**.
4. Producing a composite compliance score known as the **Privacy-Utility Score (PU-Score)**.

---

## 🎓 Academic Context
- **Project Title:** Synthetic Data Generation and Evaluation Using CTGAN
- **Developer:** Prajval T Rathod (1MS24MC074)
- **Institution:** MSRIT, Department of MCA
- **Academic Year:** 2025-26

---

## 🚀 Key Research Configurations

### 1. The Core Architecture: DP-CTGAN
- **CTGAN (Conditional Tabular GAN):** The base synthesizer, optimized for capturing complex modal relationships between categorical and continuous columns.
- **DP-SGD (Differentially Private Stochastic Gradient Descent):** Injected directly into the CTGAN's **Discriminator** using Meta's `opacus` library. It clips dataset gradients and injects Gaussian noise to blind the Discriminator from memorizing records.

### 2. Adversarial Privacy Audit (MIA)
- Employs a zero-knowledge "Black-Box" shadow model.
- Evaluates if a Random Forest classifier can guess whether an external record was used during training by measuring latent Euclidean distances stringing across generated records.
- **Goal**: Push the attacker's accuracy strictly down to exactly random guessing ($\approx$ 50-60%).

### 3. The Privacy-Utility Score (PU-Score)
To effectively measure the success of DP-CTGAN against regular models, we use a novel unified metric:
- Combines Downstream ML F1-Score ($U$) with Advesarial Resistance ($R$). 
- Uses the *harmonic mean* to heavily punish models if they either utterly fail to generate useful patterns or fail to protect users.

---

## 📂 Project Structure
```text
├── app.py                # Main Flask Application
├── results/              # Final benchmark outputs
├── src/                  
│   ├── prepare_and_run.py    # Master runner (handles all pipelines)
│   └── run_experiments.py    # Backup runner
├── modules/              
│   ├── dp_ctgan_trainer.py   # The complex Opacus & PyTorch DP-CTGAN Training Loop
│   └── mia_evaluator.py      # The black-box Membership Inference logic
└── datasets/             # Directory to download and store local real datasets
```

---

## 🚦 Usage Instructions

### 1. Activate Virtual Environment (Windows PowerShell)
Before installing dependencies or running the project, you must activate the isolated environment:
```powershell
.\venv\Scripts\Activate.ps1
```
*(You will know it worked when `(venv)` appears to the left of your typing prompt).*

### 2. Install Dependencies
Ensure you are using Python 3.10+ (Tested in Python 3.11).
```bash
pip install -r requirements.txt
```

### 2. Start the Server (Web App)
If you want to use the visual, browser-based UI to generate data and view statistics:
```bash
python app.py
```
*Then open `http://127.0.0.1:5000` in your web browser.*

### 3. Run the Backend Benchmark (CLI)
If you wish to re-create the numerical results or run terminal experiments for the Adult, ILPD, and Pima datasets:
```bash
python src/prepare_and_run.py
```
*Note: Depending on CPU availability, running the multiple $\varepsilon$ tiers for all datasets via terminal takes approximately 5–8 minutes.*



## 📜 License
This project is developed for educational and research purposes as part of the MCA curriculum at MSRIT.
