**Hybrid Markov-LSTM Threat Modelling Framework**  
**A Hybrid Markov-LSTM Threat Modelling Framework for Predicting Lateral Movement in Zero Trust Environments Using Policy Violation Artifacts**  
MTech Cyber Security Science — Federal University of Technology, Minna  
  
---  
**Results Summary**  
  
|  |  |  |  |  |  
| - | - | - | - | - |  
| **Model** |  **Accuracy** |  **F1-Score** |  **AUC-ROC** |  **Latency** |  
| Markov Chain |  78.2% |  72.6% |  0.812 |  0.23ms |  
| LSTM |  89.7% |  87.4% |  0.932 |  4.87ms |  
| **Hybrid Markov-LSTM** |  **93.4%** |  **92.0%** |  **0.963** |  **5.02ms** |  
  
   
  
---  
**Project Structure**  
hybrid-markov-lstm-zt/  
   
 ├── data/  
   
 │ ├── zt_policy_violations.csv # Synthetic dataset (6,016 events)  
   
 │ └── processed/ # Preprocessed train/val/test splits  
   
 ├── models/  
   
 │ ├── lstm_model.keras # Trained LSTM  
   
 │ ├── markov_model.pkl # Trained Markov Chain  
   
 │ └── scaler.pkl # StandardScaler  
   
 ├── notebooks/  
   
 │ └── hybrid_markov_lstm_zt.ipynb # Full runnable notebook  
   
 ├── results/  
   
 │ ├── figures/ # 10 evaluation figures (PNG)  
   
 │ └── tables/ # 11 result tables (CSV)  
   
 ├── src/  
   
 │ ├── generate_dataset.py # Synthetic data generator  
   
 │ ├── data_preprocessing.py # Preprocessing pipeline  
   
 │ ├── markov_model.py # Markov Chain model  
   
 │ ├── lstm_model.py # LSTM model  
   
 │ ├── hybrid_model.py # Hybrid ensemble  
   
 │ └── evaluation.py # Evaluation and figures  
   
 ├── app.py # Streamlit interactive app  
   
 ├── requirements.txt  
   
 └── README.md  
  
---  
**Architecture**  
Raw CSV (6,016 events, 57 features)  
   
 ↓  
   
 Preprocessing → 28 features, 70/15/15 split, SMOTE  
   
 ↓  
   
 ┌─────────────────┬──────────────────┐  
   
 │ Markov Chain │ LSTM │  
   
 │ Transition │ 128→64→32→3 │  
   
 │ Matrix P\[i\]\[j\] │ tanh, softmax │  
   
 └─────────────────┴──────────────────┘  
   
 ↓ ↓  
   
 P_hybrid = 0.35 × P_markov + 0.65 × P_lstm  
   
 ↓  
   
 Evaluation: Acc, F1, AUC, FPR, Latency  
  
---  
**Quick Start**  
**1. Clone the repository**  
```  
git clone https://github.com/Famous005/hybrid-markov-lstm-zt.git  
 cd hybrid-markov-lstm-zt  
   
```  
**2. Create virtual environment (Python 3.11)**  
```  
python3.11 -m venv venv  
 source venv/bin/activate  
   
```  
**3. Install dependencies**  
```  
pip install -r requirements.txt  
   
```  
**4. Run the interactive app**  
```  
streamlit run app.py  
   
```  
**5. Run the full pipeline**  
```  
python src/generate_dataset.py  
 python src/data_preprocessing.py  
 python src/markov_model.py  
 python src/lstm_model.py  
 python src/hybrid_model.py  
 python src/evaluation.py  
   
```  
**6. Open the Jupyter notebook**  
```  
jupyter notebook notebooks/hybrid_markov_lstm_zt.ipynb  
   
```  
  
---  
**Zero Trust Concepts**  
**Policy Violation Artifacts**: Every denied access request and authentication failure logged by the Policy Enforcement Point (NIST SP 800-207) becomes a predictive signal for lateral movement detection.  
**Lateral Movement**: An attacker's progression through a network after initial compromise — the most critical phase to detect in a Zero Trust Architecture.  
**Why Hybrid?**: The Markov Chain provides interpretable state-transition explanations a security analyst can audit. The LSTM captures complex multi-step attack patterns spanning 10+ events. Together they exceed either model alone on all accuracy metrics.  
  
---  
**Requirements**  
- Python 3.11  
- TensorFlow 2.x (CPU)  
- scikit-learn, pandas, numpy  
- matplotlib, seaborn  
- streamlit  
- imbalanced-learn  
See `requirements.txt` for exact versions.  
  
---  
**Live Demo**  
Deployed on Streamlit Cloud: *(link added after Phase 11 deployment)*  
  
---  
