1. Environment

Установите Python 3.9+ и выполните:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt


---

2. EDA and figures

`Neural_Network_PJ.ipynb`:
- график потерь (`train_losses`)
- график метрик (`val_metrics`)
- матрица корреляций
- график распределения классов

---

3. Inference

- `torch.save(model.state_dict(), "models/best_model.pt")`
- `joblib.dump(scaler, "models/scaler.pkl")`
- `inference.py` 

---
