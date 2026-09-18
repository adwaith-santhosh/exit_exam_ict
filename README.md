# Revenue Predictor

Simple Flask app to make revenue predictions using a pretrained regressor (`model.pkl`) and scaler (`std_scaler.pkl`).

Setup:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run locally:

```bash
python app.py
```

Open http://127.0.0.1:5000/ in your browser.

Notes:
- Place `model.pkl` and `std_scaler.pkl` in the project root (same folder as `app.py`).
- If your model expects a fixed set of dummy columns, save them to `model_columns.json` (list of strings) in the project root to ensure alignment.
