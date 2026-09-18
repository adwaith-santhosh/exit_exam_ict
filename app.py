from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'std_scaler.pkl')
COLUMNS_JSON = os.path.join(BASE_DIR, 'model_columns.json')

model = None
scaler = None
model_columns = None


def load_artifacts():
	global model, scaler, model_columns
	# load model
	try:
		if os.path.exists(MODEL_PATH):
			model = joblib.load(MODEL_PATH)
	except Exception:
		model = None
	# load scaler
	try:
		if os.path.exists(SCALER_PATH):
			scaler = joblib.load(SCALER_PATH)
	except Exception:
		scaler = None
	# load expected columns (if saved) or infer from model
	try:
		if os.path.exists(COLUMNS_JSON):
			with open(COLUMNS_JSON, 'r') as f:
				model_columns = json.load(f)
		elif model is not None and hasattr(model, 'feature_names_in_'):
			model_columns = list(model.feature_names_in_)
		else:
			model_columns = None
	except Exception:
		model_columns = None


def parse_year(val):
	if val is None:
		return 0
	s = str(val).strip()
	if s == '':
		return 0
	try:
		if '-' in s:
			return int(s.split('-')[0])
		return int(float(s))
	except Exception:
		return 0


def prepare_input(form):
	# input features (exclude target `revenue`)
	keys = [
		'company_type', 'employees', 'employees_year', 'revenue_currency', 'revenue_year',
		'net_income', 'net_income_currency', 'net_income_year', 'market_cap', 'market_cap_currency',
		'market_cap_date', 'legal_units_count', 'direct_subsidiaries_count', 'max_hierarchy_depth',
		'is_standalone_entity'
	]

	data = {}
	for k in keys:
		v = form.get(k, '')
		if k in ['company_type', 'revenue_currency', 'net_income_currency', 'market_cap_currency', 'is_standalone_entity']:
			data[k] = v if v != '' else 'unknown'
		elif k in ['market_cap_date', 'employees_year', 'revenue_year', 'net_income_year']:
			data[k] = parse_year(v)
		else:
			try:
				data[k] = float(v) if v not in [None, ''] else 0.0
			except Exception:
				data[k] = 0.0

	df = pd.DataFrame([data])
	# one-hot encode the categorical columns
	cat_cols = ['company_type', 'revenue_currency', 'net_income_currency', 'market_cap_currency', 'is_standalone_entity']
	df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

	# align to expected model columns if available
	if model_columns is not None:
		for c in model_columns:
			if c not in df.columns:
				df[c] = 0
		df = df[model_columns]
	else:
		df = df.reindex(sorted(df.columns), axis=1)

	X = df.values.astype(float)
	if scaler is not None:
		try:
			X = scaler.transform(X)
		except Exception:
			pass

	# Ensure feature dimension matches what the model expects.
	# Preferred: user provides `model_columns.json` so dummies align exactly.
	expected_n = None
	if model is not None:
		expected_n = getattr(model, 'n_features_in_', None)
		if expected_n is None:
			# fallback: many linear models have coef_ length
			try:
				expected_n = int(np.array(getattr(model, 'coef_', [])).shape[0])
			except Exception:
				expected_n = None

	if expected_n is not None:
		cur_n = X.shape[1]
		if cur_n < expected_n:
			# pad with zeros on the right
			pad = np.zeros((X.shape[0], expected_n - cur_n), dtype=float)
			X = np.hstack([X, pad])
		elif cur_n > expected_n:
			# trim extra columns (best-effort)
			X = X[:, :expected_n]

	return X


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'GET':
		return render_template('predict.html')

	if model is None:
		return render_template('result.html', value='Model not loaded on server')

	X = prepare_input(request.form)
	try:
		pred = model.predict(X)
		value = float(pred[0])
	except Exception as e:
		return render_template('result.html', value=f'Prediction error: {e}')

	# format and show result on a dedicated page
	return redirect(url_for('result', value=value))


@app.route('/result')
def result():
	val = request.args.get('value', None)
	try:
		num = float(val)
		pretty = f"{num:,.2f}"
	except Exception:
		pretty = val
	return render_template('result.html', value=pretty)


if __name__ == '__main__':
	load_artifacts()
	app.run(debug=True, host='0.0.0.0', port=5000)
