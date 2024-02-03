import shap
from flask import Flask, jsonify, request, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import pickle

# Charger le modèle pickle
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    if request.method == 'POST':
        try:
            data_df = request.get_json(force=True)
            df = pd.DataFrame([data_df["data"]], columns=data_df["keys"])
            prediction = model.predict_proba(df)[:, 1]
            prediction = (prediction * 100).round(2)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df)

            plt.figure()
            shap.force_plot(explainer.expected_value, shap_values[0], df, matplotlib=True)
            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_data = base64.b64encode(img.getvalue()).decode()

            return jsonify({'prediction': prediction.tolist(), 'shap_plot': 'data:image/png;base64,' + plot_data})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/predict_proba', methods=['GET'])
def predict_proba_get():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)