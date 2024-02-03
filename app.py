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

def roundVal(n):
    return (n * 100).round(2)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    if request.method == 'POST':
        try:
            data_df = request.get_json(force=True)
            #print(f"shape data df data: {data_df["data"].shape]})
            
                       
            data = list(map(float, data_df["data"]))
            data[-1]=int(data[-1])
            print(f"data est: \n: {data}")

            df = pd.DataFrame([data], columns=data_df["keys"])
            df= df.drop('SK_ID_CURR', axis=1)
            print(f'df est:\n {df}')

            prediction = model.predict_proba(df)[:, 1]
            print(prediction)
            prediction = list(map(roundVal, prediction))
            print(prediction)                          


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

            return jsonify({'prediction': prediction, 'shap_plot': 'data:image/png;base64,' + plot_data})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/predict_proba', methods=['GET'])
def predict_proba_get():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)