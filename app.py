import shap
from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 'Agg' est un backend non interactif
import matplotlib.pyplot as plt
import io
import base64
import pickle
#from joblib import load

# Charger le modèle joblib
model = pickle.load(open('model.pkl', 'rb'))


# Créer une application Flask
app = Flask(__name__)


# Définir une route pour effectuer des prédictions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Obtenir les données JSON de la requête
            data_df = request.get_json(force=True)
            df = pd.DataFrame([data_df["data"]], columns=data_df["keys"])
            feature_names = data_df["keys"]
            # Effectuer la prédiction
            prediction = model.predict_proba(df)[:, 1]
            prediction = (prediction*100).round(2)

            explainer = shap.Explainer(model)
            shap_values = explainer.shap_values(df)

            # Créer un plot SHAP et le convertir en image PNG encodée en base64
            plt.figure()
            shap.force_plot(explainer.expected_value, shap_values[0], feature_names=feature_names, matplotlib=True)
            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_data = base64.b64encode(img.getvalue()).decode()

            # Renvoyer les prédictions au format JSON
            return jsonify({'prediction': prediction.tolist(), 'shap_plot': 'data:image/png;base64,' + plot_data})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        # Servir la page HTML pour les requêtes GET
        return render_template('index.html')

# Exécuter l'application Flask sur le port 7000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)