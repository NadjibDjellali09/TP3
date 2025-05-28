import streamlit as st
import pandas as pd
import bnlearn as bn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("dataset_risque_cardiaque.csv")
    replace_map = {
        'Oui': 1, 'Non': 0,
        'Homme': 1, 'Femme': 0,
        'Jeune': 0, 'Moyen': 1, 'Adulte': 2
    }
    return data.replace(replace_map)

# Build and train the Bayesian model
@st.cache_resource
def train_model(data):
    edges = [
        ('Age', 'Risque_cardiaque'),
        ('Sexe', 'Risque_cardiaque'),
        ('Tabagisme', 'Risque_cardiaque'),
        ('Hypertension', 'Risque_cardiaque'),
        ('Cholesterol_eleve', 'Risque_cardiaque'),
        ('Antecedents_familiaux', 'Risque_cardiaque'),
        ('Activite_physique', 'Risque_cardiaque'),
        ('Diabete', 'Risque_cardiaque'),
        ('Stress_chronique', 'Risque_cardiaque')
    ]
    DAG = bn.make_DAG(edges)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    model = bn.parameter_learning.fit(DAG, train_data)

    # Evaluate model
    pred_df = bn.predict(model, test_data.drop(columns='Risque_cardiaque'), variables=['Risque_cardiaque'], verbose=False)
    acc = accuracy_score(test_data['Risque_cardiaque'].astype(int), pred_df['Risque_cardiaque'].astype(int))
    return model, acc

# UI Setup
st.set_page_config(page_title="Cardiac Risk Predictor", layout="centered")
st.title("🫀 Cardiac Risk Prediction using Bayesian Network")

data = load_data()
model, acc = train_model(data)

# Form inputs
with st.form("risk_form"):
    st.subheader("Enter Patient Information:")

    age = st.selectbox("Age Group", ["Jeune", "Moyen", "Adulte"])
    sexe = st.selectbox("Sexe", ["Homme", "Femme"])
    tabagisme = st.selectbox("Fumeur ?", ["Oui", "Non"])
    hypertension = st.selectbox("Hypertension ?", ["Oui", "Non"])
    cholesterol = st.selectbox("Cholestérol élevé ?", ["Oui", "Non"])
    antecedents = st.selectbox("Antécédents familiaux ?", ["Oui", "Non"])
    activite = st.selectbox("Activité physique régulière ?", ["Oui", "Non"])
    diabete = st.selectbox("Diabète ?", ["Oui", "Non"])
    stress = st.selectbox("Stress chronique ?", ["Oui", "Non"])

    submitted = st.form_submit_button("Prédire le risque")

    if submitted:
        map_vals = {'Oui': 1, 'Non': 0, 'Homme': 1, 'Femme': 0, 'Jeune': 0, 'Moyen': 1, 'Adulte': 2}
        evidence = {
            'Age': map_vals[age],
            'Sexe': map_vals[sexe],
            'Tabagisme': map_vals[tabagisme],
            'Hypertension': map_vals[hypertension],
            'Cholesterol_eleve': map_vals[cholesterol],
            'Antecedents_familiaux': map_vals[antecedents],
            'Activite_physique': map_vals[activite],
            'Diabete': map_vals[diabete],
            'Stress_chronique': map_vals[stress],
        }

        result = bn.inference.fit(model, variables=['Risque_cardiaque'], evidence=evidence)
        st.success("Inférence terminée !")

        st.write("### 📊 Résultat de l'inférence")
        st.dataframe(result.values.round(4), use_container_width=True)

# Accuracy display
st.sidebar.title("ℹ️ Modèle")
st.sidebar.write(f"**Accuracy sur les données de test** : {acc:.2%}")
