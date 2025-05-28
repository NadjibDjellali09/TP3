import streamlit as st
import pandas as pd
import joblib
import bnlearn as bn

# Load model from disk
@st.cache_resource
def load_model():
    return joblib.load("model_bayes.pkl")

model = load_model()

# Mapping
map_vals = {'Oui': 1, 'Non': 0, 'Homme': 1, 'Femme': 0, 'Jeune': 0, 'Moyen': 1, 'Adulte': 2}

# UI
st.title("🫀 Prédiction du Risque Cardiaque")

with st.form("input_form"):
    st.subheader("🧾 Informations du patient")

    age = st.selectbox("Tranche d'âge", ["Jeune", "Moyen", "Adulte"])
    sexe = st.selectbox("Sexe", ["Homme", "Femme"])
    tabac = st.selectbox("Fumeur ?", ["Oui", "Non"])
    htn = st.selectbox("Hypertension ?", ["Oui", "Non"])
    chol = st.selectbox("Cholestérol élevé ?", ["Oui", "Non"])
    ant_fam = st.selectbox("Antécédents familiaux ?", ["Oui", "Non"])
    sport = st.selectbox("Activité physique régulière ?", ["Oui", "Non"])
    diab = st.selectbox("Diabète ?", ["Oui", "Non"])
    stress = st.selectbox("Stress chronique ?", ["Oui", "Non"])

    if st.form_submit_button("Prédire"):
        evidence = {
            'Age': map_vals[age],
            'Sexe': map_vals[sexe],
            'Tabagisme': map_vals[tabac],
            'Hypertension': map_vals[htn],
            'Cholesterol_eleve': map_vals[chol],
            'Antecedents_familiaux': map_vals[ant_fam],
            'Activite_physique': map_vals[sport],
            'Diabete': map_vals[diab],
            'Stress_chronique': map_vals[stress]
        }

        result = bn.inference.fit(model, variables=['Risque_cardiaque'], evidence=evidence)
        st.success("✅ Inférence réalisée avec succès")
        st.write("### 📊 Résultat")
        st.dataframe(result.values.round(4))
