import streamlit as st
import pandas as pd
import bnlearn as bn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Caching dataset load
@st.cache_data
def load_data():
    data = pd.read_csv("dataset_risque_cardiaque.csv")
    replace_map = {
        'Oui': 1, 'Non': 0,
        'Homme': 1, 'Femme': 0,
        'Jeune': 0, 'Moyen': 1, 'Adulte': 2
    }
    return data.replace(replace_map)

# Caching model training
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

    # Accuracy
    pred_df = bn.predict(
        model,
        test_data.drop(columns='Risque_cardiaque'),
        variables=['Risque_cardiaque'],
        verbose=False
    )
    acc = accuracy_score(test_data['Risque_cardiaque'].astype(int), pred_df['Risque_cardiaque'].astype(int))
    return model, acc

# Page config
st.set_page_config(page_title="🫀 Risque Cardiaque", layout="centered")
st.title("🫀 Prédiction du Risque Cardiaque avec un Réseau Bayésien")

# Load data and train model
data = load_data()
model, acc = train_model(data)

# Interface utilisateur
with st.form("formulaire"):
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

    if st.form_submit_button("🧠 Prédire le Risque"):
        map_vals = {'Oui': 1, 'Non': 0, 'Homme': 1, 'Femme': 0, 'Jeune': 0, 'Moyen': 1, 'Adulte': 2}
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
        st.write("### 📊 Probabilité de Risque Cardiaque")
        st.dataframe(result.values.round(4))

# Sidebar info
st.sidebar.title("ℹ️ Modèle")
st.sidebar.markdown(f"**Accuracy (ensemble de test)** : `{acc:.2%}`")
