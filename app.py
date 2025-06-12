import streamlit as st
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt


custom_css = """
<style>
/* App background and main text */
[data-testid="stAppViewContainer"] {
    background-color: #EAF1F2;
    color: #1F2A30;}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #4D6E75;
    color: white;}

label, [data-testid*="label"] {color: #1F2A30 !important;}

h1, h2, h3, h4, h5, h6 {color: #000000;}

/* Buttons */
.stButton > button {
    background-color: #799DA7;
    color: white;
    border-radius: 6px;
    border: none;}

/* Input fields */
input, textarea, .stTextInput > div > input, .stNumberInput input {
    background-color: #ffffff;
    color: black; /* dark text */
    border: 1px solid #799DA7;}

/* Selectboxes and dropdowns */
.css-1wa3eu0-placeholder, .css-qc6sy-singleValue { color: #1F2A30 !important;}

/* Highlight areas (e.g. alerts) */
.stAlert {
    background-color: #A77979;
    color: white;}

/* Default button style */
div.stButton > button {
    background-color: #4D6E75;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 0.5em 1em;
    border: none;
    transition: background-color 0.3s ease;}

/* Hover effect */
div.stButton > button:hover {
    background-color: #3B6C87;
    color: white;}

/* Active (clicked) state */
div.stButton > button:active {
    background-color: #2F566D;
    color: white;}

</style>

"""
footer_css = """
<style>
footer {visibility: hidden;}
.footer {
    position: fixed;    bottom: 0;
    left: 0;            width: 100%;
    padding: 10px 0;    background-color: #799DA7;
    color: white;       text-align: center;
    font-size: 14px;    z-index: 100;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


endpoint_clusters = 'https://decp-708609074810.europe-west1.run.app/api/predict'
endpoint_estimation = 'https://decp-708609074810.europe-west1.run.app/api/montant'

tranches_effectif = {
    'NN' : "Unité non-employeuse ou présumée non-employeuse (faute de déclaration reçue)",
    '00' : "0 salarié (ayant employé mais aucun au 31/12)",
    '01' : "1 ou 2 salariés",
    '02' : "3 à 5 salariés",
    '03' : "6 à 9 salariés",
    '11' : "10 à 19 salariés",
    '12' : "20 à 49 salariés",
    '21' : "50 à 99 salariés",
    '22' : "100 à 199 salariés",
    '31' : "200 à 249 salariés",
    '32' : "250 à 499 salariés",
    '41' : "500 à 999 salariés",
    '42' : "1 000 à 1 999 salariés",
    '51' : "2 000 à 4 999 salariés",
    '52' : "5 000 à 9 999 salariés",
    '53' : "10 000 salariés et plus",
    'null' : 'Donnée manquante ou "sans objet"'
}
options = {v: k for k, v in tranches_effectif.items()}
id = 10
# Sidebar: Module selector
module = st.sidebar.radio("Choix du module :", ["Exploration des données", "Estimation du montant et marchés similaires"])

# Main Panel
st.title("Lanterne publique")

if module == "Estimation du montant et marchés similaires":
    st.header("💰 Estimation du montant et marchés similaires")
    st.write("Utilisez ce module pour estimer le montant d'un futur marché public et trouver des marchés similaires.")
    st.write("Les champs obligatoires sont marqués d'un astérisque (*)")
    Code_CPV = st.selectbox('Code CPV', [
        3000000,  9000000, 15000000, 18000000, 22000000, 30000000, 31000000,
        32000000, 33000000, 34000000, 37000000, 39000000, 42000000, 44000000,
        45000000, 45100000, 45200000, 45300000, 45400000, 48000000, 50000000,
        55000000, 60000000, 64000000, 66000000, 71000000, 71200000, 71300000,
        71400000, 71600000, 72000000, 74000000, 75000000, 77000000, 79000000,
        80000000, 85000000, 90000000, 92000000, 98000000
 ])
    DureeMois = st.slider('Estimation de la durée en mois*', 1, 48, 6)
    OffresRecues = st.number_input('Nombre d\'offres reçues*', min_value=0, max_value=100, value=3, step=1)
    ccag = st.selectbox('CCAG', ['Travaux', 'Fournitures courantes et services', 'Pas de CCAG', 'Autre'])
    nature = st.selectbox('Nature du marché*', ['Marché', 'Marché de partenariat', 'Marché de défense ou de sécurité'])
    formePrix = st.selectbox('Forme du prix*', ['Forfaitaire', 'Unitaire', 'Mixte'])
    procedure = st.selectbox('Procédure du marché*', ['Procédure adaptée', 'Appel d\'offres ouvert', 'Marché passé sans publicité ni mise en concurrence préalable'])
    titulaire_categorie = st.selectbox("Taille de l'entreprise", ['PME', 'ETI', 'GE'])
    siret = st.number_input("Entrer le numéro SIRET", min_value=10000000000000, max_value=99999999999999, value=80866548300018, step=1, format="%d")
    effectif = st.selectbox("Choisir une tranche d'effectif :", list(options.keys()))

    st.write("Le montant estimé du marché est de :")
    if st.button("Estimer le montant"):
        params = {
            "dureeMois": DureeMois,
            "offresRecues": OffresRecues,
            "procedure": procedure,
            "nature": nature,
            "formePrix": formePrix,
            "ccag": ccag,
            "codeCPV_3": Code_CPV,
            "acheteur_tranche_effectif" : options[effectif],
            "annee": 2025,
            "sousTraitanceDeclaree": 0,
            "origineFrance": 0,
            "marcheInnovant": 0,
            "idAccordCadre": 0,
            "typeGroupementOperateurs": "Pas de groupement",
            "tauxAvance": 0,
            "acheteur_categorie": titulaire_categorie
            }
        response = requests.post(endpoint_estimation, json=params)
        if response.status_code == 200:
            data = response.json()
            # On récupère les prédictions de la première instance
            prob_raw = data["prediction"][0]  # Liste de N valeurs
            probabilities = np.array(prob_raw)

            # Supposons 100 bins sur une échelle log (par exemple de 6 à 18 log-euros)
            bins = np.linspace(6, 18, len(probabilities) + 1)  # len(bins) = N+1
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # Création du DataFrame
            df = pd.DataFrame({
                'bin_center': np.exp(bin_centers),
                'probability': probabilities
            })
            df['smoothed'] = df['probability'].rolling(window=10, center=True, min_periods=1).mean()

            # Affichage avec Streamlit
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=df, x='bin_center', y='smoothed', label='Lissé', color='blue', ax=ax)
            # sns.scatterplot(data=df, x='bin_center', y='probability', color='gray', alpha=0.5, label='Original', ax=ax)
            ax.set_title("Distribution prédite des montants")
            ax.set_xlabel("Montant (€)")
            ax.set_ylabel("Probabilité")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        else:
            st.error("Erreur lors de l'estimation du montant. Veuillez vérifier les paramètres et réessayer.")

    if st.button("Voir les marchés similaires"):
        params = {
            "montant": 0,
            "dureeMois": DureeMois,
            "offresRecues": OffresRecues,
            "procedure": procedure,
            "nature": nature,
            "formePrix": formePrix,
            "ccag": ccag,
            "codeCPV_2_3": Code_CPV
            }
        response = requests.post(endpoint_clusters, json=params)
        if response.status_code == 200:
            data = response.json()
            st.write(data)
        else:
            st.error("Erreur lors de la récupération des marchés similaires. Veuillez vérifier les paramètres et réessayer.")

elif module == "Exploration des données":
    st.header("🔍 Exploration des données")





footer_html = """
<div class="footer">
    Le wagon batch #1992 -  Ronan Bernard, Paul Colas, Loïc Dogon, Julie Hallez – 13 Juin 2025
</div>
"""

# Display both
st.markdown(footer_css, unsafe_allow_html=True)
st.markdown(footer_html, unsafe_allow_html=True)
