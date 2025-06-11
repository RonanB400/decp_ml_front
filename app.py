import streamlit as st
import numpy as np
import pandas as pd
import requests


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



# Sidebar: Module selector
module = st.sidebar.radio("Choix du module :", ["Exploration des données", "Estimation du montant et marchés similaires"])

# Main Panel
st.title("Lanterne publique")

if module == "Estimation du montant et marchés similaires":
    st.header("💰 Estimation du montant et marchés similaires")
    st.write("Utilisez ce module pour estimer le montant d'un futur marché public et trouver des marchés similaires.")
    Code_CPV = st.selectbox('Code CPV', [
        3000000,  9000000, 15000000, 18000000, 22000000, 30000000, 31000000,
        32000000, 33000000, 34000000, 37000000, 39000000, 42000000, 44000000,
        45000000, 45100000, 45200000, 45300000, 45400000, 48000000, 50000000,
        55000000, 60000000, 64000000, 66000000, 71000000, 71200000, 71300000,
        71400000, 71600000, 72000000, 74000000, 75000000, 77000000, 79000000,
        80000000, 85000000, 90000000, 92000000, 98000000
 ])
    DureeMois = st.slider('Estimation de la durée en mois', 1, 48, 6)
    OffresRecues = st.number_input('Nombre d\'offres reçues', min_value=0, max_value=100, value=3, step=1)
    ccag = st.selectbox('CCAG', ['Travaux', 'Fournitures courantes et services', 'Pas de CCAG', 'Autre'])
    nature = st.selectbox('Nature du marché', ['Marché', 'Marché de partenariat', 'Marché de défense ou de sécurité'])
    formePrix = st.selectbox('Forme du prix', ['Forfaitaire', 'Unitaire', 'Mixte'])
    procedure = st.selectbox('Procédure du marché', ['Procédure adaptée', 'Appel d\'offres ouvert', 'Marché passé sans publicité ni mise en concurrence préalable'])
    titulaire_categorie = st.selectbox("Taille de l'entreprise", ['PME', 'ETI', 'GE'])
    siret = st.number_input("Entrer le numéro SIRET", min_value=10000000000000, max_value=99999999999999, value=80866548300018, step=1, format="%d")

    st.write("Le montant estimé du marché est de :")
    #if st.button("Estimer le montant"):
        #st.write(requests.get(f'{endpoint}/pikachu'))

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
