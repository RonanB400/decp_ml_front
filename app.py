import streamlit as st
import numpy as np
import pandas as pd
import requests


st.set_page_config(page_title="Lanterne publique", layout="wide")


# Sidebar: Module selector
module = st.sidebar.radio("Choix du module :", ["Estimation du montant et marchés similaires", "Détection d'anomalies"])

# Main Panel
st.title("Business Intelligence App")

if module == "Estimation du montant et marchés similaires":
    st.header("💰 Estimation du montant et marchés similaires")
    Code_CPV = st.selectbox('Code CPV', [
        '45233140-2', '45233141-9', '45233142-6', '45233143-3',
        '45233144-0', '45233145-7', '45233146-4', '45233147-1'
    ])
    DureeMois = st.slider('Estimation de la durée en mois', 1, 48, 1)
    ccag = st.selectbox('CCAG', ['Travaux', 'Fournitures courantes et services', 'Pas de CCAG', 'Autre'])
    nature = st.selectbox('Nature du marché', ['Marché', 'Marché de partenariat', 'Marché de défense ou de sécurité'])
    procedure = st.selectbox('Procédure du marché', ['Procédure adaptée', 'Appel d\'offres ouvert', 'Marché passé sans publicité ni mise en concurrence préalable'])
    titulaire_categorie = st.selectbox("Taille de l'entreprise", ['PME', 'ETI', 'GE'])
    siret = st.number_input("Entrer le numéro SIRET", min_value=10000000000000, max_value=99999999999999, step=1, format="%d")

elif module == "Détection d'anomalies":
    st.header("🚨 Détection d'anomalies")
    st.write("Upload file or select row to flag anomalies.")


endpoint = 'https://pokeapi.co/api/v2/pokemon'

params= {
    'name': 'pikachu',
}

response = requests.get(f'{endpoint}/pikachu')

if response.status_code == 200:
    data = response.json()
    st.write(response.json()['moves'][0])
