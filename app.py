import streamlit as st
import numpy as np
import pandas as pd


st.set_page_config(page_title="Lanterne publique", layout="wide")


# Sidebar: Module selector
module = st.sidebar.radio("Choix du module :", ["Estimation du montant et marchés similaires", "Détection d'anomalies"])

# Main Panel
st.title("Business Intelligence App")

if module == "Estimation du montant et marchés similaires":
    st.header("💰 Estimation du montant et marchés similaires")
    Code_CPV = st.slider('Code CPV')
    DureeMois = st.slider('Estimation de la durée')
    ccag = st.selectbox('CCAG', ['Travaux', 'Fournitures courantes et services', 'Pas de CCAG', 'Autre'])
    nature = st.selectbox('Nature du marché', ['Marché', 'Marché de partenariat', 'Marché de défense ou de sécurité'])
    procedure = st.selectbox('Procédure du marché', ['Procédure adaptée', 'Appel d\'offres ouvert', 'Marché passé sans publicité ni mise en concurrence préalable'])
    acheteur_categorie = st.selectbox("taille de l'entreprise", ['PME', 'ETI', 'GE'])

elif module == "Détection d'anomalies":
    st.header("🚨 Détection d'anomalies")
    st.write("Upload file or select row to flag anomalies.")
