import streamlit as st
import numpy as np
import pandas as pd
import requests
from graph_plot_builder import GraphPlotBuilder
import os


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


endpoint_clusters = (
    'https://decp-708609074810.europe-west1.run.app/api/predict'
)



# Sidebar: Module selector
module = st.sidebar.radio(
    "Choix du module :", 
    ["Exploration des données", "Estimation du montant et marchés similaires"]
)

# Main Panel
st.title("Lanterne publique")

if module == "Estimation du montant et marchés similaires":
    st.header("💰 Estimation du montant et marchés similaires")
    st.write(
        "Utilisez ce module pour estimer le montant d'un futur marché "
        "public et trouver des marchés similaires."
    )
    Code_CPV = st.selectbox('Code CPV', [
        3000000,  9000000, 15000000, 18000000, 22000000, 30000000, 31000000,
        32000000, 33000000, 34000000, 37000000, 39000000, 42000000, 44000000,
        45000000, 45100000, 45200000, 45300000, 45400000, 48000000, 50000000,
        55000000, 60000000, 64000000, 66000000, 71000000, 71200000, 71300000,
        71400000, 71600000, 72000000, 74000000, 75000000, 77000000, 79000000,
        80000000, 85000000, 90000000, 92000000, 98000000
    ])
    DureeMois = st.slider('Estimation de la durée en mois', 1, 48, 6)
    OffresRecues = st.number_input(
        'Nombre d\'offres reçues', 
        min_value=0, max_value=100, value=3, step=1
    )
    ccag = st.selectbox(
        'CCAG', 
        ['Travaux', 'Fournitures courantes et services', 'Pas de CCAG', 'Autre']
    )
    nature = st.selectbox(
        'Nature du marché', 
        ['Marché', 'Marché de partenariat', 'Marché de défense ou de sécurité']
    )
    formePrix = st.selectbox(
        'Forme du prix', 
        ['Forfaitaire', 'Unitaire', 'Mixte']
    )
    procedure = st.selectbox(
        'Procédure du marché', 
        ['Procédure adaptée', 'Appel d\'offres ouvert', 
         'Marché passé sans publicité ni mise en concurrence préalable']
    )
    titulaire_categorie = st.selectbox(
        "Taille de l'entreprise", 
        ['PME', 'ETI', 'GE']
    )
    siret = st.number_input(
        "Entrer le numéro SIRET", 
        min_value=10000000000000, 
        max_value=99999999999999, 
        value=80866548300018, 
        step=1, 
        format="%d"
    )

    st.write("Le montant estimé du marché est de :")
    # if st.button("Estimer le montant"):
    #     st.write(requests.get(f'{endpoint}/pikachu'))

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
    
    # Part 1: RAG Query
    st.subheader("💬 Interroger la base de données")
    st.write("Posez une question sur les marchés publics et obtenez une réponse basée sur nos données.")
    
    question = st.text_area(
        "Votre question :", 
        placeholder="Ex: Quels sont les principaux codeCPV et leurs signification ?",
        height=100
    )
    
    if st.button("Poser la question"):
        if question.strip():
            with st.spinner("Recherche en cours..."):
                try:
                    rag_endpoint = (
                        'https://decp-708609074810.europe-west1.run.app'
                        '/api/rag'
                    )
                    payload = {"query": question}
                    response = requests.post(rag_endpoint, json=payload)
                    
                    if response.status_code == 200:
                        answer = response.json()
                        st.success("Réponse trouvée !")
                        st.write("**Réponse :**")
                        st.write(answer.get("response", answer))
                    else:
                        st.error(
                            f"Erreur lors de la requête: "
                            f"{response.status_code}"
                        )
                        st.write(response.text)
                except Exception as e:
                    st.error(f"Erreur de connexion: {str(e)}")
        else:
            st.warning("Veuillez saisir une question.")
    
    st.divider()
    
    # Part 2: Graph Visualization
    st.subheader("📊 Visualisation des relations")
    st.write(
        "Explorez les relations entre acheteurs et titulaires "
        "dans les marchés publics."
    )
    
    entity_siren = st.text_input(
        "Numéro SIREN :",
        placeholder="Ex: 552015228 ou 130005481",
        help="Numéro SIREN à 9 chiffres (titulaire ou acheteur)"
    )
    
    min_amount = st.slider(
        "Montant minimum des contrats (€) :",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=1000,
        help="Filtrer les contrats en dessous de ce montant"
    )
    
    if st.button("Générer le graphique"):
        if entity_siren.strip():
            with st.spinner("Génération du graphique en cours..."):
                try:
                    # Initialize GraphPlotBuilder
                    builder = GraphPlotBuilder()
                    
                    # Create focused graph
                    graph_data = builder.create_focused_graph(
                        entity_siren=entity_siren,
                        min_contract_amount=min_amount
                    )
                    
                    if graph_data:
                        # Generate visualization
                        safe_siren = entity_siren.replace(' ', '_')
                        entity_type = graph_data.get('entity_type', 'unknown')
                        output_path = f"graph_{entity_type}_{safe_siren}.html"
                        builder.plot_focused_graph(
                            graph_data=graph_data,
                            output_path=output_path,
                            physics_enabled=True
                        )
                        
                        # Display results
                        st.success("Graphique généré avec succès !")
                        
                        # Show statistics
                        contract_data = graph_data['contract_data']
                        central_entity = graph_data['central_entity']
                        st.write(f"**Entité centrale :** {central_entity}")
                        st.write(f"**Nombre de contrats :** {len(contract_data)}")
                        total_amount = contract_data['montant'].sum()
                        avg_amount = contract_data['montant'].mean()
                        st.write(f"**Montant total :** {total_amount:,.2f}€")
                        st.write(f"**Montant moyen :** {avg_amount:,.2f}€")
                        
                        # Display the graph
                        if os.path.exists(output_path):
                            with open(output_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=800)
                            
                            # Cleanup
                            os.remove(output_path)
                        else:
                            st.error(
                                "Erreur lors de la génération du fichier "
                                "graphique"
                            )
                            
                    else:
                        st.warning(
                            f"Aucun contrat trouvé pour le SIREN: "
                            f"{entity_siren}"
                        )
                        
                except Exception as e:
                    st.error(
                        f"Erreur lors de la génération du graphique: "
                        f"{str(e)}"
                    )
                    st.write(
                        "Assurez-vous que les variables d'environnement "
                        "BigQuery sont configurées."
                    )
        else:
            st.warning("Veuillez saisir un SIREN d'entité.")

footer_html = """
<div class="footer">
    Le wagon batch #1992 -  Ronan Bernard, Paul Colas, Loïc Dogon, 
    Julie Hallez – 13 Juin 2025
</div>
"""

# Display both
st.markdown(footer_css, unsafe_allow_html=True)
st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    """Test the graph functionality with COLAS FRANCE."""
    print("Testing GraphPlotBuilder with COLAS FRANCE...")
    
    try:
        # Initialize GraphPlotBuilder
        builder = GraphPlotBuilder()
        
        # Test with sample SIREN
        entity_siren = '552015228'  # Example SIREN
        
        print(f"Creating focused graph for SIREN {entity_siren}...")
        
        # Create focused graph
        graph_data = builder.create_focused_graph(
            entity_siren=entity_siren,
            min_contract_amount=0
        )
        
        if graph_data:
            print("✓ Successfully created focused graph")
            
            # Generate visualization
            output_path = "test_colas_france_graph.html"
            builder.plot_focused_graph(
                graph_data=graph_data,
                output_path=output_path,
                physics_enabled=True
            )
            
            # Show statistics
            contract_data = graph_data['contract_data']
            print(f"Central entity: {graph_data['central_entity']}")
            print(f"Number of contracts: {len(contract_data)}")
            print(f"Total amount: {contract_data['montant'].sum():,.2f}€")
            print(f"Average amount: {contract_data['montant'].mean():,.2f}€")
            print(f"Connected entities: {len(graph_data['nodes']) - 1}")
            print(f"Graph saved to: {output_path}")
            
        else:
            print(f"No contracts found for SIREN {entity_siren}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure BigQuery environment variables are configured:")
        print("- GCP_PROJECT")
        print("- BQ_DATASET") 
        print("- BQ_TABLE")
