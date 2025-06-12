import streamlit as st
import numpy as np
import pandas as pd
import requests
from graph_plot_builder import GraphPlotBuilder
import os
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
    background-color: #445967;
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
    padding: 10px 0;    background-color: #59778A;
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

cpv_codes = [
    [3000000, "Matériel et fournitures informatiques"],
    [9000000, "Huiles lubrifiantes et agents lubrifiants"],
    [15000000, "Produits alimentaires divers"],
    [18000000, "Vêtements professionnels, vêtements de travail spéciaux et accessoires"],
    [22000000, "Livres de bibliothèque"],
    [30000000, "Matériel et fournitures informatiques"],
    [31000000, "Modules"],
    [32000000, "Système de surveillance vidéo"],
    [33000000, "Papier hygiénique, mouchoirs, essuie-mains et serviettes de table"],
    [34000000, "Véhicules à moteur"],
    [37000000, "Instruments de musique, articles de sport, jeux, jouets, articles pour artisanat, articles pour travaux artistiques et accessoires"],
    [39000000, "Caisses à compost"],
    [42000000, "Chariots de manutention"],
    [44000000, "Serrurerie"],
    [45000000, "Travaux de charpente"],
    [45100000, "Travaux de charpente"],
    [45200000, "Travaux de charpente"],
    [45300000, "Travaux de charpente"],
    [45400000, "Travaux de charpente"],
    [48000000, "Logiciels et systèmes d'information"],
    [50000000, "Services de réparation et d'entretien de chauffage central"],
    [55000000, "Services de colonies de vacances"],
    [60000000, "Services spécialisés de transport routier de passagers"],
    [64000000, "Services de courrier"],
    [66000000, "Services d'assurance de véhicules à moteur"],
    [71000000, "Services d'architecture"],
    [71200000, "Services d'architecture"],
    [71300000, "Services d'architecture"],
    [71400000, "Services d'architecture"],
    [71600000, "Services d'architecture"],
    [72000000, "Services de maintenance et de réparation de logiciels"],
    [74000000, "Services de conseil en développement"],
    [75000000, "Prestations de services pour la collectivité"],
    [77000000, "Réalisation et entretien d'espaces verts"],
    [79000000, "Services d'impression et de livraison"],
    [80000000, "Services de formation professionnelle"],
    [85000000, "Services de crèches et garderies d'enfants"],
    [90000000, "Service de gestion du réseau d'assainissement"],
    [92000000, "Services de musées"],
    [98000000, "Autres services"]
]
cpv = {f"{code} – {desc}": code for code, desc in cpv_codes}

bins = pd.read_csv('data/bins.csv', header=None).values.flatten()

# Sidebar: Module selector
module = st.sidebar.radio(
    "Choix du module :", 
    ["Exploration des données", "Estimation du montant et marchés similaires"]
)

# Main Panel
st.title("Lanterne publique")

if module == "Estimation du montant et marchés similaires":
    st.header("💰 Estimation du montant et marchés similaires")
    st.write("Utilisez ce module pour estimer le montant d'un futur marché public et trouver des marchés similaires.")
    st.write("Les champs obligatoires sont marqués d'un astérisque (*)")
    Code_CPV = st.selectbox("Choisissez une catégorie CPV :", list(cpv.keys()))
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
            "codeCPV_3": cpv[Code_CPV],
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
            # Récupération des probabilités (1 seule prédiction ici)
            probabilities = np.array(data["prediction"][0])
            # Génération des bins si pas fournis
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            montants = np.exp(bin_centers)  # retransforme en euros
            # Construction du DataFrame
            df = pd.DataFrame({
                'montant': montants,
                'probability': probabilities            })
            df['smoothed'] = df['probability'].rolling(window=10, center=True, min_periods=1).mean()
            # Trouver le montant le plus probable
            peak_montant = df.loc[df['smoothed'].idxmax(), 'montant']
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=df, x='montant', y='smoothed', color='blue', ax=ax)
            ax.set_title("Distribution prédite des montants")
            ax.set_xlabel("Montant estimé (€)")
            ax.set_ylabel("Probabilité")
            ax.grid(True)

            # Centrer le graphique autour du pic
            ax.set_xlim(peak_montant * 0.1, peak_montant * 2.5)
            st.pyplot(fig)

        else:
            st.error("Erreur lors de l'estimation du montant. Veuillez vérifier les paramètres et réessayer.")


    montant = st.number_input("Montant du marché (en euros)", min_value=0, value=40000, step=1000, format="%d")

    if st.button("Voir les marchés similaires"):
        params = {
            "montant": montant,
            "dureeMois": DureeMois,
            "offresRecues": OffresRecues,
            "procedure": procedure,
            "nature": nature,
            "formePrix": formePrix,
            "ccag": ccag,
            "codeCPV_3": cpv[Code_CPV],
            "sousTraitanceDeclaree": 0.0,
            "origineFrance": 0.0,
            "marcheInnovant": 0.0,
            "idAccordCadre": " ",
            "typeGroupementOperateurs": "Pas de groupement",
            "tauxAvance": 0.0,
        }

        response = requests.post(endpoint_clusters, json=params)
        if response.status_code == 200:
            data = response.json()
            st.write(data)
        else:
            st.error(f"Erreur lors de la récupération des marchés similaires. "
                    f"Code d'erreur: {response.status_code}")
            st.write("**Détails de l'erreur:**")
            st.write(response.text)
            st.write("**Paramètres envoyés:**")
            st.json(params)



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
                    payload = {"question": question}
                    response = requests.post(rag_endpoint, json=payload)
                    
                    if response.status_code == 200:
                        answer = response.json()
                        st.success("Réponse trouvée !")
                        st.write("**Réponse :**")
                        
                        # Extract the actual answer from the nested structure
                        if "answer" in answer and "answer" in answer["answer"]:
                            final_answer = answer["answer"]["answer"]
                            st.write(final_answer)
                        else:
                            st.write(answer)
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


