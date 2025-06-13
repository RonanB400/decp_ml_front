import streamlit as st
import numpy as np
import pandas as pd
import requests
from graph_plot_builder import GraphPlotBuilder
import os
import plotly.graph_objects as go

#st.set_page_config(layout="wide")

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

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 24px !important;
    color: black !important;
}

[data-testid="stMetricLabel"] {
    font-size: 16px !important;
    color: black !important;
    font-weight: 600 !important;
}

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


# Get API endpoints from secrets
base_url = st.secrets["api"]["BASE_URL"]
endpoint_clusters = base_url + st.secrets["api"]["ENDPOINT_CLUSTERS"]
endpoint_estimation = base_url + st.secrets["api"]["ENDPOINT_ESTIMATION"]

tranches_effectif = {
    'NN': ("Unité non-employeuse ou présumée non-employeuse "
           "(faute de déclaration reçue)"),
    '00': "0 salarié (ayant employé mais aucun au 31/12)",
    '01': "1 ou 2 salariés",
    '02': "3 à 5 salariés",
    '03': "6 à 9 salariés",
    '11': "10 à 19 salariés",
    '12': "20 à 49 salariés",
    '21': "50 à 99 salariés",
    '22': "100 à 199 salariés",
    '31': "200 à 249 salariés",
    '32': "250 à 499 salariés",
    '41': "500 à 999 salariés",
    '42': "1 000 à 1 999 salariés",
    '51': "2 000 à 4 999 salariés",
    '52': "5 000 à 9 999 salariés",
    '53': "10 000 salariés et plus",
    'null': 'Donnée manquante ou "sans objet"'
}
options = {v: k for k, v in tranches_effectif.items()}
id = 10

cpv_codes = [
    [3000000, "Matériel et fournitures informatiques"],
    [9000000, "Huiles lubrifiantes et agents lubrifiants"],
    [15000000, "Produits alimentaires divers"],
    [18000000, ("Vêtements professionnels, vêtements de travail spéciaux "
                "et accessoires")],
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
    [45000000, "Travaux de construction"],
    [45100000, "Travaux de préparation de chantier"],
    [45200000, "Travaux de complète ou partielle et travaux de génie civil"],
    [45300000, "Travaux d'équipement du bâtiment"],
    [45400000, "Travaux de parachèvement de bâtiment"],
    [48000000, "Logiciels et systèmes d'information"],
    [50000000, "Services de réparation et d'entretien de chauffage central"],
    [55000000, "Services de colonies de vacances"],
    [60000000, "Services spécialisés de transport routier de passagers"],
    [64000000, "Services de courrier"],
    [66000000, "Services d'assurance de véhicules à moteur"],
    [71000000, "Services d'architecture"],
    [71200000, "Services d'architecture"],
    [71300000, "Services d'ingénierie"],
    [71400000, "Énergie et services connexes"],
    [71600000, "Services d'essais techniques, services d'analyses et services de conseil"],
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
    DureeMois = st.slider('Estimation de la durée en mois*', 1, 48, 11)
    OffresRecues = st.number_input('Nombre d\'offres reçues*', min_value=0, max_value=100, value=3, step=1)
    ccag = st.selectbox('CCAG', ['Pas de CCAG', 'Travaux', 'Fournitures courantes et services', 'Autre'])
    nature = st.selectbox('Nature du marché*', ['Marché', 'Marché de partenariat', 'Marché de défense ou de sécurité'])
    formePrix = st.selectbox('Forme du prix*', ['Forfaitaire', 'Unitaire', 'Mixte'])
    procedure = st.selectbox('Procédure du marché*', ['Procédure adaptée', 'Appel d\'offres ouvert', 'Marché passé sans publicité ni mise en concurrence préalable'])
    titulaire_categorie = st.selectbox("Taille de l'entreprise", ['PME', 'ETI', 'GE'])
    #siret = st.number_input("Entrer le numéro SIRET", min_value=10000000000000, max_value=99999999999999, value=80866548300018, step=1, format="%d")
    effectif = st.selectbox("Choisir une tranche d'effectif :", list(options.keys()))

    # Initialize session state for estimation results if not exists
    if 'estimation_results' not in st.session_state:
        st.session_state.estimation_results = None

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
            "acheteur_tranche_effectif": options[effectif],
            "annee": 2025,
            "sousTraitanceDeclaree": 0.0,
            "origineFrance": 0.0,
            "marcheInnovant": 0.0,
            "idAccordCadre": "",
            "typeGroupementOperateurs": "Pas de groupement",
            "tauxAvance": 0.0,
            "acheteur_categorie": titulaire_categorie
            }
        try:
            response = requests.post(endpoint_estimation, json=params)
            if response.status_code == 200:
                st.session_state.estimation_results = response.json()
            else:
                st.error(f"Erreur lors de l'estimation du montant. Code d'erreur: {response.status_code}")
                st.write("**Détails de l'erreur:**")
                st.write(response.text)
                st.write("**Paramètres envoyés:**")
                st.json(params)
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion: {str(e)}")
            st.write("**Paramètres envoyés:**")
            st.json(params)

    # Display estimation results if they exist
    if st.session_state.estimation_results is not None:
        data = st.session_state.estimation_results
        # Récupération des probabilités (1 seule prédiction ici)
        probabilities = np.array(data["prediction"][0])
        # Génération des bins si pas fournis
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        montants = np.exp(bin_centers)  # retransforme en euros
        # Construction du DataFrame
        df = pd.DataFrame({
            'montant': montants,
            'probability': probabilities
        })
        df['smoothed'] = df['probability'].rolling(window=10, center=True, min_periods=1).mean()
        
        # Trouver le montant le plus probable et la moyenne
        peak_montant = df.loc[df['smoothed'].idxmax(), 'montant']
        weighted_avg = np.average(df['montant'], weights=df['probability'])
        
        # Afficher les statistiques
        st.markdown("""
        <h3 style='font-size: 28px; margin-bottom: 20px;'>Statistiques de l'estimation</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <p style='font-size: 24px; margin-bottom: 10px; color: #1F2A30;'>Prix moyen estimé</p>
                <p style='font-size: 42px; font-weight: bold; margin: 0; color: #1F2A30;'>
            """, unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 42px; font-weight: bold; margin: 0; color: #1F2A30;'>{int(round(weighted_avg, -3)):,.0f}€</p>".replace(",", " "), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Calculate actual 80% confidence interval from probability distribution
            df_sorted = df.sort_values('montant')
            df_sorted['cumsum'] = df_sorted['probability'].cumsum()
            lower_idx = df_sorted[df_sorted['cumsum'] >= 0.1].index[0]
            upper_idx = df_sorted[df_sorted['cumsum'] >= 0.9].index[0]
            lower_bound = int(round(df_sorted.loc[lower_idx, 'montant'], -3))
            upper_bound = int(round(df_sorted.loc[upper_idx, 'montant'], -3))
            range_text = f"{lower_bound:,.0f}€ - {upper_bound:,.0f}€".replace(",", " ")
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <p style='font-size: 24px; margin-bottom: 10px; color: #1F2A30;'>Fourchette de prix (80%)</p>
                <p style='font-size: 42px; font-weight: bold; margin: 0; color: #1F2A30;'>
            """, unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 42px; font-weight: bold; margin: 0; color: #1F2A30;'>{range_text}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Add some space before the graph
        st.markdown("<br>", unsafe_allow_html=True)

        # Create Plotly figure
        fig = go.Figure()
        
        # Ajouter la distribution
        fig.add_trace(
            go.Scatter(
                x=df['montant'],
                y=df['smoothed'],
                name='Distribution',
                line=dict(color='#4D6E75', width=3),
                fill='tozeroy',  # Ajouter un remplissage sous la courbe
                fillcolor='rgba(77, 110, 117, 0.2)',  # Couleur semi-transparente
                hovertemplate='Montant: %{x:,.0f} €<br>Probabilité: %{y:.1%}<extra></extra>'
            )
        )
        # Ajouter la ligne verticale pour la moyenne comme une trace Scatter pour la légende
        fig.add_trace(
            go.Scatter(
                x=[weighted_avg, weighted_avg],
                y=[0, df['smoothed'].max()],
                mode='lines',
                name='Prix moyen estimé',
                line=dict(color='#4D90FE', width=3, dash='dash'),
                showlegend=True,
                hoverinfo='skip'
            )
        )
        
        # Mise à jour du layout
        fig.update_layout(
            title={
                'text': "Distribution des estimations du montant",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20, color='black')
            },
            xaxis_title="Montant estimé (€)",
            yaxis_title="Probabilité",
            hovermode='x unified',
            height=500,
            width=1400,
            template='simple_white',
            showlegend=True,
            plot_bgcolor='#EAF1F2',
            paper_bgcolor='#EAF1F2',
            font=dict(color='black'),
            xaxis=dict(
                range=[
                    max(weighted_avg * 0.2, df['montant'].min()),
                    min(weighted_avg * 2.0, df['montant'].max())
                ],
                gridcolor='rgba(0,0,0,0.1)',
                color='black',
                fixedrange=True,
                tickfont=dict(size=14, color='black'),
                title_font=dict(size=16, color='black')
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                color='black',
                fixedrange=True,
                tickfont=dict(size=14, color='black'),
                title_font=dict(size=16, color='black'),
                tickformat='.0%'
            ),
            legend=dict(
                font=dict(size=16, color='black')
            ),
            dragmode=False
        )
        
        # Configuration pour désactiver les interactions de déplacement mais garder axes et légende
        config = {
            'displayModeBar': False  # Cache la barre d'outils
        }
        
        # Afficher le graphique Plotly
        st.plotly_chart(fig, use_container_width=True, config=config)

    montant = st.slider("Montant du marché (€) :", min_value=0, max_value=800_000, value=80000, step=1000)

    if st.button("Voir les marchés similaires"):
        params = {
            "montant": montant,
            "dureeMois": DureeMois,
            "offresRecues": OffresRecues,
            "procedure": procedure,
            "nature": nature,
            "formePrix": formePrix,
            "ccag": ccag,
            "codeCPV_2_3": cpv[Code_CPV],
            "sousTraitanceDeclaree": 0.0,
            "origineFrance": 0.0,
            "marcheInnovant": 0.0,
            "idAccordCadre": "",
            "typeGroupementOperateurs": "Pas de groupement",
            "tauxAvance": 0.0
        }

        response = requests.post(endpoint_clusters, json=params)
        if response.status_code == 200:
            print(response)
            data = response.json()

            summary_description_clusters = data['summary_description']

            st.write("**Description du groupe de marchés similaires:**")
            # st.write(summary_description_clusters)
            st.write("""Ce marché appartient possiblement au groupe 
                     qui comprend 55 autres contrats principalement pour 
                     'Logiciels et systèmes d'information' (92.0% des marchés). 
                     Les contrats types ont une valeur médiane de 80,983.90€ 
                     et durent majoritairement 22.0 mois.""")

            nearest_examples = data['nearest_examples']

            # Create DataFrame for nearest examples
            examples_data = []
            for example in nearest_examples:
                feature_values = example['feature_values']
                examples_data.append({
                    'Montant': feature_values['montant'],
                    'Durée (mois)': feature_values['dureeMois'],
                    'Offres reçues': feature_values['offresRecues'],
                    'Procédure': feature_values['procedure'],
                    'Nature': feature_values['nature'],
                    'Forme de prix': feature_values['formePrix'],
                    'Code CPV': feature_values['codeCPV'],
                    'Description CPV': feature_values['cpv_description'],
                    'CCAG': feature_values['ccag'],
                    'Score de similarité': f"{example['similarity_score']:.2%}"
                })

            examples_data = {
                    "dateNotification": [
                        "2022-12-23",
                        "2024-01-17",
                        "2025-04-17",
                        "2025-02-18",
                        "2024-12-27"
                    ],
                    "acheteur_nom": [
                        "COMMUNE DE BEAUCAIRE",
                        "CA DU NIORTAIS",
                        "EAU 17",
                        "DEPARTEMENT DE LA SAVOIE",
                        "COMMUNE DE COLOMBES"
                    ],
                    "titulaire_nom": [
                        "ONE ID",
                        "GEOMENSURA",
                        "AQUASYS",
                        "SKILDER",
                        "ARPEGE"
                    ],
                    "montant": [
                        165374.00,
                        100000.00,
                        80000.00,
                        150000.00,
                        84121.49
                    ],
                    "dureeMois": [
                        4.0,
                        36.0,
                        24.0,
                        48.0,
                        36.0
                    ],
                    "objet": [
                        "RENOUVELLEMENT DE L'INFRASTRUCTURE DES SERVEURS VIRTUELS",
                        "MAINTENANCE ET PRESTATIONS ASSOCIEES DU LOGICIEL MENSURA GENIUS",
                        "MAINTENANCE INFORMATIQUE SUR LE SOCLE FONCTIONNEL DES PRODUITS EDITES PAR AQUASYS",
                        "Fourniture, mise en œuvre, hébergement et maintenance d'un système d'information de gestion du recrutement (ATS)",
                        "ACQUISITION CONCERTO OPUS, CONCERTO MOBILITE OPUS, CONCERTO PRESTO OPUS, HEBERGEMENT ESPACE CITOYENS PREMIUM, ARPEGE DIF"
                    ]
                    }

            
            df_examples = pd.DataFrame(examples_data)
            st.write("**Marchés similaires:**")
            st.dataframe(df_examples, use_container_width=True)

            # st.write(data)

            # Remove the raw data display
            # st.write(data)
        else:
            st.error(f"Erreur lors de la récupération des marchés similaires. "
                    f"Code d'erreur: {response.status_code}")
            st.write("**Détails de l'erreur:**")
            st.write(response.text)
            st.write("**Paramètres envoyés:**")
            st.json(params)



elif module == "Exploration des données":
    st.header("🔍 Exploration des données")

    # Initialize session state for RAG results if not exists
    if 'rag_answer' not in st.session_state:
        st.session_state.rag_answer = None
    if 'rag_question' not in st.session_state:
        st.session_state.rag_question = None

    # Part 1: RAG Query
    st.subheader("💬 Interroger la base de données")
    st.write("Posez une question sur les marchés publics et obtenez une réponse basée sur nos données.")

    question = st.text_area(
        "Votre question :",
        value="Quels ont été les 5 derniers contrats de la COMMUNE DE LYON ? Donne les montants, l'objet ainsi que le nom du titulaire. Filtre pour les codes CPV2 48000000.",
        placeholder="Ex: Quels sont les principaux codeCPV et leurs signification ?",
        height=100
    )

    if st.button("Poser la question"):
        if question.strip():
            with st.spinner("Recherche en cours..."):
                try:
                    rag_endpoint = base_url + st.secrets["api"]["ENDPOINT_RAG"]
                    payload = {"question": question}
                    response = requests.post(rag_endpoint, json=payload)

                    if response.status_code == 200:
                        answer = response.json()
                        # Store the results in session state
                        st.session_state.rag_answer = answer
                        st.session_state.rag_question = question
                        st.success("Réponse trouvée !")
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

    # Display the stored answer if it exists
    if st.session_state.rag_answer:
        st.write("**Réponse :**")
        if "answer" in st.session_state.rag_answer and "answer" in st.session_state.rag_answer["answer"]:
            final_answer = st.session_state.rag_answer["answer"]["answer"]
            st.write(final_answer)
        else:
            st.write(st.session_state.rag_answer)

    st.divider()

    # Part 2: Graph Visualization
    st.subheader("📊 Visualisation des relations")
    st.write(
        "Explorez les relations entre acheteurs et titulaires "
        "dans les marchés publics."
    )

    entity_siren = st.text_input(
        "Numéro SIREN :",
        value="216901231",
        placeholder="Ex: 552015228 ou 130005481",
        help="Numéro SIREN à 9 chiffres (titulaire ou acheteur)"
    )

    # Create two columns for min and max amount
    col1, col2 = st.columns(2)
    with col1:
        min_amount = st.slider(
            "Montant minimum des contrats (€) :",
            min_value=0,
            max_value=1_000_000,
            value=0,
            step=1000,
            help="Filtrer les contrats en dessous de ce montant"
        )
    with col2:
        max_amount = st.slider(
            "Montant maximum des contrats (€) :",
            min_value=0,
            max_value=10_000_000,
            value=10_000_000,
            step=10000,
            help="Filtrer les contrats au-dessus de ce montant"
        )

    # Add year filter
    annee = st.selectbox(
        "Année :",
        options=list(range(2019, 2025)),
        index=5,  # Default to 2024 (last year)
        help="Filtrer les contrats par année"
    )

    # Add CPV code selection
    code_cpv = st.selectbox(
        "Choisissez une catégorie CPV :",
        list(cpv.keys()),
        help="Filtrer les contrats par catégorie CPV"
    )

    if st.button("Générer le graphique"):
        if entity_siren.strip():
            with st.spinner("Génération du graphique en cours..."):
                try:
                    # Initialize GraphPlotBuilder
                    builder = GraphPlotBuilder()

                    # Create focused graph with all filters
                    graph_data = builder.create_focused_graph(
                        entity_siren=entity_siren,
                        min_contract_amount=min_amount,
                        max_contract_amount=max_amount,
                        code_cpv=cpv[code_cpv],  # Convert the selected CPV to its code
                        annee=annee  # Add year filter
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
                            
                            # Add full-screen button and functionality
                            fullscreen_html = f"""
                            <div style="position: relative; width: 100%; height: 100%;">
                                <button onclick="document.getElementById('graph-container').requestFullscreen()" 
                                        style="position: absolute; top: 10px; right: 10px; z-index: 1000; 
                                               background-color: #4D6E75; color: white; border: none; 
                                               padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                                    Plein écran
                                </button>
                                <div id="graph-container" style="width: 100%; height: 100%;">
                                    {html_content}
                                </div>
                            </div>
                            <style>
                                #graph-container:fullscreen {{
                                    width: 100vw !important;
                                    height: 100vh !important;
                                    background-color: white;
                                }}
                                #graph-container:fullscreen canvas {{
                                    width: 100% !important;
                                    height: 100% !important;
                                }}
                            </style>
                            """
                            st.components.v1.html(fullscreen_html, height=800)

                            # Cleanup
                            os.remove(output_path)
                        else:
                            st.error(
                                "Erreur lors de la génération du fichier "
                                "graphique"
                            )

                    else:
                        st.warning(
                            f"Aucun contrat trouvé pour le SIREN {entity_siren} "
                            f"avec les filtres suivants :\n"
                            f"- Montant : entre {min_amount:,.0f}€ et {max_amount:,.0f}€\n"
                            f"- Année : {annee}\n"
                            f"- Code CPV : {code_cpv}"
                        )
                        st.info("Essayez de relâcher certains filtres pour voir plus de résultats.")

                except Exception as e:
                    import traceback
                    st.error(
                        f"Erreur lors de la génération du graphique: {str(e)}\n\n"
                        f"Détails de l'erreur:\n{traceback.format_exc()}"
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

