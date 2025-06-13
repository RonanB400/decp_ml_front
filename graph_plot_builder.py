import pandas as pd
import os
import logging
from pyvis.network import Network
import networkx as nx
import json

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    import streamlit as st
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    print("Warning: google-cloud-bigquery or streamlit not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphPlotBuilder:
    """Build graph structures from procurement data."""
    
    def __init__(self, gcp_project: str = None, bq_dataset: str = None,
                 bq_table: str = None):
        """Initialize with BigQuery configuration.
        
        Args:
            gcp_project: GCP project ID
            bq_dataset: BigQuery dataset name
            bq_table: BigQuery table name
        """
        # Use Streamlit secrets configuration
        try:
            self.gcp_project = gcp_project or st.secrets["gcp"]["GCP_PROJECT"]
            self.bq_dataset = bq_dataset or st.secrets["gcp"]["BQ_DATASET"]
            self.bq_table = bq_table or st.secrets["gcp"]["BQ_TABLE"]
            
            # Get service account info from secrets
            if "gcp_service_account" in st.secrets:
                credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                self.client = bigquery.Client(
                    project=self.gcp_project,
                    credentials=credentials
                )
            else:
                logger.warning("No service account credentials found in secrets")
                self.client = None
                
        except (KeyError, AttributeError) as e:
            raise ValueError(
                f"Missing required secrets configuration: {e}. "
                "Please ensure secrets.toml is properly configured."
            )
    
    def load_data_from_bigquery(self, entity_siren: str
                                ) -> tuple[pd.DataFrame, str, pd.DataFrame]:
        """Load procurement data from BigQuery for a specific entity.
        
        Args:
            entity_siren: SIREN number of the entity to focus on
        
        Returns:
            Tuple of (DataFrame with direct contracts, entity_type, DataFrame with secondary contracts)
        """
        if not self.client:
            raise ValueError("BigQuery client not available. Check your "
                             "configuration.")
        
        # Query for both titulaire and acheteur to determine entity type
        query_titulaire = f"""
            SELECT *, 'titulaire' as entity_type
            FROM {self.gcp_project}.{self.bq_dataset}.{self.bq_table}
            WHERE CAST(titulaire_siren AS STRING) = '{entity_siren}'
        """
        
        query_acheteur = f"""
            SELECT *, 'acheteur' as entity_type
            FROM {self.gcp_project}.{self.bq_dataset}.{self.bq_table}
            WHERE CAST(acheteur_siren AS STRING) = '{entity_siren}'
        """
        
        logger.info(f"Querying BigQuery for SIREN: {entity_siren}")
        
        try:
            # Try titulaire first
            query_job_titulaire = self.client.query(query_titulaire)
            df_titulaire = query_job_titulaire.result().to_dataframe()
            
            # Try acheteur
            query_job_acheteur = self.client.query(query_acheteur)
            df_acheteur = query_job_acheteur.result().to_dataframe()
            
            # Determine which has data and return accordingly
            if not df_titulaire.empty and not df_acheteur.empty:
                # SIREN appears in both - prefer titulaire
                logger.info(f"SIREN {entity_siren} found as both titulaire and acheteur, using titulaire")
                df = df_titulaire
                entity_type = 'titulaire'
            elif not df_titulaire.empty:
                logger.info(f"SIREN {entity_siren} found as titulaire")
                df = df_titulaire
                entity_type = 'titulaire'
            elif not df_acheteur.empty:
                logger.info(f"SIREN {entity_siren} found as acheteur")
                df = df_acheteur
                entity_type = 'acheteur'
            else:
                logger.warning(f"No contracts found for SIREN {entity_siren}")
                return pd.DataFrame(), None, pd.DataFrame()
            
            logger.info(f"Retrieved {len(df)} contracts for {entity_siren} as {entity_type}")

            selected_columns = ['dateNotification', 'acheteur_nom', 'acheteur_siren',
                                'titulaire_nom', 'titulaire_siren', 'montant', 
                                'dureeMois', 'codeCPV', 'procedure', 'objet', 'codeCPV_2_3', 'annee']
        
            # Ensure all required columns exist
            missing_columns = [col for col in selected_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns in BigQuery data: {missing_columns}")
                raise ValueError(f"Missing columns in BigQuery data: {missing_columns}")
            
            df = df[selected_columns]
            
            # Convert montant to numeric, replacing any non-numeric values with NaN
            df['montant'] = pd.to_numeric(df['montant'], errors='coerce')
            df = df.dropna(subset=['montant'])
            
            if df.empty:
                logger.warning(f"No valid contracts with montant values found for SIREN {entity_siren}")
                return pd.DataFrame(), None, pd.DataFrame()
            
            # Get secondary contracts for connected entities
            if entity_type == 'titulaire':
                # Get all contracts for buyers connected to this supplier
                connected_sirens = df['acheteur_siren'].unique()
                query_secondary = f"""
                    SELECT {', '.join(selected_columns)}
                    FROM {self.gcp_project}.{self.bq_dataset}.{self.bq_table}
                    WHERE CAST(acheteur_siren AS STRING) IN (
                        {', '.join([f"'{siren}'" for siren in connected_sirens])}
                    )
                    AND CAST(titulaire_siren AS STRING) != '{entity_siren}'
                """
            else:
                # Get all contracts for suppliers connected to this buyer
                connected_sirens = df['titulaire_siren'].unique()
                query_secondary = f"""
                    SELECT {', '.join(selected_columns)}
                    FROM {self.gcp_project}.{self.bq_dataset}.{self.bq_table}
                    WHERE CAST(titulaire_siren AS STRING) IN (
                        {', '.join([f"'{siren}'" for siren in connected_sirens])}
                    )
                """  # Removed the filter that excluded contracts with our central buyer
            
            # Execute secondary query if we have connected entities
            if len(connected_sirens) > 0:
                logger.info(f"Querying secondary contracts for {len(connected_sirens)} connected entities")
                query_job_secondary = self.client.query(query_secondary)
                df_secondary = query_job_secondary.result().to_dataframe()
                
                # Convert montant to numeric in secondary data
                df_secondary['montant'] = pd.to_numeric(df_secondary['montant'], errors='coerce')
                df_secondary = df_secondary.dropna(subset=['montant'])
                
                logger.info(f"Retrieved {len(df_secondary)} secondary contracts")
            else:
                df_secondary = pd.DataFrame(columns=selected_columns)

            return df, entity_type, df_secondary
            
        except Exception as e:
            logger.error(f"Error querying BigQuery: {e}")
            raise

    def create_focused_graph(self, entity_siren: str,
                            min_contract_amount: float = 0,
                            max_contract_amount: float = None,
                            code_cpv: int = None,
                            annee: int = None) -> dict:
        """Create a graph focused on a specific entity using BigQuery data.
        
        Args:
            entity_siren: SIREN number of the entity to focus on
            min_contract_amount: Minimum contract amount to include
            max_contract_amount: Maximum contract amount to include
            code_cpv: CPV code to filter by
            annee: Year to filter by
        
        Returns:
            Graph data dictionary optimized for focused visualization
        """
        try:
            # Load data from BigQuery and determine entity type
            X_filtered, entity_type, X_secondary = self.load_data_from_bigquery(entity_siren)
            
            if X_filtered.empty or entity_type is None:
                logger.warning(f"No contracts found for SIREN: {entity_siren}")
                return None
            
            # Log the columns we have
            logger.info(f"Columns in X_filtered: {X_filtered.columns.tolist()}")
            if not X_secondary.empty:
                logger.info(f"Columns in X_secondary: {X_secondary.columns.tolist()}")
            
            # Ensure required columns exist
            required_columns = ['acheteur_nom', 'titulaire_nom', 'montant',
                              'dureeMois', 'codeCPV', 'procedure', 'annee']
            missing_columns = [col for col in required_columns
                             if col not in X_filtered.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert montant to float and handle any invalid values
            try:
                logger.info(f"Converting montant column. Current dtype: {X_filtered['montant'].dtype}")
                logger.info(f"Sample of montant values before conversion: {X_filtered['montant'].head()}")
                
                X_filtered['montant'] = pd.to_numeric(X_filtered['montant'], errors='coerce')
                logger.info(f"Montant conversion successful. New dtype: {X_filtered['montant'].dtype}")
                logger.info(f"Sample of montant values after conversion: {X_filtered['montant'].head()}")
                
                if X_secondary is not None and not X_secondary.empty:
                    logger.info(f"Converting secondary montant column. Current dtype: {X_secondary['montant'].dtype}")
                    X_secondary['montant'] = pd.to_numeric(X_secondary['montant'], errors='coerce')
                    logger.info(f"Secondary montant conversion successful")
            except Exception as e:
                logger.error(f"Error converting montant to numeric: {e}")
                logger.error(f"X_filtered columns: {X_filtered.columns}")
                if 'montant' in X_filtered.columns:
                    logger.error(f"Sample montant values causing error: {X_filtered['montant'].head()}")
                raise ValueError(f"Could not convert montant column to numeric values: {str(e)}")
            
            # Remove rows with NaN montant
            original_len = len(X_filtered)
            X_filtered = X_filtered.dropna(subset=['montant'])
            logger.info(f"Removed {original_len - len(X_filtered)} rows with NaN montant values")
            
            if X_secondary is not None and not X_secondary.empty:
                original_len = len(X_secondary)
                X_secondary = X_secondary.dropna(subset=['montant'])
                logger.info(f"Removed {original_len - len(X_secondary)} rows with NaN montant values from secondary data")
            
            # Remove rows with NaN buyer or supplier names
            valid_mask = (X_filtered['acheteur_nom'].notna() & 
                          X_filtered['titulaire_nom'].notna())
            X_filtered = X_filtered[valid_mask].copy()
            
            # Apply amount filters to primary contracts
            amount_mask = X_filtered['montant'] >= min_contract_amount
            if max_contract_amount is not None:
                amount_mask = amount_mask & (X_filtered['montant'] <= max_contract_amount)
            X_filtered = X_filtered[amount_mask].copy()
            
            # Apply CPV filter if specified
            if code_cpv is not None:
                cpv_mask = X_filtered['codeCPV_2_3'] == code_cpv
                X_filtered = X_filtered[cpv_mask].copy()
                logger.info(f"Filtered contracts with CPV code: {code_cpv}")
            
            # Apply year filter if specified
            if annee is not None:
                year_mask = X_filtered['annee'] == annee
                X_filtered = X_filtered[year_mask].copy()
                logger.info(f"Filtered contracts for year: {annee}")
            
            # Check if we have any data after filtering
            if X_filtered.empty:
                logger.warning("No contracts found after applying filters")
                return None
            
            logger.info(f"Processing {len(X_filtered)} valid primary contracts")
            
            # Apply same filters to secondary contracts
            if not X_secondary.empty:
                # Apply amount filter
                amount_mask = X_secondary['montant'] >= min_contract_amount
                if max_contract_amount is not None:
                    amount_mask = amount_mask & (X_secondary['montant'] <= max_contract_amount)
                X_secondary = X_secondary[amount_mask].copy()
                
                # Apply CPV filter
                if code_cpv is not None:
                    cpv_mask = X_secondary['codeCPV_2_3'] == code_cpv
                    X_secondary = X_secondary[cpv_mask].copy()
                
                # Apply year filter
                if annee is not None:
                    year_mask = X_secondary['annee'] == annee
                    X_secondary = X_secondary[year_mask].copy()
                
                logger.info(f"Processing {len(X_secondary)} valid secondary contracts")
            
            # Create focused graph structure
            if entity_type == 'titulaire':
                # Central node is the supplier, connected nodes are buyers
                central_entity = X_filtered['titulaire_nom'].iloc[0] if not X_filtered.empty else f"SIREN {entity_siren}"
                connected_entities = X_filtered['acheteur_nom'].unique().tolist()
                central_type = 1  # Supplier
                connected_type = 0  # Buyers

                # If we have secondary data, add additional buyers
                if not X_secondary.empty:
                    additional_buyers = X_secondary['acheteur_nom'].unique().tolist()
                    # Only add buyers that aren't already connected to our supplier
                    new_buyers = [buyer for buyer in additional_buyers if buyer not in connected_entities]
                    connected_entities.extend(new_buyers)
            else:
                # Central node is the buyer, connected nodes are suppliers
                central_entity = X_filtered['acheteur_nom'].iloc[0] if not X_filtered.empty else f"SIREN {entity_siren}"
                connected_entities = X_filtered['titulaire_nom'].unique().tolist()
                central_type = 0  # Buyer
                connected_type = 1  # Suppliers

                # Get all buyers connected to our suppliers from secondary data
                if not X_secondary.empty:
                    # Get unique buyers from secondary data
                    secondary_buyers = X_secondary['acheteur_nom'].unique().tolist()
                    # Remove our central buyer if present
                    if central_entity in secondary_buyers:
                        secondary_buyers.remove(central_entity)
                    # Add these buyers to our node list
                    connected_entities.extend(secondary_buyers)
                    logger.info(f"Added {len(secondary_buyers)} secondary buyers")
            
            # Create node mappings - central node gets ID 0
            all_nodes = [central_entity] + connected_entities
            node_to_id = {node: i for i, node in enumerate(all_nodes)}
            
            # Create edges - one edge per contract
            edges = []
            edge_features = []
            contract_data_list = []
            
            # Add primary edges (connections to central node)
            for _, row in X_filtered.iterrows():
                if entity_type == 'titulaire':
                    # Edge from supplier (central) to buyer
                    buyer_id = node_to_id[row['acheteur_nom']]
                    edges.append((0, buyer_id))  # Central node is always 0
                else:
                    # Edge from buyer (central) to supplier
                    supplier_id = node_to_id[row['titulaire_nom']]
                    edges.append((0, supplier_id))  # Central node is always 0
                
                # Store edge features (contract details)
                edge_features.append([
                    float(row['montant']) if pd.notna(row['montant']) else 0,
                    float(row['dureeMois']) if pd.notna(row['dureeMois']) else 0,
                    row.get('codeCPV', ''),
                    row.get('procedure', ''),
                    row.get('dateNotification', ''),
                    row.get('objet', '')  # Add contract object
                ])
                
                # Store contract data for analysis
                contract_data_list.append({
                    'acheteur_nom': row['acheteur_nom'],
                    'titulaire_nom': row['titulaire_nom'],
                    'montant': row['montant'],
                    'codeCPV': row.get('codeCPV', ''),
                    'procedure': row.get('procedure', ''),
                    'dureeMois': row['dureeMois'],
                    'dateNotification': row.get('dateNotification', ''),
                    'objet': row.get('objet', ''),  # Add contract object
                    'is_secondary': False  # Flag for primary contracts
                })
            
            # Add secondary edges
            if not X_secondary.empty:
                for _, row in X_secondary.iterrows():
                    if entity_type == 'titulaire':
                        # For supplier view: connect central supplier to additional buyers
                        buyer_id = node_to_id[row['acheteur_nom']]
                        edges.append((0, buyer_id))  # Connect to central supplier node
                    else:
                        # For buyer view: connect suppliers to their other buyers
                        supplier_name = row['titulaire_nom']
                        buyer_name = row['acheteur_nom']
                        
                        # Only add edge if the supplier is connected to our central buyer
                        # and the buyer exists in our node list
                        if (supplier_name in node_to_id and 
                            buyer_name in node_to_id and 
                            supplier_name in connected_entities):
                            supplier_id = node_to_id[supplier_name]
                            buyer_id = node_to_id[buyer_name]
                            edges.append((supplier_id, buyer_id))
                            logger.info(f"Added secondary edge: {supplier_name} -> {buyer_name}")
                    
                    # Store edge features for secondary contract
                    edge_features.append([
                        float(row['montant']) if pd.notna(row['montant']) else 0,
                        float(row['dureeMois']) if pd.notna(row['dureeMois']) else 0,
                        row.get('codeCPV', ''),
                        row.get('procedure', ''),
                        row.get('dateNotification', ''),
                        row.get('objet', '')  # Add contract object
                    ])
                    
                    # Store contract data
                    contract_data_list.append({
                        'acheteur_nom': row['acheteur_nom'],
                        'titulaire_nom': row['titulaire_nom'],
                        'montant': row['montant'],
                        'codeCPV': row.get('codeCPV', ''),
                        'procedure': row.get('procedure', ''),
                        'dureeMois': row['dureeMois'],
                        'dateNotification': row.get('dateNotification', ''),
                        'objet': row.get('objet', ''),
                        'is_secondary': True  # Flag for secondary contracts
                    })
            
            # Create contract data DataFrame
            contract_data = pd.DataFrame(contract_data_list)
            
            # Central node features with proper NaN handling
            central_contracts = contract_data[~contract_data['is_secondary']]  # Only primary contracts
            central_total_amount = central_contracts['montant'].sum()
            central_total_amount = float(central_total_amount) if pd.notna(central_total_amount) else 0.0
            
            central_mean_amount = central_contracts['montant'].mean()
            central_mean_amount = float(central_mean_amount) if pd.notna(central_mean_amount) else 0.0
            
            central_mean_duration = central_contracts['dureeMois'].mean()
            central_mean_duration = float(central_mean_duration) if pd.notna(central_mean_duration) else 0.0
            
            central_features = [
                len(central_contracts),
                central_total_amount,
                central_mean_amount,
                central_mean_duration
            ]
            
            # Connected nodes features
            node_features = [central_features]
            node_types = [central_type]
            
            for connected_entity in connected_entities:
                # Get both primary and secondary contracts for this entity
                entity_mask = (
                    (contract_data['acheteur_nom'] == connected_entity) |
                    (contract_data['titulaire_nom'] == connected_entity)
                )
                entity_contracts = contract_data[entity_mask]
                
                # Calculate features with proper NaN handling
                total_amount = entity_contracts['montant'].sum()
                total_amount = float(total_amount) if pd.notna(total_amount) else 0.0
                
                mean_amount = entity_contracts['montant'].mean()
                mean_amount = float(mean_amount) if pd.notna(mean_amount) else 0.0
                
                mean_duration = entity_contracts['dureeMois'].mean()
                mean_duration = float(mean_duration) if pd.notna(mean_duration) else 0.0
                
                features = [
                    len(entity_contracts),
                    total_amount,
                    mean_amount,
                    mean_duration
                ]
                node_features.append(features)
                node_types.append(connected_type)
            
            graph_data = {
                'nodes': all_nodes,
                'edges': edges,
                'node_features': node_features,
                'edge_features': edge_features,
                'node_types': node_types,
                'central_entity': central_entity,
                'entity_type': entity_type,
                'contract_data': contract_data,
                'focus_node_id': 0  # Central node is always at index 0
            }
            
            logger.info(f"Created focused graph with {len(all_nodes)} nodes and {len(edges)} edges")
            return graph_data
            
        except Exception as e:
            logger.error(f"Error in create_focused_graph: {str(e)}")
            raise

    def plot_focused_graph(self, graph_data: dict,
                          output_path: str = "focused_graph_visualization.html",
                          physics_enabled: bool = True) -> None:
        """Visualize a focused graph with the central entity prominently displayed.
        
        Args:
            graph_data: Graph data from create_focused_graph
            output_path: Path to save the HTML visualization
            physics_enabled: Whether to enable physics simulation
        """
        if not graph_data:
            logger.error("No graph data provided")
            return
        
        logger.info("Creating focused graph visualization...")
        
        # Get graph components
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        node_features = graph_data['node_features']
        node_types = graph_data['node_types']
        edge_features = graph_data['edge_features']
        central_entity = graph_data['central_entity']
        entity_type = graph_data['entity_type']
        contract_data = graph_data['contract_data']
        
        # Create network with larger size
        net = Network(height="800px", width="100%", directed=False)
        
        # Add nodes with special handling for the central node
        for i, (node_name, features, node_type) in enumerate(
                zip(nodes, node_features, node_types)):
            
            # Get all contracts for this entity
            if i == 0:  # Central node
                node_contracts = contract_data[~contract_data['is_secondary']]
                secondary_contracts = contract_data[contract_data['is_secondary']]
            else:
                # Get primary contracts
                primary_mask = (
                    (~contract_data['is_secondary']) &
                    ((contract_data['acheteur_nom'] == node_name) |
                     (contract_data['titulaire_nom'] == node_name))
                )
                node_contracts = contract_data[primary_mask]
                
                # Get secondary contracts
                secondary_mask = (
                    (contract_data['is_secondary']) &
                    ((contract_data['acheteur_nom'] == node_name) |
                     (contract_data['titulaire_nom'] == node_name))
                )
                secondary_contracts = contract_data[secondary_mask]
            
            # Calculate contract amounts
            primary_amount = node_contracts['montant'].sum()
            primary_amount = primary_amount if not pd.isna(primary_amount) else 0.0
            
            secondary_amount = secondary_contracts['montant'].sum()
            secondary_amount = secondary_amount if not pd.isna(secondary_amount) else 0.0
            
            total_amount = primary_amount + secondary_amount
            
            # Set node properties based on type and position
            if i == 0:  # Central node
                color = "#ffaa00"  # Orange for central node
                size = min(40 + (total_amount / 50000), 100)  # More dramatic scaling
                shape = "circle"  # Always use circle for central node
                type_label = f"Central {entity_type.title()}"
                logger.info(f"Central node {node_name}: €{total_amount:,.2f} -> size {size}")
            elif node_type == 0:  # Buyer
                color = "#ff9999"  # Light red
                shape = "box"
                size = min(10 + (total_amount / 50000), 50)  # Scale by 50k euros
                type_label = "Acheteur"
                logger.info(f"Buyer {node_name}: €{total_amount:,.2f} -> size {size}")
            else:  # Supplier
                color = "#99ccff"  # Light blue
                shape = "circle"
                size = min(10 + (total_amount / 50000), 50)  # Scale by 50k euros
                type_label = "Titulaire"
                logger.info(f"Supplier {node_name}: €{total_amount:,.2f} -> size {size}")
            
            # Position nodes - central at center, others in a circle
            if i == 0:
                x, y = 0, 0  # Center
            else:
                # Arrange other nodes in a circle around the center
                angle = 2 * 3.14159 * (i - 1) / (len(nodes) - 1)
                radius = 800
                x = radius * nx.Graph().nodes.__class__.__dict__.get('cos', lambda a: 1)(angle)
                y = radius * nx.Graph().nodes.__class__.__dict__.get('sin', lambda a: 1)(angle)
            
            # Truncate label for display (max 20 characters)
            display_label = str(node_name)
            if len(display_label) > 20:
                display_label = display_label[:17] + "..."
            
            # Create tooltip with node information
            title = [f"{type_label}: {node_name}"]  # Full name in tooltip
            
            # Add primary contract information
            if not node_contracts.empty:
                title.extend([
                    f"Contrats primaires: {len(node_contracts)}",
                    f"Montant total primaire: {primary_amount:,.2f}€",
                    f"Montant moyen primaire: {primary_amount/len(node_contracts):,.2f}€"
                ])
            
            # Add secondary contract information
            if not secondary_contracts.empty:
                title.extend([
                    "",  # Empty line for spacing
                    f"Contrats secondaires: {len(secondary_contracts)}",
                    f"Montant total secondaire: {secondary_amount:,.2f}€",
                    f"Montant moyen secondaire: {secondary_amount/len(secondary_contracts):,.2f}€"
                ])
            
            title = '\n'.join(title)
            
            net.add_node(
                i,
                label=display_label,  # Use truncated label for display
                color=color,
                size=size,
                shape=shape,
                title=title,  # Full information in tooltip
                x=x,
                y=y,
                fixed=(i == 0)  # Fix central node position
            )
        
        # Add edges with variable width based on contract amount and tooltip
        for i, edge in enumerate(edges):
            if i < len(edge_features):
                amount = edge_features[i][0] if len(edge_features[i]) > 0 else 1
                width = min(2 + amount / 50000, 8)  # Scale width
                
                # Create tooltip with contract details
                tooltip = (
                    f"Contrat\n"
                    f"Montant: {amount:,.2f}€\n"
                    f"Durée: {edge_features[i][1]:.1f} mois\n"
                    f"Code CPV: {edge_features[i][2]}\n"
                    f"Procédure: {edge_features[i][3]}\n"
                    f"Date: {edge_features[i][4]}\n"
                    f"Objet: {edge_features[i][5]}"
                )
                
                # Determine if this is a second-level edge
                is_secondary = edge[0] != 0 and edge[1] != 0
                
                # Add edge with hover effect
                net.add_edge(
                    edge[0], edge[1],
                    width=width,
                    title=tooltip,
                    hoverWidth=width * 1.5,  # Make edge wider on hover
                    color='#666666' if not is_secondary else '#999999',  # Lighter color for secondary edges
                    style='solid' if not is_secondary else 'dashed',  # Dashed line for secondary edges
                    hoverColor='#ff0000'  # Red on hover
                )
            else:
                net.add_edge(edge[0], edge[1], width=2)
        
        # Configure physics for radial layout
        if physics_enabled:
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 40},
                "barnesHut": {
                  "gravitationalConstant": -3000,
                  "centralGravity": 0.3,
                  "springLength": 500,
                  "springConstant": 0.02,
                  "damping": 0.2,
                  "avoidOverlap": 0.1
                }
              },
              "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
              }
            }
            """)
        else:
            net.set_options('{"physics": {"enabled": false}}')
        
        # Save the visualization
        net.save_graph(output_path)
        logger.info(f"Focused graph visualization saved to {output_path}")
        logger.info(f"Central entity: {central_entity} ({entity_type})")
        logger.info(f"Connected entities: {len(nodes) - 1}")
        
        # Log contract statistics
        primary_contracts = contract_data[~contract_data['is_secondary']]
        secondary_contracts = contract_data[contract_data['is_secondary']]
        logger.info(f"Total primary contracts: {len(primary_contracts)}")
        logger.info(f"Total secondary contracts: {len(secondary_contracts)}")
        logger.info(f"Total contracts: {len(contract_data)}")

    

if __name__ == "__main__":
    """Test the GraphPlotBuilder functionality."""
    import sys
    
    def test_bigquery_integration():
        """Test BigQuery integration with sample data."""
        print("Testing GraphPlotBuilder with BigQuery integration...")
        
        # Initialize builder with environment variables or defaults
        builder = GraphPlotBuilder()
        
        if not builder.client:
            print("Warning: BigQuery client not available. Skipping tests.")
            print("To test BigQuery functionality, configure .streamlit/secrets.toml")
            print("with GCP_PROJECT, BQ_DATASET, and BQ_TABLE")
            return False
        
        try:
            # Test with a sample SIREN number
            # sample_siren = "216901231"  # COMMUNE DE LYON
            sample_siren = "217401058" # COMMUNE DE DOUVAINE
            print(f"Testing focused graph creation for SIREN: {sample_siren}")
            
            # Test focused graph creation
            graph_data = builder.create_focused_graph(
                entity_siren=sample_siren,
                min_contract_amount=40_000
            )
            
            if graph_data:
                print("✓ Successfully created focused graph")
                output_file = "test_titulaire_graph.html"
                builder.plot_focused_graph(
                    graph_data=graph_data,
                    output_path=output_file
                )
                print("✓ Successfully generated visualization")
                
                # Open the HTML file in the default browser
                import webbrowser
                import subprocess
                abs_path = os.path.abspath(output_file)
                if os.path.exists(abs_path):
                    print(f"✓ File exists at: {abs_path}")
                    print(f"✓ Opening {output_file} in browser...")
                    
                    # Try different methods for WSL2 compatibility
                    try:
                        # Method 1: Try WSL integration with Windows browser
                        if 'microsoft' in os.uname().release.lower():
                            # Convert WSL path to Windows path
                            result = subprocess.run(['wslpath', '-w', abs_path], 
                                                  capture_output=True, text=True)
                            if result.returncode == 0:
                                windows_path = result.stdout.strip()
                                print(f"Windows path: {windows_path}")
                                webbrowser.open(f"file:///{windows_path}")
                            else:
                                raise Exception("WSL path conversion failed")
                        else:
                            # Standard Linux/Mac approach
                            webbrowser.open(f"file://{abs_path}")
                        
                        print("✓ Browser opened successfully")
                    except Exception as e:
                        print(f"Error opening browser: {e}")
                        print(f"Alternative: copy this path to your browser:")
                        print(f"file://{abs_path}")
                else:
                    print(f"✗ File not found at: {abs_path}")
                
                return True
            else:
                print(f"No contracts found for SIREN {sample_siren}")
                return False
                
        except Exception as e:
            print(f"Error during BigQuery test: {e}")
            return False
    
    def test_csv_functionality():
        """Test original CSV-based functionality."""
        print("\nTesting original CSV functionality...")
        
        # Create sample CSV data for testing
        import tempfile
        sample_data = """dateNotification,acheteur_nom,titulaire_nom,montant,dureeMois,codeCPV,procedure,objet
2023-01-01,Buyer A,Supplier X,50000,12,12345,Open,Test Contract 1
2023-01-02,Buyer B,Supplier X,75000,18,12346,Restricted,Test Contract 2
2023-01-03,Buyer A,Supplier Y,30000,6,12347,Open,Test Contract 3
2023-01-04,Buyer C,Supplier Z,100000,24,12348,Negotiated,Test Contract 4"""
        
        try:
            # Create temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', 
                                             delete=False) as f:
                f.write(sample_data)
                temp_csv = f.name
            
            # Create temporary directory structure
            import os
            temp_dir = os.path.dirname(temp_csv)
            data_clean_path = os.path.join(temp_dir, 'data_clean.csv')
            
            # Copy sample data to expected filename
            with open(data_clean_path, 'w') as f:
                f.write(sample_data)
            
            builder = GraphPlotBuilder()
            
            # Test loading data
            data = builder.load_data(temp_dir)
            print(f"✓ Successfully loaded {len(data)} rows from CSV")
            
            # Test creating general graph
            graph_data = builder.create_graph(data, min_contract_amount=0)
            print(f"✓ Successfully created graph with {len(graph_data['nodes'])} nodes")
            
            # Test visualization
            builder.plot_graph(
                graph_data=graph_data,
                output_path="test_general_graph.html",
                max_nodes=50,
                max_edges=100
            )
            print("✓ Successfully generated general graph visualization")
            
            # Cleanup
            os.unlink(temp_csv)
            os.unlink(data_clean_path)
            
            return True
            
        except Exception as e:
            print(f"Error during CSV test: {e}")
            return False
    
    def main():
        """Main test function."""
        print("=" * 60)
        print("GraphPlotBuilder Test Suite")
        print("=" * 60)
        
        # Test BigQuery functionality
        bigquery_success = test_bigquery_integration()
        
        # Test CSV functionality
        # csv_success = test_csv_functionality()
        
        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"BigQuery Integration: {'✓ PASS' if bigquery_success else '✗ SKIP/FAIL'}")
        # print(f"CSV Functionality: {'✓ PASS' if csv_success else '✗ FAIL'}")
        

        if bigquery_success:
            print("- test_titulaire_graph.html")
        else:
            print("\n✗ Some tests failed. Check error messages above.")
            sys.exit(1)
        
        print("\nTo use with your own data:")
        print("1. Configure .streamlit/secrets.toml with GCP credentials")
        print("2. Call create_focused_graph() with your entity names")
        print("3. Use plot_focused_graph() to generate visualizations")
    
    main()