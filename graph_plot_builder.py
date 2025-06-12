import pandas as pd
import os
import pickle
import logging
from typing import Optional, Union
from pyvis.network import Network
import networkx as nx
import random

try:
    from google.cloud import bigquery
    from dotenv import load_dotenv
    BIGQUERY_AVAILABLE = True
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    BIGQUERY_AVAILABLE = False
    print("Warning: google-cloud-bigquery or python-dotenv not available")

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
        self.gcp_project = gcp_project or os.getenv('GCP_PROJECT')
        self.bq_dataset = bq_dataset or os.getenv('BQ_DATASET')
        self.bq_table = bq_table or os.getenv('BQ_TABLE')
        
        if BIGQUERY_AVAILABLE and self.gcp_project:
            self.client = bigquery.Client(project=self.gcp_project)
        else:
            self.client = None
            logger.warning("BigQuery client not available or project not "
                           "configured")
    
    def load_data_from_bigquery(self, entity_siren: str
                                ) -> tuple[pd.DataFrame, str]:
        """Load procurement data from BigQuery for a specific entity.
        
        Args:
            entity_siren: SIREN number of the entity to focus on
        
        Returns:
            Tuple of (DataFrame with contracts, entity_type)
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
                return pd.DataFrame(), None
            
            logger.info(f"Retrieved {len(df)} contracts for {entity_siren} as {entity_type}")


            selected_columns = ['dateNotification', 'acheteur_nom', 'acheteur_siren',
                                'titulaire_nom', 'titulaire_siren', 'montant', 
                                'dureeMois', 'codeCPV', 'procedure', 'objet']
        
            df = df[selected_columns]

            return df, entity_type
            
        except Exception as e:
            logger.error(f"Error querying BigQuery: {e}")
            raise

    def create_focused_graph(self, entity_siren: str,
                            min_contract_amount: float = 0) -> dict:
        """Create a graph focused on a specific entity using BigQuery data.
        
        Args:
            entity_siren: SIREN number of the central entity
            min_contract_amount: Minimum contract amount to include
        
        Returns:
            Graph data dictionary optimized for focused visualization
        """
        # Load data from BigQuery and determine entity type
        X_filtered, entity_type = self.load_data_from_bigquery(entity_siren)
        
        if X_filtered.empty or entity_type is None:
            logger.warning(f"No contracts found for SIREN: {entity_siren}")
            return None
        
        # Ensure required columns exist
        required_columns = ['acheteur_nom', 'titulaire_nom', 'montant',
                            'dureeMois', 'codeCPV', 'procedure']
        missing_columns = [col for col in required_columns
                           if col not in X_filtered.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with NaN buyer or supplier names
        valid_mask = (X_filtered['acheteur_nom'].notna() & 
                      X_filtered['titulaire_nom'].notna())
        X_filtered = X_filtered[valid_mask].copy()
        
        # Filter by minimum contract amount if specified
        if min_contract_amount > 0:
            amount_mask = X_filtered['montant'] >= min_contract_amount
            X_filtered = X_filtered[amount_mask].copy()
            logger.info(f"Filtered contracts with amount >= {min_contract_amount:,.2f}€")
        
        logger.info(f"Processing {len(X_filtered)} valid contracts")
        
        # Create focused graph structure
        if entity_type == 'titulaire':
            # Central node is the supplier, connected nodes are buyers
            # Get the actual name for display from the first matching record
            central_entity = X_filtered['titulaire_nom'].iloc[0] if not X_filtered.empty else f"SIREN {entity_siren}"
            connected_entities = X_filtered['acheteur_nom'].unique().tolist()
            central_type = 1  # Supplier
            connected_type = 0  # Buyers
        else:
            # Central node is the buyer, connected nodes are suppliers
            # Get the actual name for display from the first matching record
            central_entity = X_filtered['acheteur_nom'].iloc[0] if not X_filtered.empty else f"SIREN {entity_siren}"
            connected_entities = X_filtered['titulaire_nom'].unique().tolist()
            central_type = 0  # Buyer
            connected_type = 1  # Suppliers
        
        # Create node mappings - central node gets ID 0
        all_nodes = [central_entity] + connected_entities
        node_to_id = {node: i for i, node in enumerate(all_nodes)}
        
        # Create edges - all edges connect to the central node (ID 0)
        edges = []
        edge_features = []
        contract_data_list = []
        
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
                float(row['dureeMois']) if pd.notna(row['dureeMois']) else 0
            ])
            
            # Store contract data for analysis
            contract_data_list.append({
                'acheteur_nom': row['acheteur_nom'],
                'titulaire_nom': row['titulaire_nom'],
                'montant': row['montant'],
                'codeCPV': row.get('codeCPV', ''),
                'procedure': row.get('procedure', ''),
                'dureeMois': row['dureeMois']
            })
        
        # Compute node features
        logger.info("Computing node features...")
        
        # Central node features with proper NaN handling
        central_contracts = X_filtered
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
            if entity_type == 'titulaire':
                entity_contracts = X_filtered[X_filtered['acheteur_nom'] == connected_entity]
            else:
                entity_contracts = X_filtered[X_filtered['titulaire_nom'] == connected_entity]
            
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
        
        # Create contract data DataFrame
        contract_data = pd.DataFrame(contract_data_list)
        
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
        
        # Create network with larger size
        net = Network(height="800px", width="100%", directed=False)
        
        # Add nodes with special handling for the central node
        for i, (node_name, features, node_type) in enumerate(
                zip(nodes, node_features, node_types)):
            
            # Set node properties based on type and position
            total_amount = features[1] if len(features) > 1 else 0.0
            total_amount = total_amount if not pd.isna(total_amount) else 0.0
            
            if i == 0:  # Central node
                color = "#ffaa00"  # Orange for central node
                # Size based on total amount for central node
                size = min(50 + (total_amount / 50000), 100)  # More dramatic scaling
                shape = "star" if entity_type == 'titulaire' else "diamond"
                type_label = f"Central {entity_type.title()}"
                logger.info(f"Central node {node_name}: €{total_amount:,.2f} -> size {size}")
            elif node_type == 0:  # Buyer
                color = "#ff9999"  # Light red
                shape = "box"
                # Size based on total amount - more dramatic scaling
                size = min(10 + (total_amount / 50000), 70)  # Scale by 50k euros
                type_label = "Acheteur"
                logger.info(f"Buyer {node_name}: €{total_amount:,.2f} -> size {size}")
            else:  # Supplier
                color = "#99ccff"  # Light blue
                shape = "circle"
                # Size based on total amount - more dramatic scaling
                size = min(10 + (total_amount / 50000), 70)  # Scale by 50k euros
                type_label = "Titulaire"
                logger.info(f"Supplier {node_name}: €{total_amount:,.2f} -> size {size}")
            
            # Create tooltip with node information
            title = (f"{type_label}: {node_name}\n"
                     f"Contracts: {features[0]}\n"
                     f"Total Amount: {features[1]:,.2f}€\n"
                     f"Avg Amount: {features[2]:,.2f}€\n"
                     f"Avg Duration: {features[3]:.1f} months")
            
            # Position central node at center, others in a circle around it
            if i == 0:
                x, y = 0, 0  # Center
            else:
                # Arrange other nodes in a circle around the center
                angle = 2 * 3.14159 * (i - 1) / (len(nodes) - 1)
                radius = 800
                x = radius * nx.Graph().nodes.__class__.__dict__.get('cos', lambda a: 1)(angle)
                y = radius * nx.Graph().nodes.__class__.__dict__.get('sin', lambda a: 1)(angle)
                # Simplified positioning
                x = radius * (1 if (i % 2) else -1) * ((i - 1) / len(nodes))
                y = radius * (1 if (i % 4 < 2) else -1) * ((i - 1) / len(nodes))
            
            net.add_node(i, label=str(node_name)[:25], color=color,
                         size=size, shape=shape, title=title,
                         x=x, y=y, fixed=(i == 0))  # Fix central node position
        
        # Add edges with variable width based on contract amount
        for i, edge in enumerate(edges):
            if i < len(edge_features):
                amount = edge_features[i][0] if len(edge_features[i]) > 0 else 1
                width = min(2 + amount / 50000, 8)  # Scale width
            else:
                width = 2
            net.add_edge(edge[0], edge[1], width=width)
        
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
        logger.info(f"Total contracts: {len(edges)}")

    

if __name__ == "__main__":
    """Test the GraphPlotBuilder functionality."""
    import sys
    
    def test_bigquery_integration():
        """Test BigQuery integration with sample data."""
        print("Testing GraphPlotBuilder with BigQuery integration...")
        
        # Initialize builder with environment variables or defaults
        builder = GraphPlotBuilder()
        
        if not builder.client:
            print("Warning: BigQuery client not available. Skipping BigQuery tests.")
            print("To test BigQuery functionality, set these environment variables:")
            print("- GCP_PROJECT")
            print("- BQ_DATASET") 
            print("- BQ_TABLE")
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
        print("1. Set environment variables: GCP_PROJECT, BQ_DATASET, BQ_TABLE")
        print("2. Call create_focused_graph() with your entity names")
        print("3. Use plot_focused_graph() to generate visualizations")
    
    main()