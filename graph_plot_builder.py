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
    
    def load_data_from_bigquery(self, entity_name: str,
                                entity_type: str = 'titulaire'
                                ) -> pd.DataFrame:
        """Load procurement data from BigQuery for a specific entity.
        
        Args:
            entity_name: Name of the entity to focus on
            entity_type: Either 'titulaire' or 'acheteur'
        
        Returns:
            DataFrame with contracts related to the specified entity
        """
        if not self.client:
            raise ValueError("BigQuery client not available. Check your "
                             "configuration.")
        
        if entity_type not in ['titulaire', 'acheteur']:
            raise ValueError("entity_type must be either 'titulaire' or "
                             "'acheteur'")
        
        # Build the query based on entity type
        if entity_type == 'titulaire':
            where_clause = f"titulaire_nom = '{entity_name}'"
        else:  # acheteur
            where_clause = f"acheteur_nom = '{entity_name}'"
        
        query = f"""
            SELECT *
            FROM {self.gcp_project}.{self.bq_dataset}.{self.bq_table}
            WHERE {where_clause}
        """
        
        logger.info(f"Querying BigQuery for {entity_type}: {entity_name}")
        logger.info(f"Query: {query}")
        
        try:
            query_job = self.client.query(query)
            result = query_job.result()
            df = result.to_dataframe()
            
            logger.info(f"Retrieved {len(df)} contracts for {entity_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error querying BigQuery: {e}")
            raise

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load procurement data from CSV files."""
        logger.info(f"Loading data from {data_path}")

        data_file_path = os.path.join(data_path, 'data_clean.csv')
        X = pd.read_csv(data_file_path, encoding='utf-8')
        # Basic data validation
        selected_columns = ['dateNotification', 'acheteur_nom',
                            'titulaire_nom', 'montant', 'dureeMois',
                            'codeCPV', 'procedure', 'objet']
        
        X = X[selected_columns]
        
        return X

    def create_focused_graph(self, entity_name: str, entity_type: str = 'titulaire',
                            min_contract_amount: float = 0) -> dict:
        """Create a graph focused on a specific entity using BigQuery data.
        
        Args:
            entity_name: Name of the central entity
            entity_type: Either 'titulaire' or 'acheteur'
            min_contract_amount: Minimum contract amount to include
        
        Returns:
            Graph data dictionary optimized for focused visualization
        """
        # Load data from BigQuery
        X_filtered = self.load_data_from_bigquery(entity_name, entity_type)
        
        if X_filtered.empty:
            logger.warning(f"No contracts found for {entity_type}: {entity_name}")
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
            central_entity = entity_name
            connected_entities = X_filtered['acheteur_nom'].unique().tolist()
            central_type = 1  # Supplier
            connected_type = 0  # Buyers
        else:
            # Central node is the buyer, connected nodes are suppliers
            central_entity = entity_name
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
        
        # Central node features
        central_contracts = X_filtered
        central_features = [
            len(central_contracts),
            float(central_contracts['montant'].sum()),
            float(central_contracts['montant'].mean()),
            float(central_contracts['dureeMois'].mean()) if central_contracts['dureeMois'].notna().any() else 0
        ]
        
        # Connected nodes features
        node_features = [central_features]
        node_types = [central_type]
        
        for connected_entity in connected_entities:
            if entity_type == 'titulaire':
                entity_contracts = X_filtered[X_filtered['acheteur_nom'] == connected_entity]
            else:
                entity_contracts = X_filtered[X_filtered['titulaire_nom'] == connected_entity]
            
            features = [
                len(entity_contracts),
                float(entity_contracts['montant'].sum()),
                float(entity_contracts['montant'].mean()),
                float(entity_contracts['dureeMois'].mean()) if entity_contracts['dureeMois'].notna().any() else 0
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

    def create_graph(self, X: pd.DataFrame,
                     min_contract_amount: float = 20_000) -> dict:
        """Transform procurement data into graph structure.
        
        Args:
            X: DataFrame from load_data with original columns
            min_contract_amount: Minimum contract amount to include
        """
        logger.info("Creating graph structure from procurement data...")
        
        # Remove rows with NaN buyer or supplier names
        valid_mask = (X['acheteur_nom'].notna() &
                      X['titulaire_nom'].notna())
        X_filtered = X[valid_mask].copy()
        
        # Filter by minimum contract amount if specified
        if min_contract_amount > 0:
            amount_mask = X_filtered['montant'] >= min_contract_amount
            X_filtered = X_filtered[amount_mask].copy()
            logger.info(f"Filtered contracts with amount >= "
                        f"{min_contract_amount:,.2f}€")
        
        logger.info(f"Filtered to {len(X_filtered)} valid contracts "
                    f"(removed {(~valid_mask).sum()} contracts with "
                    f"missing names)")
        
        # Create unique identifiers for buyers and suppliers
        buyers = X_filtered['acheteur_nom'].unique()
        suppliers = X_filtered['titulaire_nom'].unique()
        
        # Create node mappings
        buyer_to_id = {buyer: i for i, buyer in enumerate(buyers)}
        supplier_to_id = {supplier: i + len(buyers)
                          for i, supplier in enumerate(suppliers)}
        
        # Combine all nodes
        all_nodes = list(buyers) + list(suppliers)
        
        logger.info("Creating edges from procurement data...")
        
        # OPTIMIZATION: Vectorized edge creation
        buyer_ids = X_filtered['acheteur_nom'].map(buyer_to_id).values
        supplier_ids = X_filtered['titulaire_nom'].map(supplier_to_id).values
        edges = [(int(b), int(s)) for b, s in zip(buyer_ids, supplier_ids)]
        
        # OPTIMIZATION: Vectorized edge features creation
        edge_features_raw = (X_filtered[['montant', 'dureeMois']]
                             .fillna(0).values.tolist())
        # Convert numpy types to Python types for JSON serialization
        edge_features = [[float(x) for x in row] for row in edge_features_raw]
        
        # Store contract IDs for analysis
        contract_ids = [int(x) for x in X_filtered.index.tolist()]
        
        logger.info("Computing node features with vectorized operations...")
        
        # OPTIMIZATION: Bulk computation using groupby
        buyer_features, supplier_features = (
            self._compute_bulk_node_features(X_filtered))
        
        # Build node features arrays
        node_features = []
        node_types = []
        
        # Buyer features
        for buyer in buyers:
            features = buyer_features.get(buyer, [0, 0, 0, 0])
            # Convert numpy types to Python types for JSON serialization
            features = [float(x) if isinstance(x, (int, float)) else x for x in features]
            node_features.append(features)
            node_types.append(0)  # Buyer
        
        # Supplier features
        for supplier in suppliers:
            features = supplier_features.get(supplier, [0, 0, 0, 0])
            # Convert numpy types to Python types for JSON serialization
            features = [float(x) if isinstance(x, (int, float)) else x for x in features]
            node_features.append(features)
            node_types.append(1)  # Supplier
        
        # Create contract data for analysis
        contract_data = X_filtered[['acheteur_nom', 'titulaire_nom',
                                    'montant', 'codeCPV', 'procedure',
                                    'dureeMois']].copy()
        
        graph_data = {
            'nodes': all_nodes,
            'edges': edges,
            'node_features': node_features,
            'edge_features': edge_features,
            'node_types': node_types,
            'buyer_to_id': buyer_to_id,
            'supplier_to_id': supplier_to_id,
            'contract_ids': contract_ids,
            'contract_data': contract_data
        }

        # Save the graph data to a pickle file
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data')
        os.makedirs(data_dir, exist_ok=True)
        graph_file_path = os.path.join(data_dir, 'graph_data_clean.pkl')
        with open(graph_file_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        return graph_data

    def _compute_bulk_node_features(self, data: pd.DataFrame) -> tuple:
        """Compute features for all nodes using vectorized operations."""
        
        # Compute buyer features using groupby (much faster)
        buyer_stats = data.groupby('acheteur_nom').agg({
            'montant': ['count', 'sum', 'mean'],
            'dureeMois': 'mean'
        }).round(2)
        
        # Flatten column names
        buyer_stats.columns = ['contract_count', 'total_amount',
                               'avg_amount', 'avg_duration']
        buyer_stats = buyer_stats.fillna(0)
        
        # Convert to dictionary format
        buyer_features = {}
        for buyer in buyer_stats.index:
            row = buyer_stats.loc[buyer]
            buyer_features[buyer] = [
                int(float(row['contract_count'])),
                float(row['total_amount']),
                float(row['avg_amount']),
                float(row['avg_duration'])
            ]
        
        # Compute supplier features using groupby
        supplier_stats = data.groupby('titulaire_nom').agg({
            'montant': ['count', 'sum', 'mean'],
            'dureeMois': 'mean'
        }).round(2)
        
        # Flatten column names
        supplier_stats.columns = ['contract_count', 'total_amount',
                                  'avg_amount', 'avg_duration']
        supplier_stats = supplier_stats.fillna(0)
        
        # Convert to dictionary format
        supplier_features = {}
        for supplier in supplier_stats.index:
            row = supplier_stats.loc[supplier]
            supplier_features[supplier] = [
                int(float(row['contract_count'])),
                float(row['total_amount']),
                float(row['avg_amount']),
                float(row['avg_duration'])
            ]
        
        return buyer_features, supplier_features

    def _compute_node_features(self, data: pd.DataFrame,
                               entity_col: str, entities: list) -> dict:
        """Compute features for nodes based on their contracts."""
        features = {}
        
        for entity in entities:
            entity_data = data[data[entity_col] == entity]
            
            # Basic statistics
            contract_count = len(entity_data)
            total_amount = (entity_data['montant'].sum()
                            if len(entity_data) > 0 else 0)
            avg_amount = (entity_data['montant'].mean()
                          if len(entity_data) > 0 else 0)
            avg_duration = (entity_data['dureeMois'].mean()
                            if len(entity_data) > 0 else 0)
            
            features[entity] = [
                contract_count,
                total_amount,
                avg_amount,
                avg_duration
            ]
        
        return features

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
        net = Network(height="1000px", width="100%", directed=False)
        
        # Add nodes with special handling for the central node
        for i, (node_name, features, node_type) in enumerate(
                zip(nodes, node_features, node_types)):
            
            # Set node properties based on type and position
            if i == 0:  # Central node
                color = "#ffaa00"  # Orange for central node
                size = 60  # Large size for central node
                shape = "star" if entity_type == 'titulaire' else "diamond"
                type_label = f"Central {entity_type.title()}"
            elif node_type == 0:  # Buyer
                color = "#ff9999"  # Light red
                shape = "box"
                size = min(20 + features[0] * 2, 40)
                type_label = "Acheteur"
            else:  # Supplier
                color = "#99ccff"  # Light blue
                shape = "circle"
                size = min(20 + features[0] * 2, 40)
                type_label = "Titulaire"
            
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
                "stabilization": {"iterations": 100},
                "barnesHut": {
                  "gravitationalConstant": -2000,
                  "centralGravity": 0.3,
                  "springLength": 200,
                  "springConstant": 0.02,
                  "damping": 0.4,
                  "avoidOverlap": 0.8
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

    def plot_graph(self, graph_data: dict,
                   output_path: str = "graph_visualization.html",
                   focus_node: Optional[Union[str, int]] = None,
                   max_nodes: int = 100,
                   max_edges: int = 200,
                   physics_enabled: bool = True) -> None:
        """Visualize the graph using pyvis network.
        
        Args:
            graph_data: Graph data dictionary from create_graph
            output_path: Path to save the HTML visualization
            focus_node: Node name or ID to zoom/focus on
            max_nodes: Maximum number of nodes to display
            max_edges: Maximum number of edges to display
            physics_enabled: Whether to enable physics simulation
        """
        logger.info("Creating graph visualization with pyvis...")
        
        # Get graph components
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        node_features = graph_data['node_features']
        node_types = graph_data['node_types']
        edge_features = graph_data['edge_features']
        
        # Limit nodes if necessary
        if len(nodes) > max_nodes:
            logger.info(f"Limiting display to {max_nodes} nodes "
                        f"(total: {len(nodes)})")
            # Get top nodes by contract count
            node_importance = [features[0] for features in node_features]
            top_indices = sorted(range(len(node_importance)),
                                 key=lambda i: node_importance[i],
                                 reverse=True)[:max_nodes]
            
            # Filter nodes and features
            nodes = [nodes[i] for i in top_indices]
            node_features = [node_features[i] for i in top_indices]
            node_types = [node_types[i] for i in top_indices]
            
            # Create mapping for filtered nodes
            old_to_new = {old_idx: new_idx
                          for new_idx, old_idx in enumerate(top_indices)}
            
            # Filter edges to only include nodes in our subset
            filtered_edges = []
            filtered_edge_features = []
            for i, edge in enumerate(edges):
                if edge[0] in old_to_new and edge[1] in old_to_new:
                    filtered_edges.append((old_to_new[edge[0]],
                                           old_to_new[edge[1]]))
                    filtered_edge_features.append(edge_features[i])
            edges = filtered_edges
            edge_features = filtered_edge_features
        
        # Limit total edges if needed
        if len(edges) > max_edges:
            logger.info(f"Limiting edges to {max_edges} (total: {len(edges)})")
            
            # Create list of (edge, features, amount) for sorting
            edge_data = []
            for i, (edge, features) in enumerate(zip(edges, edge_features)):
                amount = features[0] if len(features) > 0 else 0
                edge_data.append((edge, features, amount))
            
            # Sort by contract amount (descending) and limit
            edge_data.sort(key=lambda x: x[2], reverse=True)
            edge_data = edge_data[:max_edges]
            
            # Extract filtered edges and features
            edges = [item[0] for item in edge_data]
            edge_features = [item[1] for item in edge_data]
            
            logger.info(f"Using {len(edges)} edges after filtering")
        
        # Create network with larger size
        net = Network(height="1000px", width="100%", directed=False)

        # net = Network(height="600px", width="100%", directed=False,
        #                select_menu=True, filter_menu=True)
        
        # Create initial positions to avoid center clustering
        logger.info("Computing initial node positions...")
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from(edges)
        
        # Use spring layout for initial positioning with more spacing
        try:
            pos = nx.spring_layout(G, k=5, iterations=50, seed=42)
        except Exception:
            # Fallback to circular layout if spring layout fails
            pos = nx.circular_layout(G)
        
        # Scale positions to fill the canvas better with much larger spread
        scale_factor = 3000  # Dramatically increased for much larger spacing
        for node_id in pos:
            pos[node_id] = (pos[node_id][0] * scale_factor,
                            pos[node_id][1] * scale_factor)
        
        # Add nodes to the network with initial positions
        for i, (node_name, features, node_type) in enumerate(
                zip(nodes, node_features, node_types)):
            
            # Set node properties based on type
            if node_type == 0:  # Buyer
                color = "#ff9999"  # Light red
                shape = "box"
                type_label = "Acheteur"
            else:  # Supplier
                color = "#99ccff"  # Light blue
                shape = "circle"
                type_label = "Titulaire"
            
            # Scale node size based on contract count
            size = min(15 + features[0] * 3, 60)
            
            # Create tooltip with node information
            title = (f"{type_label}: {node_name}\n"
                     f"Contracts: {features[0]}\n"
                     f"Total Amount: {features[1]:,.2f}€\n"
                     f"Avg Amount: {features[2]:,.2f}€\n"
                     f"Avg Duration: {features[3]:.1f} months")
            
            # Highlight focus node if specified
            if focus_node is not None:
                if ((isinstance(focus_node, str) and
                     node_name == focus_node) or
                        (isinstance(focus_node, int) and i == focus_node)):
                    color = "#ffff00"  # Yellow for focus
                    size *= 1.5
            
            # Get initial position for this node with wider range
            x, y = pos.get(i, (random.uniform(-1500, 1500),
                               random.uniform(-1500, 1500)))
            
            net.add_node(i, label=str(node_name)[:20], color=color,
                         size=size, shape=shape, title=title,
                         x=x, y=y, fixed=False)
        
        # Add edges to the network with variable width
        for i, edge in enumerate(edges):
            # Scale edge width based on contract amount
            if i < len(edge_features):
                features_list = edge_features[i]
                amount = features_list[0] if len(features_list) > 0 else 1
                width = min(1 + amount / 50000, 5)  # Scale width
            else:
                width = 1
            net.add_edge(edge[0], edge[1], width=width)
        
        # Configure physics for stability
        if physics_enabled:
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 50},
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "centralGravity": 0.01,
                  "springLength": 300,
                  "springConstant": 0.005,
                  "damping": 0.3,
                  "avoidOverlap": 1.0
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
        logger.info(f"Graph visualization saved to {output_path}")
        logger.info(f"Visualization created with {len(nodes)} nodes "
                    f"and {len(edges)} edges")
        if focus_node is not None:
            logger.info(f"Focused on node: {focus_node}")


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
            # Test with a sample entity name
            sample_entity = "COLAS FRANCE"
            print(f"Testing focused graph creation for: {sample_entity}")
            
            # Test titulaire focus
            graph_data = builder.create_focused_graph(
                entity_name=sample_entity,
                entity_type="titulaire",
                min_contract_amount=1000
            )
            
            if graph_data:
                print("✓ Successfully created focused graph for titulaire")
                builder.plot_focused_graph(
                    graph_data=graph_data,
                    output_path="test_titulaire_graph.html"
                )
                print("✓ Successfully generated visualization")
                return True
            else:
                print(f"No contracts found for {sample_entity}")
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
        csv_success = test_csv_functionality()
        
        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"BigQuery Integration: {'✓ PASS' if bigquery_success else '✗ SKIP/FAIL'}")
        print(f"CSV Functionality: {'✓ PASS' if csv_success else '✗ FAIL'}")
        
        if csv_success:
            print("\n✓ Core functionality working correctly!")
            print("Generated test files:")
            print("- test_general_graph.html")
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