#!/usr/bin/env python3
"""
Example usage of the GraphPlotBuilder with BigQuery integration.

This script demonstrates how to:
1. Initialize the GraphPlotBuilder with BigQuery configuration
2. Create a focused graph for a specific entity (titulaire or acheteur)
3. Visualize the graph with the entity as the central node
"""

import os
from graph_plot_builder import GraphPlotBuilder


def main():
    # Example configuration - these should be set in your environment
    # or passed directly to the constructor
    GCP_PROJECT = os.getenv('GCP_PROJECT', 'your-gcp-project')
    BQ_DATASET = os.getenv('BQ_DATASET', 'your-dataset')
    BQ_TABLE = os.getenv('BQ_TABLE', 'your-table')
    
    # Initialize the graph builder
    builder = GraphPlotBuilder(
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        bq_table=BQ_TABLE
    )
    
    # Example 1: Focus on a specific supplier (titulaire)
    entity_name = "COLAS FRANCE"
    entity_type = "titulaire"  # or "acheteur"
    
    try:
        # Create focused graph data
        print(f"Creating focused graph for {entity_type}: {entity_name}")
        graph_data = builder.create_focused_graph(
            entity_name=entity_name,
            entity_type=entity_type,
            min_contract_amount=10000  # Optional: filter by minimum amount
        )
        
        if graph_data:
            # Generate the visualization
            output_file = (f"focused_graph_{entity_type}_"
                           f"{entity_name.replace(' ', '_')}.html")
            builder.plot_focused_graph(
                graph_data=graph_data,
                output_path=output_file,
                physics_enabled=True
            )
            print(f"Visualization saved to: {output_file}")
            
            # Print some statistics
            print("Graph statistics:")
            print(f"- Central entity: {graph_data['central_entity']}")
            print(f"- Entity type: {graph_data['entity_type']}")
            print(f"- Total nodes: {len(graph_data['nodes'])}")
            print(f"- Total contracts: {len(graph_data['edges'])}")
            print(f"- Connected entities: {len(graph_data['nodes']) - 1}")
            
        else:
            print(f"No contracts found for {entity_type}: {entity_name}")
            
    except Exception as e:
        print(f"Error creating focused graph: {e}")
    
    # Example 2: Focus on a specific buyer (acheteur)
    entity_name_buyer = "VILLE DE PARIS"
    entity_type_buyer = "acheteur"
    
    try:
        print(f"\nCreating focused graph for {entity_type_buyer}: "
              f"{entity_name_buyer}")
        graph_data_buyer = builder.create_focused_graph(
            entity_name=entity_name_buyer,
            entity_type=entity_type_buyer,
            min_contract_amount=5000
        )
        
        if graph_data_buyer:
            output_file_buyer = (f"focused_graph_{entity_type_buyer}_"
                                 f"{entity_name_buyer.replace(' ', '_')}.html")
            builder.plot_focused_graph(
                graph_data=graph_data_buyer,
                output_path=output_file_buyer,
                physics_enabled=True
            )
            print(f"Buyer visualization saved to: {output_file_buyer}")
            
        else:
            print(f"No contracts found for {entity_type_buyer}: "
                  f"{entity_name_buyer}")
            
    except Exception as e:
        print(f"Error creating buyer focused graph: {e}")


if __name__ == "__main__":
    main() 