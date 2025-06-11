#!/usr/bin/env python3
"""
Example usage of the GraphPlotBuilder with CSV data (for testing).

This script demonstrates how to:
1. Create sample CSV data with COLAS FRANCE and other entities
2. Use the original CSV functionality to create graphs
3. Focus visualizations on specific entities
"""

import os
import tempfile
import pandas as pd
from graph_plot_builder import GraphPlotBuilder


def create_sample_data():
    """Create sample procurement data with COLAS FRANCE and other entities."""
    sample_data = """dateNotification,acheteur_nom,titulaire_nom,montant,dureeMois,codeCPV,procedure,objet
2023-01-15,VILLE DE PARIS,COLAS FRANCE,500000,24,45233141,Open,Road Construction Project
2023-02-10,CONSEIL REGIONAL ILE-DE-FRANCE,COLAS FRANCE,750000,18,45233142,Restricted,Highway Maintenance
2023-03-05,METROPOLE DE LYON,COLAS FRANCE,300000,12,45233143,Open,Urban Road Repair
2023-01-20,VILLE DE PARIS,BOUYGUES CONSTRUCTION,450000,36,45212212,Negotiated,Building Construction
2023-02-15,VILLE DE PARIS,VINCI CONSTRUCTION,600000,30,45212213,Open,Infrastructure Project
2023-03-10,CONSEIL REGIONAL ILE-DE-FRANCE,EIFFAGE,400000,24,45212214,Restricted,Bridge Construction
2023-04-05,METROPOLE DE LYON,BOUYGUES CONSTRUCTION,350000,18,45212215,Open,Public Building
2023-04-12,VILLE DE MARSEILLE,COLAS FRANCE,280000,15,45233144,Open,Street Renovation
2023-05-08,DEPARTEMENT DU RHONE,VINCI CONSTRUCTION,520000,28,45212216,Restricted,Road Infrastructure
2023-05-15,VILLE DE TOULOUSE,COLAS FRANCE,420000,20,45233145,Open,Parking Construction"""
    
    return sample_data


def main():
    """Main function to demonstrate focused graph creation with CSV data."""
    print("=" * 60)
    print("GraphPlotBuilder CSV Example with COLAS FRANCE")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_data()
    
    try:
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', 
                                         delete=False) as f:
            f.write(sample_data)
            temp_csv = f.name
        
        # Create temporary directory structure
        temp_dir = os.path.dirname(temp_csv)
        data_clean_path = os.path.join(temp_dir, 'data_clean.csv')
        
        # Copy sample data to expected filename
        with open(data_clean_path, 'w') as f:
            f.write(sample_data)
        
        # Initialize GraphPlotBuilder
        builder = GraphPlotBuilder()
        
        # Load the CSV data
        print("Loading sample CSV data...")
        data = builder.load_data(temp_dir)
        print(f"✓ Loaded {len(data)} contracts from CSV")
        
        # Show data summary
        print(f"\nData Summary:")
        print(f"- Unique buyers: {data['acheteur_nom'].nunique()}")
        print(f"- Unique suppliers: {data['titulaire_nom'].nunique()}")
        print(f"- Total contract value: {data['montant'].sum():,.2f}€")
        
        # Example 1: Focus on COLAS FRANCE (supplier) using general graph
        print(f"\n" + "=" * 50)
        print("Example 1: Analyzing COLAS FRANCE contracts")
        print("=" * 50)
        
        colas_data = data[data['titulaire_nom'] == 'COLAS FRANCE']
        print(f"COLAS FRANCE statistics:")
        print(f"- Total contracts: {len(colas_data)}")
        print(f"- Total value: {colas_data['montant'].sum():,.2f}€")
        print(f"- Average value: {colas_data['montant'].mean():,.2f}€")
        print(f"- Clients: {', '.join(colas_data['acheteur_nom'].unique())}")
        
        # Create general graph
        graph_data = builder.create_graph(data, min_contract_amount=0)
        
        # Create visualization focused on COLAS FRANCE
        builder.plot_graph(
            graph_data=graph_data,
            output_path="colas_france_network.html",
            focus_node="COLAS FRANCE",
            max_nodes=20,
            max_edges=50,
            physics_enabled=True
        )
        print("✓ Created network visualization: colas_france_network.html")
        
        # Example 2: Focus on VILLE DE PARIS (buyer)
        print(f"\n" + "=" * 50)  
        print("Example 2: Analyzing VILLE DE PARIS contracts")
        print("=" * 50)
        
        paris_data = data[data['acheteur_nom'] == 'VILLE DE PARIS']
        print(f"VILLE DE PARIS statistics:")
        print(f"- Total contracts: {len(paris_data)}")
        print(f"- Total value: {paris_data['montant'].sum():,.2f}€")
        print(f"- Average value: {paris_data['montant'].mean():,.2f}€")
        print(f"- Suppliers: {', '.join(paris_data['titulaire_nom'].unique())}")
        
        # Create visualization focused on VILLE DE PARIS
        builder.plot_graph(
            graph_data=graph_data,
            output_path="ville_de_paris_network.html",
            focus_node="VILLE DE PARIS",
            max_nodes=20,
            max_edges=50,
            physics_enabled=True
        )
        print("✓ Created network visualization: ville_de_paris_network.html")
        
        # Example 3: Create a filtered graph for high-value contracts
        print(f"\n" + "=" * 50)
        print("Example 3: High-value contracts (>400,000€)")
        print("=" * 50)
        
        high_value_graph = builder.create_graph(data, min_contract_amount=400000)
        builder.plot_graph(
            graph_data=high_value_graph,
            output_path="high_value_contracts.html",
            max_nodes=15,
            max_edges=30,
            physics_enabled=True
        )
        
        high_value_data = data[data['montant'] >= 400000]
        print(f"High-value contracts summary:")
        print(f"- Count: {len(high_value_data)}")
        print(f"- Total value: {high_value_data['montant'].sum():,.2f}€")
        print("✓ Created high-value network: high_value_contracts.html")
        
        # Summary
        print(f"\n" + "=" * 60)
        print("Generated Visualizations:")
        print("- colas_france_network.html (focused on COLAS FRANCE)")
        print("- ville_de_paris_network.html (focused on VILLE DE PARIS)") 
        print("- high_value_contracts.html (contracts >400k€)")
        print("\nOpen these HTML files in your browser to explore the networks!")
        
        # Cleanup
        os.unlink(temp_csv)
        os.unlink(data_clean_path)
        
    except Exception as e:
        print(f"Error: {e}")
        # Cleanup on error
        try:
            os.unlink(temp_csv)
            os.unlink(data_clean_path)
        except:
            pass


if __name__ == "__main__":
    main() 