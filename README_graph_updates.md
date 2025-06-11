# Graph Plot Builder - BigQuery Integration

The `GraphPlotBuilder` module has been updated to support BigQuery integration and focused graph visualization.

## New Features

### 1. BigQuery Integration
- Query procurement data directly from BigQuery
- Support for filtering by specific entities (titulaire or acheteur)
- Automatic data validation and processing

### 2. Focused Graph Visualization
- Create graphs centered on a specific entity
- All contracts appear as edges connected to the central node
- Enhanced visualization with special styling for the central entity

## Setup

### Environment Variables
Set the following environment variables:
```bash
export GCP_PROJECT="your-gcp-project-id"
export BQ_DATASET="your-bigquery-dataset"
export BQ_TABLE="your-bigquery-table"
```

### Dependencies
Make sure you have the required dependencies:
```bash
pip install google-cloud-bigquery python-dotenv pandas pyvis networkx
```

## Usage

### Basic Usage
```python
from graph_plot_builder import GraphPlotBuilder

# Initialize with BigQuery configuration
builder = GraphPlotBuilder(
    gcp_project="your-project",
    bq_dataset="your-dataset", 
    bq_table="your-table"
)

# Create a focused graph for a specific supplier
graph_data = builder.create_focused_graph(
    entity_name="Supplier Name",
    entity_type="titulaire",  # or "acheteur"
    min_contract_amount=10000
)

# Generate visualization
builder.plot_focused_graph(
    graph_data=graph_data,
    output_path="supplier_graph.html"
)
```

### Key Methods

#### `create_focused_graph(entity_name, entity_type, min_contract_amount=0)`
- **entity_name**: Name of the central entity to focus on
- **entity_type**: Either "titulaire" (supplier) or "acheteur" (buyer)
- **min_contract_amount**: Optional minimum contract amount filter

#### `plot_focused_graph(graph_data, output_path, physics_enabled=True)`
- **graph_data**: Output from `create_focused_graph()`
- **output_path**: Path for the HTML visualization file
- **physics_enabled**: Whether to enable interactive physics simulation

### BigQuery Query Structure
The module automatically constructs queries like:
```sql
SELECT *
FROM {project}.{dataset}.{table}
WHERE titulaire_nom = 'Entity Name'
-- or
WHERE acheteur_nom = 'Entity Name'
```

## Graph Structure

### Node Types
- **Central Node**: The focused entity (star/diamond shape, orange color)
- **Buyers**: Connected buyers (box shape, light red)
- **Suppliers**: Connected suppliers (circle shape, light blue)

### Edge Features
- Edge width represents contract amount
- Each edge represents one or more contracts
- Hover tooltips show detailed contract information

## Example Output
The focused graph will show:
- Central entity at the center (fixed position)
- All related entities arranged around it
- Contract relationships as edges
- Interactive tooltips with financial details

## Error Handling
- Validates BigQuery connection and credentials
- Checks for required table columns
- Handles missing data gracefully
- Provides detailed logging for debugging 