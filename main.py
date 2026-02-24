import pandas as pd
from src.dataset import load_dataset, convert_to_graph

def main():
    # Step 1: Load dataset
    df = load_dataset("data/train-00000-of-00001.parquet")
    print(f"Dataset loaded with {len(df)} rows")

    # Step 2: Convert each row into a graph
    graphs, all_nodes, all_edges = [], [], []
    for idx, row in df.iterrows():
        try:
            graph, node_info, edge_info = convert_to_graph(row, idx)
            graphs.append(graph)
            all_nodes.append(node_info)
            all_edges.append(edge_info)
        except Exception as e:
            print(f"Error converting row {idx}: {e}")

    # Step 3: Save node and edge info for paper
    if all_nodes:
        nodes_df = pd.concat(all_nodes, ignore_index=True)
        nodes_df.to_csv("results/graph_nodes.csv", index=False)
        print("Saved node info to results/graph_nodes.csv")

    if all_edges:
        edges_df = pd.concat(all_edges, ignore_index=True)
        edges_df.to_csv("results/graph_edges.csv", index=False)
        print("Saved edge info to results/graph_edges.csv")

    # Step 4: Summary
    print(f"Created {len(graphs)} graph objects")
    print("Project setup complete. You can now train your GNN using train.py")

if __name__ == "__main__":
    main()
