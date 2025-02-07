import numpy as np
import pandas as pd
from Node import Node
from BayesNet import BayesNet
import matplotlib.pyplot as plt
import networkx as nx


def create_smoking_bn():
    smokingBN = BayesNet(label="SmokingBN")

    smokingBN.AddNode(Node(name="S", label="Smoking", values=["Yes", "No"], parents=[]))
    smokingBN.AddNode(
        Node(name="B", label="Bronchitis", values=["Yes", "No"], parents=["S"])
    )
    smokingBN.AddNode(
        Node(name="C", label="Cancer", values=["Yes", "No"], parents=["S"])
    )

    return smokingBN


def create_and_set_cpts(bn):
    # Node S
    sCpt = bn.CreateCPT("S")
    sCpt.loc[0, "Prob"] = 0.3  # Probability of smoking
    sCpt.loc[1, "Prob"] = 0.7  # Probability of not smoking
    bn.SetCpt("S", sCpt)

    # Node B
    bCpt = bn.CreateCPT("B")
    bCpt.loc[0, "Prob"] = 0.6  # P(B=Yes|S=Yes)
    bCpt.loc[1, "Prob"] = 0.4  # P(B=No|S=Yes)
    bCpt.loc[2, "Prob"] = 0.3  # P(B=Yes|S=No)
    bCpt.loc[3, "Prob"] = 0.7  # P(B=No|S=No)
    bn.SetCpt("B", bCpt)

    # Node C
    cCpt = bn.CreateCPT("C")
    cCpt.loc[0, "Prob"] = 0.3  # P(C=Yes|S=Yes)
    cCpt.loc[1, "Prob"] = 0.7  # P(C=No|S=Yes)
    cCpt.loc[2, "Prob"] = 0.1  # P(C=Yes|S=No)
    cCpt.loc[3, "Prob"] = 0.9  # P(C=No|S=No)
    bn.SetCpt("C", cCpt)


def plot_network(bn):
    G = nx.DiGraph()
    pos = {"S": (1, 2), "B": (1, 1), "C": (2, 1)}  # Flipped y-coordinates

    for node in bn.nodes:
        G.add_node(node.name)
        for parent in node.parents:
            G.add_edge(parent, node.name)

    plt.figure(figsize=(12, 8))

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, arrowsize=20, arrowstyle="-|>"
    )

    # Draw nodes and labels
    for node in bn.nodes:
        # Create CPT text
        cpt_text = f"{node.name}\n{node.label}\n"
        for _, row in node.cpt.iterrows():
            condition = " ".join(
                [f"{col}={row[col]}" for col in node.cpt.columns if col != "Prob"]
            )
            cpt_text += f"{condition}: {row['Prob']:.3f}\n"

        # Calculate node size based on text content, but with a smaller base size
        lines = cpt_text.count("\n") + 1
        node_size = max(3000, lines * 600)  # Reduced from max(5000, lines * 1000)

        # Draw custom node with smaller radius
        circle = plt.Circle(
            pos[node.name],
            radius=node_size / 15000,
            fill=True,
            color="lightblue",
            ec="blue",
        )
        plt.gca().add_patch(circle)

        # Add text inside the node with smaller font size
        plt.text(
            pos[node.name][0],
            pos[node.name][1],
            cpt_text,
            ha="center",
            va="center",
            fontsize=6,
            wrap=True,
        )

    plt.title("Smoking Bayesian Network")
    plt.axis("off")
    plt.tight_layout()
    plt.xlim(0.5, 2.5)
    plt.ylim(0.5, 2.5)
    plt.show()


def main():
    smokingBN = create_smoking_bn()
    create_and_set_cpts(smokingBN)
    plot_network(smokingBN)


if __name__ == "__main__":
    main()
