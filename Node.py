import numpy as np
import pandas as pd


class Node:
    def __init__(
        self,
        name: str,
        label: str = "",
        values: list[str] = [],
        parents: list[str] = [],
    ):
        self.name = name
        self.label = label
        self.values = values
        self.parents = parents
        self.children: list[str] = []
        self.cpt = pd.DataFrame()  # conditional probability table
        self.finding = dict()  # for storing evidence

        # Display attributes
        self.xPos = 0.0
        self.yPos = 0.0
        self.width = 0.9
        self.height = 0.4
        self.edgecolor = "blue"
        self.facecolor = "lightblue"
        self.linewidth = 1
        self.fontsize = 12
        self.textcolor = "black"
        self.ha = "center"
        self.va = "center"
        self.font = "palatino linotype"

    def set_finding(self, value: str):
        """Set a finding for this node"""
        self.finding = {self.name: value}

    def clear_finding(self):
        """Clear any finding for this node"""
        self.finding = dict()


if __name__ == "__main__":
    aNode = Node(name="a", label="b")
    print(aNode.label)
