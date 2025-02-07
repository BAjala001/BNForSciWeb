import numpy as np
import matplotlib
matplotlib.use('Agg')  # Change from TkAgg to Agg
import matplotlib.pyplot as plt
from Node import Node
from BayesNet import BayesNet
import pandas as pd
from BNTranslator import add_pgmpy_methods

# First, add pgmpy methods to BayesNet class
add_pgmpy_methods(BayesNet)

# Create smoking BN example
smokingBN = BayesNet(label="SmokingBN")

# Add the nodes
smokingBN.AddNode(Node(name="S", label="Smoking", values=["0", "1"], parents=[]))
smokingBN.AddNode(Node(name="B", label="Bronchitis", values=["0", "1"], parents=["S"]))
smokingBN.AddNode(Node(name="C", label="Cancer", values=["0", "1"], parents=["S"]))

# Add the CPTs
sCpt = smokingBN.CreateCPT("S")
sCpt.loc[0, "Prob"] = 0.9  # P(S=0) = 0.9
sCpt.loc[1, "Prob"] = 0.1  # P(S=1) = 0.1
smokingBN.SetCpt("S", sCpt)

bCpt = smokingBN.CreateCPT("B")
bCpt.loc[0, "Prob"] = 0.6  # P(B=0|S=0) = 0.6
bCpt.loc[1, "Prob"] = 0.4  # P(B=1|S=0) = 0.4
bCpt.loc[2, "Prob"] = 0.7  # P(B=0|S=1) = 0.7
bCpt.loc[3, "Prob"] = 0.3  # P(B=1|S=1) = 0.3
smokingBN.SetCpt("B", bCpt)

cCpt = smokingBN.CreateCPT("C")
cCpt.loc[0, "Prob"] = 0.9  # P(C=0|S=0) = 0.9
cCpt.loc[1, "Prob"] = 0.1  # P(C=1|S=0) = 0.1
cCpt.loc[2, "Prob"] = 0.8  # P(C=0|S=1) = 0.8
cCpt.loc[3, "Prob"] = 0.2  # P(C=1|S=1) = 0.2
smokingBN.SetCpt("C", cCpt)

# Add some findings
# smokingBN.SetFinding("S", "1")  # Evidence that S=1
smokingBN.SetFinding("C", "1")  # Evidence that C=1


# Before calculating anything, let's verify the nodes are properly added
# print("Nodes in network:", smokingBN.nodes)  # Add this debug line

# Calculate all necessary distributions
smokingBN.CalcJointDist()
smokingBN.CalcMarginals()
smokingBN.PlotDag()
