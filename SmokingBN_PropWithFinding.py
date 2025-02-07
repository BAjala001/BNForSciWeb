import numpy as np
from Node import Node
from BayesNet import BayesNet
import pandas as pd

# Create smoking BN example as shown in the image
smokingBN = BayesNet(label="SmokingBN")

# Add the nodes
smokingBN.AddNode(Node(name="S", label="Smoking", values=["a", "b"], parents=[]))
smokingBN.AddNode(Node(name="B", label="Bronchitis", values=["0", "1"], parents=["S"]))
smokingBN.AddNode(Node(name="C", label="Lung Cancer", values=["0", "1"], parents=["S"]))

# Node S (Smoking)
sCpt = pd.DataFrame()
sCpt.loc[0, "Prob"] = 0.9  # P(S=0) = 0.9
sCpt.loc[1, "Prob"] = 0.1  # P(S=1) = 0.1
smokingBN.SetCpt("S", sCpt)

# Node B (Bronchitis)
bCpt = smokingBN.CreateCPT("B")
bCpt.loc[0, "Prob"] = 0.6  # P(B=0|S=0) = 0.6
bCpt.loc[1, "Prob"] = 0.4  # P(B=1|S=0) = 0.4
bCpt.loc[2, "Prob"] = 0.7  # P(B=0|S=1) = 0.7
bCpt.loc[3, "Prob"] = 0.3  # P(B=1|S=1) = 0.3
smokingBN.SetCpt("B", bCpt)

# Node C (Lung Cancer)
cCpt = smokingBN.CreateCPT("C")
cCpt.loc[0, "Prob"] = 0.9  # P(C=0|S=0) = 0.9
cCpt.loc[1, "Prob"] = 0.1  # P(C=1|S=0) = 0.1
cCpt.loc[2, "Prob"] = 0.8  # P(C=0|S=1) = 0.8
cCpt.loc[3, "Prob"] = 0.2  # P(C=1|S=1) = 0.2
smokingBN.SetCpt("C", cCpt)

# Calculate joint distribution
smokingBN.CalcJointDist()
smokingBN.MakeBN()
# print(smokingBN.jointDist)
# print('Sum of probabilities=',sum(smokingBN.jointDist['Pr']))

# Set some findings (for example, evidence that Smoking=a)
# smokingBN.SetFinding('S', 'a')  # or SetFinding('S', 'b') for the other value
smokingBN.SetFinding("S", "b")
smokingBN.SetFinding("B", "1")

# Recalculate marginals with the new findings
smokingBN.CalcMargGivenFindings()
# print(smokingBN.jointDist)

# Plot marginal distributions with findings
smokingBN.PlotMarginals(
    with_findings=True, finding_node_color="red", finding_node_penwidth=3.0
)
# smokingBN.CalcJointDist()
# smokingBN.MakeBN()
# print(smokingBN.jointDist)

# problematic_nodes = smokingBN.CheckCpts()
# print(problematic_nodes)
