from Node import Node
from BayesNet import BayesNet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BNTranslator import add_pgmpy_methods

# Add PGMPY methods to BayesNet class BEFORE creating the network
add_pgmpy_methods(BayesNet)

# Now create your network
sibBN = BayesNet("SiblingsBN")

mmaNode = Node(name="mma", label="MotherMatAllele", values=["a", "b"], parents=[])
sibBN.AddNode(mmaNode)

mpaNode = Node(name="mpa", label="MotherPatAllele", values=["a", "b"], parents=[])
sibBN.AddNode(mpaNode)

fmaNode = Node(name="fma", label="FatherMatAllele", values=["a", "b"], parents=[])
sibBN.AddNode(fmaNode)

fpaNode = Node(name="fpa", label="FatherPatAllele", values=["a", "b"], parents=[])
sibBN.AddNode(fpaNode)

c1maNode = Node(
    name="c1ma", label="Child1MatAllele", values=["a", "b"], parents=["mma", "mpa"]
)
sibBN.AddNode(c1maNode)

c1paNode = Node(
    name="c1pa", label="Child1PatAllele", values=["a", "b"], parents=["fma", "fpa"]
)
sibBN.AddNode(c1paNode)

c2maNode = Node(
    name="c2ma", label="Child2MatAllele", values=["a", "b"], parents=["mma", "mpa"]
)
sibBN.AddNode(c2maNode)

c2paNode = Node(
    name="c2pa", label="Child2PatAllele", values=["a", "b"], parents=["fma", "fpa"]
)
sibBN.AddNode(c2paNode)

c1gtNode = Node(
    name="c1gt",
    label="Child1Genotype",
    values=["aa", "ab", "bb"],
    parents=["c1ma", "c1pa"],
)
sibBN.AddNode(c1gtNode)

c2gtNode = Node(
    name="c2gt",
    label="Child2Genotype",
    values=["aa", "ab", "bb"],
    parents=["c2ma", "c2pa"],
)
sibBN.AddNode(c2gtNode)

# Adjust xPos of specific nodes by name (instead of integer indexing)
if "c1gt" in sibBN.nodes:
    sibBN.nodes["c1gt"].xPos += 0.5
if "c2gt" in sibBN.nodes:
    sibBN.nodes["c2gt"].xPos += 1.5

# Set CPTs
sibBN.SetCpt("mma", pd.DataFrame({"ma": ["a", "b"], "Prob": [0.6, 0.4]}))
sibBN.SetCpt("mpa", pd.DataFrame({"pa": ["a", "b"], "Prob": [0.6, 0.4]}))

sibBN.SetCpt("fma", pd.DataFrame({"ma": ["a", "b"], "Prob": [0.6, 0.4]}))
sibBN.SetCpt("fpa", pd.DataFrame({"pa": ["a", "b"], "Prob": [0.6, 0.4]}))

# c1ma
c1maCpt = sibBN.CreateCPT("c1ma")
c1maCpt["Prob"] = 0.5
for idx, row in c1maCpt.iterrows():
    if row["mma"] == row["mpa"] and row["mma"] == row["c1ma"]:
        c1maCpt.loc[idx, "Prob"] = 1.0
    elif row["mma"] == row["mpa"] and row["mma"] != row["c1ma"]:
        c1maCpt.loc[idx, "Prob"] = 0.0
sibBN.SetCpt("c1ma", c1maCpt)

# c1pa
c1paCpt = sibBN.CreateCPT("c1pa")
c1paCpt["Prob"] = c1maCpt["Prob"]  # same pattern
sibBN.SetCpt("c1pa", c1paCpt)

# c2ma
c2maCpt = sibBN.CreateCPT("c2ma")
c2maCpt["Prob"] = c1maCpt["Prob"]
sibBN.SetCpt("c2ma", c2maCpt)

# c2pa
c2paCpt = sibBN.CreateCPT("c2pa")
c2paCpt["Prob"] = c1maCpt["Prob"]
sibBN.SetCpt("c2pa", c2paCpt)

# c1gt
c1gtCpt = sibBN.CreateCPT("c1gt")
c1gtCpt["Prob"] = 0.0
for idx, row in c1gtCpt.iterrows():
    # sorted([a,b]) -> ab matched to c1gt
    if "".join(sorted([row["c1ma"], row["c1pa"]])) == row["c1gt"]:
        c1gtCpt.loc[idx, "Prob"] = 1.0
sibBN.SetCpt("c1gt", c1gtCpt)

# c2gt
c2gtCpt = sibBN.CreateCPT("c2gt")
c2gtCpt["Prob"] = c1gtCpt["Prob"]
sibBN.SetCpt("c2gt", c2gtCpt)

# Calculate joint distribution first
# sibBN.CalcJointDist()

# Now set findings
sibBN.SetFinding("c1gt", "ab")
sibBN.SetFinding("c2gt", "ab")

# Calculate marginals with findings before plotting
# sibBN.CalcMargGivenFindings()

# Finally, plot marginals with findings
sibBN.PlotMarginals(with_findings=False)
