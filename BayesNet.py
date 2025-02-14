import numpy as np
import pandas as pd
from Node import Node
import matplotlib
matplotlib.use('Agg')  # Use Agg backend instead of TkAgg
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import pickle
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
from io import BytesIO
import base64
from BNTranslator import create_pgmpy_model, get_marginals
import os


class BayesNet:
    def __init__(self, label: str, nodeSizeForDisp=5000):
        self.label = (
            label  # this is the only compulsory variable to create a BayesNet object
        )
        self.nodes = {}
        self.jointDist = pd.DataFrame()

        # Store marginals in dictionaries keyed by node_name
        self.marginals = {}
        self.margWithFindings = {}
        self.margWithNewMarg = {}

        self.nodeSizeForDisp = nodeSizeForDisp

    def CalcYPos(self):
        """
        Computes yPos for each node in the network such that:
         • Nodes with no parents have yPos = 1.
         • A child's yPos = parent's yPos + 1 (at least).
        """
        nodeQueue = []

        # First, set root nodes (no parents) to yPos = 1, others to yPos = 0.
        for node_name, node in self.nodes.items():
            if not node.parents:  # root node
                node.yPos = 1
                nodeQueue.append(node_name)
            else:
                node.yPos = 0

        # BFS-like approach to raise children's yPos
        while nodeQueue:
            thisName = nodeQueue.pop(0)
            thisNode = self.nodes[thisName]
            for childName in thisNode.children:
                childNode = self.nodes[childName]
                newYPos = thisNode.yPos + 1
                if childNode.yPos < newYPos:
                    childNode.yPos = newYPos
                if childName not in nodeQueue:
                    nodeQueue.append(childName)

        # Final pass: for nodes still at yPos=0 but having parents,
        # set them to max(parent.yPos)+1
        for node_name, node in self.nodes.items():
            if node.yPos == 0 and node.parents:
                node.yPos = max(self.nodes[p].yPos for p in node.parents) + 1

    def CalcXPos(self, xShift=0.2):
        """
        Calculate and assign xPos for each node in the network,
        using node.yPos to group nodes by layer.

        Ensures yPos is already calculated. Then for each distinct yPos:
        1) Gather all nodes with that yPos.
        2) Assign their xPos values incrementally.

        Returns:
            A list of xPos values for all nodes.
        """
        # Ensure yPos is calculated first
        self.CalcYPos()

        # Get all distinct yPos values from node objects
        all_y_positions = sorted(set(node.yPos for node in self.nodes.values()))

        # For each distinct yPos, assign xPos values incrementally
        for yPos in all_y_positions:
            # Find all node objects at this yPos
            nodes_in_this_layer = [
                node for node in self.nodes.values() if node.yPos == yPos
            ]

            currentXpos = 1 + yPos * xShift
            for node in nodes_in_this_layer:
                node.xPos = currentXpos
                currentXpos += 1

        # Return a list of all node xPos values
        return [node.xPos for node in self.nodes.values()]

    def CalcChildren(self):
        # Initialize children list for each node
        for node_name in self.nodes:
            self.nodes[node_name].children = []

        # For each node, check if it's a parent of other nodes
        for node_name in self.nodes:
            for other_node_name in self.nodes:
                if node_name in self.nodes[other_node_name].parents:
                    self.nodes[node_name].children.append(other_node_name)

    def CalcNumberOfLayers(self):
        return max(node.yPos for node in self.nodes.values())

    def CalcMaxNumberOfNodesInLayers(self):
        layers = [node.xPos for node in self.nodes.values()]
        return max(layers)

    def GetNodeNames(self):
        nNodes = len(self.nodes)
        nodeNames = []
        for nodeIdx in range(nNodes):
            nodeNames = [*nodeNames, self.nodes[nodeIdx].name]
        return nodeNames

    def GetNodeLabel(self, nodeName):
        for aNode in self.nodes:
            if aNode.name == nodeName:
                return aNode.label

    def GetNodeParents(self, nodeName):
        if nodeName in self.nodes:
            return self.nodes[nodeName].parents
        return None  # or raise an exception if the node isn't found

    def GetNodeValues(self, nodeName):
        if nodeName in self.nodes:
            return self.nodes[nodeName].values
        return None

    def PlotDag(self):
        """Plot the Directed Acyclic Graph (DAG) of the network"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create figure and axis
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Calculate positions if not already calculated
        self.CalcYPos()
        self.CalcXPos()
        
        # Draw nodes and store node boundaries
        node_boundaries = {}
        for node_name, node in self.nodes.items():
            # Create ellipse for node
            ellipse = patches.Ellipse(
                (node.xPos, -node.yPos),
                width=node.width,
                height=node.height,
                facecolor=node.facecolor,
                edgecolor=node.edgecolor,
                linewidth=node.linewidth,
                alpha=0.7
            )
            ax.add_patch(ellipse)
            
            # Store node boundaries
            node_boundaries[node_name] = {
                'center': (node.xPos, -node.yPos),
                'width': node.width,
                'height': node.height
            }
            
            # Add node label
            ax.text(
                node.xPos,
                -node.yPos,
                f"{node.label}\n({node_name})",
                horizontalalignment=node.ha,
                verticalalignment=node.va,
                fontsize=node.fontsize,
                color=node.textcolor
            )
            
            # Draw arrows between nodes
            for parent in node.parents:
                # Get node boundaries
                child = node_boundaries[node_name]
                parent_bound = node_boundaries[parent]
                
                # Calculate intersection points with ellipses
                dx = child['center'][0] - parent_bound['center'][0]
                dy = child['center'][1] - parent_bound['center'][1]
                angle = np.arctan2(dy, dx)
                
                # Calculate start point (on parent ellipse)
                start_x = parent_bound['center'][0] + (parent_bound['width']/2) * np.cos(angle)
                start_y = parent_bound['center'][1] + (parent_bound['height']/2) * np.sin(angle)
                
                # Calculate end point (on child ellipse)
                end_x = child['center'][0] - (child['width']/2) * np.cos(angle)
                end_y = child['center'][1] - (child['height']/2) * np.sin(angle)
                
                # Draw arrow
                arrow = patches.FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    connectionstyle="arc3,rad=0",  # Straight lines
                    arrowstyle="-|>",
                    mutation_scale=15,
                    color='gray'
                )
                ax.add_patch(arrow)
        
        # Set plot bounds
        all_x = [node.xPos for node in self.nodes.values()]
        all_y = [-node.yPos for node in self.nodes.values()]
        margin = 0.5
        
        if all_x and all_y:  # Check if lists are not empty
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.title("Network Structure")
        plt.tight_layout()
        
        return fig

    def AddNode(self, aNode: Node):
        # From the GUI, if there is no label, pass label=''
        # If there are no parents, pass parents =[]
        if not isinstance(aNode, Node):
            raise TypeError("Must add a Node object, not a string")
        self.nodes[aNode.name] = aNode
        # the task should be sepaeated. One is create a node and then add it.
        # it can be called aNode or node.

    def CreateCPT(self, nodeName):
        parents = self.GetNodeParents(nodeName)
        nCols = len(parents) + 2
        cptLens = np.zeros(nCols - 1)
        i = -1
        for aParent in parents:
            i += 1
            values = self.GetNodeValues(aParent)
            cptLens[i] = len(values)
        cptLens[i + 1] = len(self.GetNodeValues(nodeName))
        # print('cptLens=',cptLens)
        nRows = np.prod(cptLens)
        # print('nRows=',nRows)
        # print('nCols=',nCols)
        colNames = parents + [nodeName, "Prob"]
        # print('colNames=',colNames)
        nodeValues = self.GetNodeValues(nodeName)

        allValues = [None for _ in range(nCols - 1)]
        # print('CreateCPT:L96:allValies = ', allValues)
        for parentIdx in range(len(parents)):
            allValues[parentIdx] = self.GetNodeValues(parents[parentIdx])
        allValues[nCols - 2] = nodeValues
        # print('CreateCPT:L99:allValies = ', allValues)
        cpt = np.array(list(itertools.product(*allValues, [0])))
        # print('CreateCPT:L102:cpt = ', cpt)

        cpt = pd.DataFrame(cpt, columns=colNames)
        # print('CreateCPT:L120:cpt = ', cpt)
        # print(cpt)
        return cpt

    def GetNodeIndex(self, nodeName):
        if nodeName in self.nodes:
            return nodeName  # or return the actual index if you need it
        return None

    def SetCpt(self, nodeName, cpt: pd.DataFrame):
        nodeIdx = self.GetNodeIndex(nodeName)
        self.nodes[nodeIdx].cpt = cpt

    def Save(self):
        with open(self.label + ".pkl", "wb") as file:
            pickle.dump(self, file)

    def CalcJointDist(self):
        # Create the joint distribution pd data frame.
        if isinstance(self.nodes, dict):
            nodeNames = [node.name for node in self.nodes.values()]
            allNodeValues = [node.values for node in self.nodes.values()]
        else:
            nodeNames = [node.name for node in self.nodes]
            allNodeValues = [node.values for node in self.nodes]

        allNodeValuesMatrix = np.array(list(itertools.product(*allNodeValues)))

        self.jointDist = pd.DataFrame(allNodeValuesMatrix, columns=nodeNames)
        self.jointDist["Pr"] = 0.0  # Initialize probability column

        # Populate the joint distribution
        for idx, row in self.jointDist.iterrows():
            prob = 1.0
            for node_name, node in (
                self.nodes.items()
                if isinstance(self.nodes, dict)
                else enumerate(self.nodes)
            ):
                node_cpt = node.cpt
                if not node.parents:
                    # For root nodes (no parents)
                    value_idx = node.values.index(row[node.name])
                    prob *= node_cpt.loc[value_idx, "Prob"]
                else:
                    # For nodes with parents
                    condition = np.ones(len(node_cpt), dtype=bool)
                    for parent in node.parents:
                        condition &= node_cpt[parent] == row[parent]
                    condition &= node_cpt[node.name] == row[node.name]
                    prob *= node_cpt.loc[condition, "Prob"].values[0]

            self.jointDist.loc[idx, "Pr"] = prob

        # Sort the DataFrame to match the order in the image
        self.jointDist = self.jointDist.sort_values(by=nodeNames).reset_index(drop=True)

    def MakeBN(self):
        # Assumption: The GUI will ensure that the parents of a node exist.
        # 1. Check the Conditional Probability Tables (CPTs). Ensure conditional probabilities add up to one
        #    If not, bring a text box saying something like: Probabilities in the CPT of node 'node name' do not add up to one.

        # Calculates the initial position of the nodes; later the user can modified them in the GUI
        self.CalcChildren()
        self.CalcYPos()
        self.CalcXPos()

        # 2. Once checks are done, then calculate the joint distribution.
        self.CalcJointDist()  # writes the joint in self.jointDist

        # 2. Once checks are done, then calculate the joint distribution.
        # self.CalcJointDist() # writes the joint in self.jointDist

    def CreateMarginals(self, margType="standard"):
        """Creates empty marginal DataFrames stored in dictionaries."""
        # Initialize the appropriate dictionary based on margType
        if margType == "standard":
            self.marginals = {}
        elif margType == "finding":
            self.margWithFindings = {}
        elif margType == "newMarg":
            self.margWithNewMarg = {}
        else:
            return

        # Create placeholder DataFrames for each node
        for node_name, node in self.nodes.items():
            # Create DataFrame with node values and zero probabilities
            df = pd.DataFrame(
                {node_name: node.values, "Prob": [0.0] * len(node.values)}
            )

            # Store in appropriate dictionary
            if margType == "standard":
                self.marginals[node_name] = df
            elif margType == "finding":
                self.margWithFindings[node_name] = df
            elif margType == "newMarg":
                self.margWithNewMarg[node_name] = df

    def CalcMarginals(self):
        """Calculate marginals using PGMPY"""
        # Create PGMPY model
        pgmpy_model = create_pgmpy_model(self)

        # Calculate marginals without evidence
        marginals_list = get_marginals(pgmpy_model)

        # Convert list of marginals to dictionary
        self.marginals = {}
        for node_name, marginal_df in zip(self.nodes.keys(), marginals_list):
            self.marginals[node_name] = marginal_df

    def CalcMargGivenFindings(self):
        """Calculate marginals given findings using pgmpy."""
        # Initialize margWithFindings as a dictionary
        self.margWithFindings = {}

        # Get evidence dictionary from nodes with findings
        evidence = {}
        for node_name, node in self.nodes.items():
            if node.finding:
                evidence.update(node.finding)

        # Create and use pgmpy model
        pgmpy_model = create_pgmpy_model(self)
        marginals_list = get_marginals(pgmpy_model, evidence)

        # Convert list of marginals to dictionary
        for node_name, marginal_df in zip(self.nodes.keys(), marginals_list):
            self.margWithFindings[node_name] = marginal_df

    def CalcMargGivenNewMarg(self):
        self.CreateMarginals(margType="newMarg")
        nNodes = len(self.nodes)
        nRowsJointDist = self.jointDist.shape[0]

        for rowIdx in range(nRowsJointDist):
            print("hello")
            # next: create a data structure to record newMarginals as you created findings
            # This is done in the class node.

    def MakeBN(self):
        # Assumption: The GUI will ensure that the parents of a node exist.
        # 1. Check the Conditional Probability Tables (CPTs). Ensure conditional probabilities add up to one
        #    If not, bring a text box saying something like: Probabilities in the CPT of node 'node name' do not add up to one.

        # Calculates the initial position of the nodes; later the user can modified them in the GUI
        self.CalcChildren()
        self.CalcYPos()
        self.CalcXPos()

        # 2. Once checks are done, then calculate the joint distribution.
        # self.CalcJointDist() # writes the joint in self.jointDist, Note: It will take forever

    def PlotMarginals(self, with_findings=False, finding_node_color="red", finding_node_penwidth=3.0):
        """
        Plots the marginal distributions for each node in the Bayesian Network.

        Parameters:
        with_findings (bool): If True, use marginals with findings. If False, use original marginals.
        finding_node_color (str): Color of the border for nodes with findings
        finding_node_penwidth (float): Width of the border for nodes with findings
        """
        print(f"PlotMarginals called with with_findings={with_findings}")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Determine which marginals to use
        if with_findings:
            marginals_to_use = self.margWithFindings
            title_suffix = "with Findings"
            findings_text = "Findings: " + ", ".join(
                [f"{node_name}={node.finding[node_name]}"
                 for node_name, node in self.nodes.items()
                 if node.finding and len(node.finding) > 0]
            )
        else:
            marginals_to_use = self.marginals
            title_suffix = ""
            findings_text = ""

        # Draw arrows between nodes first (in background)
        for node_name, node in self.nodes.items():
            for parent in node.parents:
                parent_node = self.nodes[parent]
                
                # Calculate edge points
                dx = node.xPos - parent_node.xPos
                dy = node.yPos - parent_node.yPos
                angle = np.arctan2(dy, dx)
                
                # Calculate start point (on parent ellipse)
                start_x = parent_node.xPos + (parent_node.width/2) * np.cos(angle)
                start_y = parent_node.yPos + (parent_node.height/2) * np.sin(angle)
                
                # Calculate end point (on child ellipse)
                end_x = node.xPos - (node.width/2) * np.cos(angle)
                end_y = node.yPos - (node.height/2) * np.sin(angle)
                
                # Draw arrow
                arrow = patches.FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    connectionstyle="arc3,rad=0",
                    arrowstyle="-|>",
                    mutation_scale=15,
                    color='gray',
                    zorder=1
                )
                ax.add_patch(arrow)

        # Now draw each node (ellipse and text)
        for node_name, node in self.nodes.items():
            x = node.xPos
            y = node.yPos

            has_finding = with_findings and node.finding and len(node.finding) > 0
            if has_finding:
                linewidth = finding_node_penwidth
                edgecolor = finding_node_color
            else:
                linewidth = node.linewidth
                edgecolor = node.edgecolor

            # Create ellipse with node's visual properties
            ellipse = patches.Ellipse(
                (x, y),
                node.width,
                node.height,
                edgecolor=edgecolor,
                facecolor=node.facecolor,
                linewidth=linewidth,
                alpha=0.7,
                zorder=2
            )
            ax.add_patch(ellipse)

            # Put node label above the ellipse
            ax.text(
                x,
                y + node.height / 2 + 0.05,
                node.label,
                fontsize=node.fontsize,
                color=node.textcolor,
                ha=node.ha,
                va="top",
                font=node.font,
                zorder=3
            )

            # Fetch the node's marginal from marginals_to_use
            marginal_df = marginals_to_use[node_name]
            # Ensure 'Prob' is float
            marginal_df["Prob"] = marginal_df["Prob"].astype(float)

            # Format marginal probabilities
            marg_text = []
            for _, row in marginal_df.iterrows():
                value = row.iloc[0]  # First column contains the value
                prob = row["Prob"] * 100  # Convert to percentage
                marg_text.append(f"{value}: {prob:.2f}%")
            marg_text = "\n".join(marg_text)

            # Place marginal probabilities inside the ellipse
            ax.text(
                x,
                y,
                marg_text,
                ha=node.ha,
                va=node.va,
                fontsize=node.fontsize,
                color=node.textcolor,
                font=node.font,
                zorder=3
            )

            # If there's a finding, indicate with an 'f' in red
            if has_finding:
                ax.text(
                    x + 0.8 * node.width / 2,
                    y + 0.8 * node.height / 2,
                    "f",
                    fontsize=node.fontsize + 2,
                    color="red",
                    ha="left",
                    va="top",
                    fontweight="bold",
                    zorder=4
                )

        # Set plot bounds
        all_x = [node.xPos for node in self.nodes.values()]
        all_y = [node.yPos for node in self.nodes.values()]
        margin = 0.5
        
        if all_x and all_y:  # Check if lists are not empty
            plt.xlim(min(all_x) - margin, max(all_x) + margin)
            plt.ylim(max(all_y) + margin, min(all_y) - margin)  # Invert Y-axis
        
        # Show axis with grid
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        plt.axis('on')  # Show the axis

        # Adjust the title
        if with_findings:
            plt.title(f"Marginal Probabilities {title_suffix}\n{findings_text}", pad=20, fontsize=12, fontweight='bold')
        else:
            plt.title(f"Marginal Probabilities {title_suffix}", pad=20, fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def SetFinding(self, node_name, value):
        if node_name in self.nodes:
            self.nodes[node_name].finding = {node_name: value}
        else:
            print(f"Node {node_name} not found")

    def CheckCpts(self):
        """
        Checks if conditional probabilities in each node's CPT sum to 1 for each combination of parent values.
        Returns a list of node labels where probabilities don't sum to 1 (with tolerance of 1e-10).
        """
        problematic_nodes = []
        tolerance = 1e-10  # numerical tolerance for floating point comparison

        for node in self.nodes:
            if not node.parents:
                # For root nodes, simply sum all probabilities
                total_prob = node.cpt["Prob"].sum()
                if abs(total_prob - 1.0) > tolerance:
                    problematic_nodes.append(node.label)
            else:
                # Group by parent values and check sum for each combination
                parent_columns = node.parents
                grouped = node.cpt.groupby(parent_columns)["Prob"].sum()

                # Check if any group's sum deviates from 1.0
                if not all(abs(sum - 1.0) < tolerance for sum in grouped.values):
                    problematic_nodes.append(node.label)

        return problematic_nodes

    def CheckNode(self):
        for node in self.nodes:
            print(node.cpt)
            print(node.cpt["Prob"].sum())
        return "done"

    def GetCpt(self, node_name):
        """
        Get the CPT for a given node

        Args:
            node_name: Name of the node whose CPT to retrieve

        Returns:
            pandas.DataFrame: The CPT for the specified node
        """
        if node_name in self.nodes:
            return self.nodes[node_name].cpt
        else:
            raise ValueError(f"Node {node_name} not found in network")

    def SetFindings(self, findings: dict):
        """Set multiple findings at once"""
        has_findings = False
        if findings:
            for node_name, value in findings.items():
                if value and node_name in self.nodes:  # Only set finding if value is not empty and node exists
                    self.nodes[node_name].finding = {node_name: str(value)}
                    has_findings = True
                    print(f"Set finding for {node_name}: {value}")  # Debug log
        return has_findings


# Notice that this is not a function of the class BayesNe
def LoadBN(bnName):
    with open(bnName + ".pkl", "rb") as file:
        bn = pickle.load(file)
    return bn


if __name__ == "__main__":
    smokingBN = LoadBN("SmokingBN")
    smokingBN.CalcJointDist()
    # smokingBN.CalcMarginals()

    # Kanban board for RPS
    # 1. Node names can only consist of one letter. Otherwise, they generate an error.
