from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np


def create_pgmpy_model(bayesnet):
    """Helper function to create a PGMPY model from our BayesNet"""
    # Create edges
    edges = []
    for node_name in bayesnet.nodes:
        node = bayesnet.nodes[node_name]
        for parent in node.parents:
            edges.append((parent, node_name))

    # Create PGMPY model
    pgmpy_model = BayesianNetwork(edges)

    # Add CPDs
    for node_name in bayesnet.nodes:
        node = bayesnet.nodes[node_name]
        cpt = bayesnet.GetCpt(node_name)
        values = cpt["Prob"].values

        print(f"\nNode: {node_name}")
        print("Original CPT:")
        print(cpt)

        # Get number of possible values for this node
        n_values = len(node.values)

        # Reshape probabilities
        if not node.parents:
            # For root nodes, shape should be (n_values, 1)
            values = values.reshape(n_values, 1)
        else:
            # For child nodes, reshape considering parent combinations
            num_parents = len(node.parents)
            # Reshape to have child states as rows and parent combinations as columns
            values = values.reshape(-1, n_values).T

        print(f"Values after reshape for {node_name}:", values)
        print(f"Values shape: {values.shape}")

        evidence = node.parents
        evidence_card = [len(bayesnet.nodes[parent].values) for parent in evidence]

        # Create CPD with explicit state names
        cpd = TabularCPD(
            variable=node_name,
            variable_card=n_values,
            values=values,
            evidence=evidence if evidence else None,
            evidence_card=evidence_card if evidence else None,
            state_names={
                node_name: node.values,
                **{parent: bayesnet.nodes[parent].values for parent in evidence},
            },
        )

        print(f"CPD for {node_name}:")
        print(cpd)

        # Verify CPD is valid before adding
        if cpd.is_valid_cpd():
            pgmpy_model.add_cpds(cpd)
        else:
            raise ValueError(f"Invalid CPD created for node {node_name}")

    # Verify the entire model
    if not pgmpy_model.check_model():
        raise ValueError("Invalid PGMPY model created")

    return pgmpy_model


def get_marginals(pgmpy_model, evidence=None):
    """Helper function to calculate marginals using PGMPY"""
    inference = VariableElimination(pgmpy_model)
    marginals = []

    print("\nPGMPY Evidence:", evidence)  # Debug print

    for node_name in pgmpy_model.nodes():
        node_values = pgmpy_model.get_cpds(node_name).state_names[node_name]

        # Skip if node is in evidence
        if evidence and node_name in evidence:
            # Create a deterministic distribution for the evidence
            prob_values = [
                1.0 if val == evidence[node_name] else 0.0 for val in node_values
            ]
            print(f"\nPGMPY {node_name} (evidence node):")  # Debug print
            print(f"Values: {prob_values}")  # Debug print
            df_marg = pd.DataFrame({node_name: node_values, "Prob": prob_values})
        else:
            # Calculate marginal normally
            result = inference.query(variables=[node_name], evidence=evidence)

            print(f"\nPGMPY {node_name} (non-evidence node):")  # Debug print
            print(f"Query result: {result.values}")  # Debug print

            df_marg = pd.DataFrame({node_name: node_values, "Prob": result.values})

        marginals.append(df_marg)
        print(f"Final marginal for {node_name}:")  # Debug print
        print(df_marg)  # Debug print

    return marginals


def add_pgmpy_methods(BayesNet):
    """Adds PGMPY-related methods to the BayesNet class"""

    def calculate_pgmpy_marginals(self):
        """Calculate marginals using PGMPY"""
        # First, ensure BayesNet marginals are calculated
        if not self.marginals or len(self.marginals) == 0:
            self.CalcJointDist()
            self.CalcMarginals()

        # Create PGMPY model
        pgmpy_model = create_pgmpy_model(self)

        # Calculate regular marginals (without evidence)
        self.pgmpy_marginals = get_marginals(pgmpy_model)

        # Calculate marginals with findings if any exist
        evidence = {}
        for node_name in self.nodes:
            node = self.nodes[node_name]
            if hasattr(node, "finding") and node.finding:
                evidence.update(node.finding)

        if evidence:
            # Ensure BayesNet marginals with findings are calculated
            if not self.margWithFindings or len(self.margWithFindings) == 0:
                self.CalcMargGivenFindings()
            self.pgmpy_margWithFindings = get_marginals(pgmpy_model, evidence)

    def compare_marginals(self):
        """Compare BayesNet and PGMPY marginals"""
        if not hasattr(self, "pgmpy_marginals"):
            print(
                "Please calculate PGMPY marginals first using calculate_pgmpy_marginals()"
            )
            return

        if not self.marginals or len(self.marginals) == 0:
            print("BayesNet marginals not calculated. Calculating now...")
            self.CalcJointDist()
            self.CalcMarginals()

        print("\nComparing regular marginals:")
        for idx, (node_name, node) in enumerate(self.nodes.items()):
            print(f"\nNode: {node_name}")
            print("BayesNet marginal:")
            print(self.marginals[idx])
            print("PGMPY marginal:")
            print(self.pgmpy_marginals[idx])

        if hasattr(self, "pgmpy_margWithFindings"):
            if not self.margWithFindings or len(self.margWithFindings) == 0:
                print("\nCalculating BayesNet marginals with findings...")
                self.CalcMargGivenFindings()

            print("\nComparing marginals with findings:")
            for idx, (node_name, node) in enumerate(self.nodes.items()):
                print(f"\nNode: {node_name}")
                print("BayesNet marginal with findings:")
                print(self.margWithFindings[idx])
                print("PGMPY marginal with findings:")
                print(self.pgmpy_margWithFindings[idx])

    # Add methods to BayesNet class
    BayesNet.calculate_pgmpy_marginals = calculate_pgmpy_marginals
    BayesNet.compare_marginals = compare_marginals
