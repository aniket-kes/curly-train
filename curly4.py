import pandas as pd
import numpy as np
from graphviz import Digraph
from IPython.display import Image

# Updated Dataset
data = {
    'Age': ['Youth', 'Adult', 'Senior', 'Youth', 'Senior', 'Adult', 'Youth', 'Youth', 'Senior', 'Senior', 'Adult', 'Youth', 'Adult', 'Senior'],
    'Salary Band': ['High', 'Medium', 'High', 'Low', 'Low', 'High', 'Medium', 'High', 'Medium', 'Low', 'Low', 'High', 'Medium', 'High'],
    'Household Size': ['Single', 'Single', 'Couple', 'Family', 'Single', 'Couple', 'Couple', 'Family', 'Couple', 'Family', 'Couple', 'Single', 'Family', 'Single'],
    'Loan History': ['Approved', 'Rejected', 'Approved', 'Rejected', 'Approved', 'Approved', 'Approved', 'Rejected', 'Rejected', 'Rejected', 'Approved', 'Approved', 'Rejected', 'Approved'],
    'Bought Car': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Function to calculate information gain
def info_gain(data, split_attribute_name, target_name="Bought Car"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Function to build the decision tree
def id3(data, original_data, features, target_attribute_name="Bought Car", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node_class

    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return tree

# Driver code
features = ['Age', 'Salary Band', 'Household Size', 'Loan History']
tree = id3(df, df, features)
print("Decision Tree: ", tree)

# Example classification function
def classify(query, tree, default='Yes'):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result, dict):
                return classify(query, result)
            else:
                return result

query = {'Age': 'Youth', 'Salary Band': 'Medium', 'Household Size': 'Single', 'Loan History': 'Approved'}
print("Classification result: ", classify(query, tree))

# Function to visualize the decision tree using Graphviz
def visualize_tree(tree, graph=None, parent_node=None, parent_edge_label=None):
    if graph is None:
        graph = Digraph()
        graph.attr('node', shape='ellipse')

    for node, branches in tree.items():
        if parent_node is None:
            # Root node
            graph.node(node)
            parent_node = node
        for value, subtree in branches.items():
            child_node = f'{node}_{value}'
            graph.node(child_node, label=value)
            graph.edge(parent_node, child_node, label=value)

            # If the subtree is a dictionary (more branches), call recursively
            if isinstance(subtree, dict):
                visualize_tree({node: subtree}, graph, child_node, value)
            else:
                # Leaf node
                leaf_node = f'{child_node}_leaf'
                graph.node(leaf_node, label=subtree, shape='box')
                graph.edge(child_node, leaf_node)

    return graph

# Visualizing the decision tree
tree_graph = visualize_tree(tree)
tree_graph.render("decision_tree_new_dataset", format="png", cleanup=False)

# Display the decision tree in Jupyter (or other IDEs that support image display)
Image(filename='decision_tree_new_dataset.png')
