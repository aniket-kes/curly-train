import pandas as pd
import numpy as np

# New Dataset
data = {
    'Age Group': ['Young', 'Middle', 'Senior', 'Young', 'Senior', 'Middle', 'Young', 'Young', 'Senior', 'Senior', 'Middle', 'Young', 'Middle', 'Senior'],
    'Income Level': ['High', 'Medium', 'High', 'Low', 'Low', 'High', 'Medium', 'High', 'Medium', 'Low', 'Low', 'High', 'Medium', 'High'],
    'Family Size': ['Small', 'Small', 'Medium', 'Large', 'Small', 'Medium', 'Medium', 'Large', 'Medium', 'Large', 'Medium', 'Small', 'Large', 'Small'],
    'Credit Score': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Good', 'Good', 'Bad', 'Bad', 'Bad', 'Good', 'Good', 'Bad', 'Good'],
    'Purchased Car': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Function to calculate information gain
def info_gain(data, split_attribute_name, target_name="Purchased Car"):
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
   
    # Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
   
    # Calculate the weighted entropy
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
   
    # Calculate the information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Function to build the decision tree
def id3(data, original_data, features, target_attribute_name="Purchased Car", parent_node_class=None):
    # If all target values have the same value, return that value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
   
    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
   
    # If the feature space is empty, return the parent node target feature value
    elif len(features) == 0:
        return parent_node_class
   
    # If none of the above conditions are met, grow the tree
    else:
        # Set the default value for this node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
       
        # Select the feature which best splits the dataset
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
       
        # Create the tree structure
        tree = {best_feature: {}}
       
        # Remove the feature with the best information gain from the feature space
        features = [i for i in features if i != best_feature]
       
        # Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
           
            # Call the ID3 algorithm for each branch
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
           
            # Add the subtree
            tree[best_feature][value] = subtree
       
        return tree

# Driver code
features = ['Age Group', 'Income Level', 'Family Size', 'Credit Score']
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

query = {'Age Group': 'Young', 'Income Level': 'Medium', 'Family Size': 'Small', 'Credit Score': 'Good'}
print("Classification result: ", classify(query, tree))
