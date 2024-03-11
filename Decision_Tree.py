import numpy as np
import pandas as pd
from scipy.stats import chi2
import random


class Node:

    def __init__(self, feature=None, value=None, label=None):
        self.feature = feature
        self.successors = []
        self.value = value
        self.label = label

    def is_leaf_node(self):
        bol = len(self.successors) == 0
        return bol

    def add_successor(self, successor_node):
        """Add a successor to the list of successors."""
        if isinstance(successor_node, Node):
            self.successors.append(successor_node)
        else:
            raise ValueError("Successor must be an instance of Node")


class Tree:
    def __init__(self, max_depth=None, alpha=None, tool=None):
        self.max_depth = max_depth
        self.root = None
        self.alpha = alpha
        self.tool = tool

    def fit(self, dataset):
        dataset_size = len(dataset)
        train_attributes = dataset.columns
        target_feature = train_attributes[-1]
        train_attributes = train_attributes.drop(target_feature)
        self.root = self._construct_tree(dataset, dataset_size, target_feature, train_attributes, 0)

    def _construct_tree(self, dataset, dataset_size, target_value, train_features, depth):
        """
            Recursively constructs a decision tree based on the provided dataset.

            This method uses information gain and chi-square statistical test to decide on the best feature for node splitting,
            aiming to reach the best decision based on the target value. The construction stops when it hits a base case:
            all target values are the same, the maximum depth is reached, there are no features left, or the chi-square test
            does not indicate a significant difference.

            Parameters:
            - dataset (DataFrame): The subset of data used for constructing the current node and its children.
            - dataset_size (int): The total size of the initial dataset (unused in this snippet but could be used for further enhancements).
            - target_value (str): The name of the target feature in the dataset that the decision is based on.
            - train_features (list of str): The list of feature names that can be used for splitting.
            - depth (int): The current depth of the node in the tree.
            - max_depth (int): The maximum allowed depth of the tree.
            - alpha (float): The significance level used in the chi-square test to evaluate feature splits.

            Returns:
            - Node: The root node of the constructed subtree.
            """

        # Initialize a new node for the decision tree
        t = Node()

        # Base case 1: If all target values in the dataset are the same, return a leaf node with that value
        if all(dataset[target_value] == 1):
            t.label = 1
            return t

        # Base case 2: If the maximum depth has been reached, return a leaf node with the most common target value
        if depth >= self.max_depth:
            t.label = dataset[target_value].mode()[0]  # Assign the most common target value in the current subset
            return t

        # Base case 3: If there are no features left to split on, return a leaf node with the most common target value
        if len(train_features) == 0:
            t.label = dataset['isFraud'].mode()[0]
            return t

        # Find the best attribute to split on based on information gain
        best_attribute = self.find_feature_with_max_info_gain(dataset, train_features, target_value)

        chi_square_stat = self.calculate_chi_square(dataset, best_attribute, target_value)
        critical_value = self.calculate_chi_square_critical_value(dataset, best_attribute, target_value, self.alpha)

        # Base case 4:If the chi-square statistic is below the critical value or if there's no significant
        # difference, return a leaf node
        if chi_square_stat < critical_value or critical_value is None or chi_square_stat == 0:
            t.label = dataset[target_value].mode()[0]
            return t

        t.feature = best_attribute

        # Remove the best attribute from the list of features for further splits
        remaining_features = [feature for feature in train_features if feature != best_attribute]

        # Partition the dataset based on the best attribute and create child nodes
        unique_values = dataset[best_attribute].unique()

        for value in unique_values:
            subset = dataset[dataset[best_attribute] == value]

            if len(subset) == 0:
                # If the subset for this value is empty, create a leaf node with the most common target value
                most_common_value = dataset[target_value].mode()[0]
                child_node = Node(value=value, label=most_common_value)
                t.add_successor(child_node)
            else:
                # Otherwise, recursively construct the subtree for this subset
                child_node = self._construct_tree(subset, dataset_size, target_value, remaining_features, depth + 1)
                # Ensure the child node knows the value that led to it (for decision-making / splitting purposes)
                child_node.value = value
                t.add_successor(child_node)
                
        return t

    def calculate_chi_square(self, dataset, feature, target_value):
        """
            Calculates the chi-square statistic for a given feature against a target value
            in the dataset. This statistic helps to determine how likely it is that any observed
            difference between the sets arose by chance. It's commonly used in feature selection to
            evaluate the independence between a feature and the target variable.

            Parameters:
            - dataset (DataFrame): The dataset containing the feature and target value columns.
            - feature (str): The name of the feature for which the chi-square statistic is calculated.
            - target_value (str): The name of the target variable against which the feature's independence is tested.

            Returns:
            - float: The calculated chi-square statistic for the given feature against the target value.
            """

        chi_square_stat = 0
        unique_values = dataset[feature].unique()
        total_instances = len(dataset)
        class_counts = dataset[target_value].value_counts()

        for value in unique_values:
            subset = dataset[dataset[feature] == value]
            subset_size = len(subset)
            if subset_size == 0:
                continue

            for class_value, count in class_counts.items():
                observed_frequency = len(subset[subset[target_value] == class_value])
                expected_frequency = count * subset_size / total_instances
                chi_square_stat += (
                                               observed_frequency - expected_frequency) ** 2 / expected_frequency if expected_frequency > 0 else 0

        return chi_square_stat

    def calculate_chi_square_critical_value(self, dataset, best_attribute, target_value, alpha):
        unique_values = dataset[best_attribute].unique()
        num_classes = dataset[target_value].nunique()
        degrees_of_freedom = (len(unique_values) - 1) * (num_classes - 1)
        alpha = float(alpha)
        # Calculate the critical value from the Chi-Square distribution
        critical_value = chi2.ppf(alpha, degrees_of_freedom)

        return critical_value

    def find_feature_with_max_info_gain(self, dataset, train_features, target_feature):
        """
        Finds the feature with the maximum information gain.

        Parameters:
        - dataset: A pandas DataFrame containing the data.
        - features: A list of feature column names to consider for splitting.
        - target_feature: The name of the target feature column.
        - calculate_information_gain_function: A function that calculates information gain for a given feature and target.

        Returns:
        - The name of the feature with the maximum information gain and the corresponding information gain value.
        """
        max_info_gain = -float('inf')
        best_feature = None
        for feature in train_features:
            if feature not in train_features:
                continue
            info_gain = self.information_gain(dataset, target_feature, feature)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature

        return best_feature

    def calculate_entropy(self, dataset, target_label):
        bin_count = np.bincount(dataset[target_label])
        prop = bin_count / len(dataset)
        entropy = -np.sum([p * np.log2(p) for p in prop if p > 0])
        return entropy

    def calculate_gini_index(self, dataset, target_label):
        bin_count = np.bincount(dataset[target_label])
        prop = bin_count / len(dataset)
        gini_index = 1 - np.sum([p**2 for p in prop])
        return gini_index

    def calculate_misclassification_error(self, dataset, target_label):
        bin_count = np.bincount(dataset[target_label])
        prop = bin_count / len(dataset)
        misclassification_error = 1 - np.max(prop)
        return misclassification_error

    def information_gain(self, dataset, target, feature):
        # Determine which metric to use based on self.randomization_tool
        if self.tool == 'entropy':
            metric_function = self.calculate_entropy
        elif self.tool == 'gini':
            metric_function = self.calculate_gini_index
        elif self.tool == 'mis':
            metric_function = self.calculate_misclassification_error
        else:
            raise ValueError("Invalid randomization tool specified.")

        # Calculate the metric before the split
        total_metric = metric_function(dataset, target)

        # Calculate the weighted average metric after the split
        values, counts = np.unique(dataset[feature], return_counts=True)
        weighted_metric = 0
        for value, count in zip(values, counts):
            subset = dataset[dataset[feature] == value]
            subset_metric = metric_function(subset, target)
            weighted_metric += (count / len(dataset)) * subset_metric

        # Calculate the information gain
        info_gain = total_metric - weighted_metric

        return info_gain

    def predict(self, instance):
        """
        Predict the class label for a given instance.

        Parameters:
        - instance: A dictionary or pandas Series with features as keys and feature values as values.

        Returns:
        - The predicted class label.
        """

        def traverse(node, instance):
            # If the node is a leaf node, return its label as the prediction
            if node.is_leaf_node():
                return node.label  # Use label for prediction in leaf

            # Retrieve the instance's value for the feature at this node
            feature_value = instance.get(node.feature)

            # Traverse to the next node based on the instance's feature value
            for successor in node.successors:
                if feature_value == successor.value:  # Match the successor's value with the instance's feature value
                    return traverse(successor, instance)  # Recursively traverse

            # If no matching successor is found, return None or a default prediction
            return random.randint(0, 1)

        # Start the traversal from the root of the tree
        return traverse(self.root, instance)

    def print_tree(self):
        if not self.root:
            print("The tree is empty.")
            return

        # Stack to hold nodes to visit, alongside their depth
        stack = [(self.root, 0)]

        while stack:
            current_node, depth = stack.pop()  # Pop the last element to ensure depth-first
            indent = " " * depth * 2  # Increase space with depth

            # Print current node's feature or value
            if current_node.is_leaf_node():
                print(f"{indent}Leaf value: {current_node.value}")
                print(f"{indent}Leaf Label: {current_node.label}")
            else:
                print(f"{indent}Leaf value: {current_node.value}")
                print(f"{indent}Node feature: {current_node.feature}")

            # Reverse the successors to ensure when we pop from the stack, we process the first child last
            # This is important for depth-first search to visit the left-most child first
            for successor in reversed(current_node.successors):
                stack.append((successor, depth + 1))

