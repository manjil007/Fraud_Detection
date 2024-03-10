import numpy as np
from scipy import stats
from Decision_Tree import Tree
import pandas as pd


class RandomForest:
    def __init__(self, num_trees=10, max_depth=10, alpha=None, tool=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []
        self.alpha = alpha
        self.tool = tool

    def fit(self, df_train, max_depth):
        """
            Fits the ensemble of decision trees (forest) on the training data. This method
            initializes each tree with a subset of the training data (bootstrapped sample) and
            fits the tree to that sample up to a specified maximum depth.

            Parameters:
            - df_train (DataFrame): The training dataset on which the forest is trained.
            - depth (int): The starting depth of the trees (unused in this snippet, could be for future use or an oversight).
            - max_depth (int): The maximum depth that any tree in the forest can grow to.

            Note: The method currently does not use the 'depth' parameter. It might be intended for future enhancements
            or to maintain consistency in method signatures across different models.
        """
        for i in range(self.num_trees):
            sample = self._bootstrap_samples(df_train)
            tree = Tree(max_depth=max_depth)
            tree.tool = self.tool
            tree.alpha = self.alpha
            tree.fit(sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, dataset):
        """
            Generates a bootstrapped sample from the dataset. Bootstrapping is a method that allows
            for the estimation of a distribution of statistics by resampling with replacement. This
            function creates a sample dataset that is 20% the size of the original dataset by randomly
            selecting instances with replacement.

            Parameters:
            - dataset (DataFrame): The original dataset from which to draw the bootstrapped sample.

            Returns:
            - DataFrame: A bootstrapped sample of the original dataset.
        """
        num_samples = dataset.shape[0]
        sample_size = int(num_samples * 0.2)  # Calculate 20% of the original dataset size
        index = np.random.choice(num_samples, sample_size, replace=True)  # Use sample_size instead of num_samples
        sample_df = dataset.iloc[index]
        return sample_df

    def predict(self, test_dataset):
        transaction_ids = test_dataset['TransactionID']
        # Initialize an empty list to store the predictions for all instances
        all_predictions = []

        # Iterate through each instance in the test dataset
        for index, instance in test_dataset.iterrows():
            instance_predictions = []
            # Collect predictions from each tree for the current instance
            for tree in self.trees:
                instance_predictions.append(tree.predict(instance.drop('TransactionID')))
            # instance_predictions = [tree.predict(instance.drop('TransactionID')) for tree in self.trees]

            # Aggregate the predictions for the current instance to get a final prediction
            # For example, by majority vote
            final_prediction = self._get_node_label(instance_predictions)

            # Append the final prediction for the current instance to the list of all predictions
            all_predictions.append(final_prediction)

        # Create a DataFrame from the transaction IDs and predictions
        results_df = pd.DataFrame({
            'TransactionID': transaction_ids,
            'Prediction': all_predictions
        })

        # Specify the filename for the CSV file
        filename = 'predictions_{}_alpha{:.2f}.csv'.format(self.tool, self.alpha)


        # Export the DataFrame to a CSV file
        results_df.to_csv(filename, index=False)

        print(f"Predictions have been saved to {filename}.")
        return filename

    def _get_node_label(self, tree_predictions):
        """
        Determine the final label for an instance based on predictions from all trees.

        Parameters:
        - tree_predictions: An array of predictions for a single instance from each tree.

        Returns:
        - The final predicted label based on majority voting.
        """
        # Validate input
        if not tree_predictions:
            raise ValueError("tree_predictions is empty")

        # Calculate mode
        mode_result = stats.mode(tree_predictions)

        # Validate mode result
        if mode_result.mode.size > 0:
            mod = mode_result.mode
            return mod
        else:
            raise ValueError("Invalid mode result for tree_predictions: {}".format(tree_predictions))


    def calculate_accuracy(self, original_labels, predictions):
        """
        Calculate the accuracy of the testdata predictions.

        Parameters:
        - original_labels: A numpy array or a list containing the true labels.
        - predictions: A numpy array or a list containing the predicted labels.

        Returns:
        - Accuracy: The percentage of correct predictions.
        """
        # Ensure that original_labels and predictions are numpy arrays for element-wise comparison
        original_labels = np.array(original_labels)
        predictions = np.array(predictions)

        # Calculate the number of correct predictions
        correct_predictions = np.sum(original_labels == predictions)

        # Calculate accuracy as the ratio of correct predictions to the total number of predictions
        accuracy = correct_predictions / len(original_labels)

        return accuracy

