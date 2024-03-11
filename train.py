import pandas as pd
from random_forest import RandomForest
import pickle


# Load the balanced and shuffled training dataset. TransactionID and TransactionDT columns are dropped as they are
# not predictive.
df_train = pd.read_csv('balancedTrain.shuffled.csv', index_col=False)
df_train = df_train.drop(['TransactionID', 'TransactionDT'], axis=1)

# Load the preprocessed test dataset. TransactionDT is dropped to match the training dataset's features.
df_test = pd.read_csv('test_clean.csv')
df_test = df_test.drop(['TransactionDT'], axis=1)


randomization_tool = ['entropy', 'mis', 'gini']
alpha_vals = [0.01, 0.05, 0.1, 0.25, 0.5, 0.9]

for tool in randomization_tool:
    for alpha_val in alpha_vals:
        # Initialize a Random Forest classifier.
        r = RandomForest()
        r.alpha = alpha_val
        r.tool = tool
        # Fit the Random Forest model on the training dataset.
        r.fit(df_train, 10)

        # Predict the target variable for the test dataset using the trained model.
        predictions = r.predict(df_test)

        model_filename = f'random_forest_{tool}_{alpha_val}_model.pkl'

        # Save the model to a file with the constructed name
        with open(model_filename, 'wb') as model_file:
            pickle.dump(r, model_file)





