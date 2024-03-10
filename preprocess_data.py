import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.utils import resample

# Script for preprocessing and balancing a dataset for fraud detection analysis.
# This includes handling missing values, converting categorical variables to numerical,
# binning continuous variables, and balancing the dataset based on the 'isFraud' column.


def check_notfound_columns(df):
    """
    Identifies columns in a DataFrame that contain the value 'NotFound'.
    This function applies a lambda function across all columns to check for the presence
    of 'NotFound' in their values, returning a list of such columns.

    Parameters:
    - df (DataFrame): The DataFrame to be checked for 'NotFound' values.

    Returns:
    - list: A list of column names where 'NotFound' is present in the values.
    """
    columns_with_notfound = df.apply(lambda col: 'NotFound' in col.values)
    return columns_with_notfound[columns_with_notfound].index.tolist()


def replace_notfound_with_median_or_reptitive_values(df, columns):
    """
    Replaces 'NotFound' values in specified columns with either the median (for numeric columns)
    or the most frequent value (for non-numeric columns). Numeric columns with 'NotFound' are first
    converted to NaN, then the median of the column replaces these NaN values. For non-numeric columns,
    'NotFound' is replaced by the most common value excluding 'NotFound'.

    Parameters:
    - df (DataFrame): The DataFrame where replacements should be made.
    - columns (list): A list of column names to check for 'NotFound' and replace.

    Returns:
    - DataFrame: The modified DataFrame after replacements.
    """
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column].replace('NotFound', np.nan), errors='coerce')
            df[column].fillna(df[column].median(), inplace=True)
        else:
            most_frequent_value = df[column].replace('NotFound', np.nan).mode().iloc[0]
            df[column] = df[column].replace('NotFound', most_frequent_value)
    return df


def convert_cateogorical_to_numerical(df):
    """
    Converts categorical columns in the DataFrame to numerical using label encoding,
    but only for those columns that are of object type and have fewer than 10 unique values.
    This transformation is useful for preparing data for machine learning models that
    require numerical input.

    Parameters:
    - df (DataFrame): The DataFrame containing the categorical data to be converted.

    Returns:
    - DataFrame: The DataFrame with categorical columns converted to numerical where applicable.
    """
    encoder = preprocessing.LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object' and len(df[column].unique()) < 10:
            df[column] = encoder.fit_transform(df[column])

    return df


def oversample_minority_to_specific_ratio(df, target_column, desired_ratio=0.5):
    """
    Oversamples the minority class in a dataframe to achieve a specific ratio of minority class.

    Parameters:
    - df: Pandas DataFrame containing the dataset.
    - target_column: The name of the column that contains the target variable.
    - desired_ratio: The desired ratio of minority class after oversampling (default is 0.3).

    Returns:
    - DataFrame after oversampling the minority class to achieve the desired ratio.
    """

    # Count the current number of instances in each class
    n_minority = len(df[df[target_column] == 1])
    n_majority = len(df[df[target_column] == 0])

    # Calculate the total number of samples needed in minority class to achieve the desired ratio
    total_samples_needed = n_majority / (1 - desired_ratio)
    minority_samples_needed = total_samples_needed * desired_ratio

    # Calculate how many minority samples to add
    additional_minority_samples = int(minority_samples_needed - n_minority)

    if additional_minority_samples > 0:
        # Oversample minority class to achieve the desired ratio
        df_minority_oversampled = resample(df[df[target_column] == 1],
                                           replace=True,  # sample with replacement
                                           n_samples=additional_minority_samples,  # number of samples to generate
                                           random_state=123)  # reproducible results

        # Combine the original dataset with the oversampled minority class
        df_balanced = pd.concat([df, df_minority_oversampled])
    else:
        # If no additional samples are needed, return the original dataframe
        df_balanced = df

    return df_balanced


def balance_data(training_data):
    """
    Splits the training data into two separate files based on the 'isFraud' column value.
    Rows with 'isFraud' equal to 1 are written to 'train_isfraud.csv', and rows with 'isFraud'
    equal to 0 are written to 'train_isnot_fraud.csv'. This function also formats each row
    before writing, ensuring numerical values are appropriately converted.

    Parameters:
    - training_data (DataFrame): The training dataset to be split based on the 'isFraud' value.
    """
    f_train_is_fraud = open("train_isfraud.csv", 'w')
    f_train_is_not_fraud = open("train_isnot_fraud.csv", 'w')
    for index, row in training_data.iterrows():
        # Initialize an empty list to hold formatted row values
        formatted_row_data = []
        for item in row.values:
            # Format item based on its type
            if isinstance(item, float) and item.is_integer():
                formatted_item = str(int(item))  # Convert to int first if it's a whole number
            else:
                formatted_item = str(item)
            formatted_row_data.append(formatted_item)

        # Join the formatted row values into a string
        row_str = ",".join(formatted_row_data)

        # Write the row to the appropriate file based on the 'isFraud' value
        if row['isFraud'] == 0:
            f_train_is_not_fraud.write(f"{row_str}\n")
        elif row['isFraud'] == 1:
            f_train_is_fraud.write(f"{row_str}\n")

    f_train_is_fraud.close()
    f_train_is_not_fraud.close()


def freedman_diaconis_bins(series):
    """
    Calculate the number of bins and bin edges using the Freedman-Diaconis rule.

    Parameters:
    - series: Pandas Series containing the data.

    Returns:
    - num_bins: The recommended number of bins.
    - bin_edges: The edges of the bins.
    """
    # Calculate IQR
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # Avoid division by zero if IQR is 0 by setting a default number of bins
    if IQR == 0:
        # Fall back to a default bin width or a heuristic like the square root choice
        n = len(series)
        num_bins = int(np.sqrt(n))  # A simple heuristic, adjust as necessary
    else:
        # Proceed with calculating bin width
        n = len(series)
        bin_width = 2 * IQR / (n ** (1 / 3))

        # Calculate number of bins
        data_range = series.max() - series.min()
        num_bins = max(1, int(np.ceil(data_range / bin_width)))  # Ensure at least 1 bin

    # Calculate bin edges
    bin_edges = np.linspace(series.min(), series.max(), num_bins + 1)

    return num_bins, bin_edges


def bin_column_and_rearrange(train_data, test_data, old_column, new_column_name, target_column, num_bins):
    """
    Bins a specified column in both training and testing datasets into a specified number of bins
    based on percentile ranges, then replaces the original column with the binned version. This
    function is useful for converting continuous variables into categorical ones, potentially
    improving model performance by handling outliers and non-linear relationships. The target column
    is temporarily removed and then re-added to ensure it's at the end of the DataFrame.

    Parameters:
    - train_data (DataFrame): The training dataset.
    - test_data (DataFrame): The testing dataset.
    - old_column (str): The name of the column to be binned and replaced.
    - new_column_name (str): The name for the new binned column.
    - target_column (str): The name of the target column, which will be temporarily removed and then re-added.
    - num_bins (int): The number of bins to divide the column into.

    Returns:
    - tuple: A tuple containing the modified training and testing DataFrames.
    """
    combined_data = pd.concat([train_data[old_column], test_data[old_column]])

    # Determine bin edges based on percentiles to ensure equal distribution
    num_bins, bin_edges = freedman_diaconis_bins(combined_data)

    # Remove duplicate edges
    bin_edges = np.unique(bin_edges)
    y_train = None
    y_test = None
    # Ensure that target_column is temporarily removed to avoid including it in the middle of the DataFrame
    if target_column in train_data.columns:
        y_train = train_data.pop(target_column)
    if target_column in test_data.columns:
        y_test = test_data.pop(target_column)

    # Bin the old_column in both datasets
    train_data[new_column_name] = pd.cut(train_data[old_column], bins=bin_edges, include_lowest=True, labels=False, duplicates='drop')
    test_data[new_column_name] = pd.cut(test_data[old_column], bins=bin_edges, include_lowest=True, labels=False, duplicates='drop')

    # Remove the original old_column
    train_data.drop(old_column, axis=1, inplace=True)
    test_data.drop(old_column, axis=1, inplace=True)

    # Re-add target_column to the end of both datasets
    if y_train is not None:
        train_data[target_column] = y_train
    if y_test is not None:
        test_data[target_column] = y_test

    return train_data, test_data


# Load training and testing data from CSV files.
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Identify columns with 'NotFound' values in both training and testing datasets.
not_found_columns_train = check_notfound_columns(df_train)
not_found_columns_test = check_notfound_columns(df_test)

# Replace 'NotFound' values with either median or most frequent values in the identified columns.
df_train = replace_notfound_with_median_or_reptitive_values(df_train, not_found_columns_train)
df_test = replace_notfound_with_median_or_reptitive_values(df_test, not_found_columns_test)

# Convert categorical variables with fewer than 10 unique values to numerical format using label encoding.
df_train = convert_cateogorical_to_numerical(df_train)
df_test = convert_cateogorical_to_numerical(df_test)

# Bin the 'TransactionAmt', 'card1', and 'C3' columns into specified number of bins and rearrange the datasets.
# This helps in transforming continuous variables into categorical ones, which can be beneficial for certain models.
transaction_amt_num_bins_dersired = 5

df_train, df_test = bin_column_and_rearrange(df_train, df_test, 'TransactionAmt', 'TransactionAmtBin', 'isFraud', 1000)
df_train, df_test = bin_column_and_rearrange(df_train, df_test, 'card1', 'card1Bin', 'isFraud', 83)
#df_train, df_test = bin_column_and_rearrange(df_train, df_test, 'card2', 'card2Bin', 'isFraud', 11)
df_train, df_test = bin_column_and_rearrange(df_train, df_test, 'C3', 'C3Bin', 'isFraud', 687)

# Retrieve all attribute names from the training dataset for potential future use.
train_attributes = df_train.columns

# Balance the training data by splitting it into two separate files based on the 'isFraud' value.
# This step is crucial for dealing with imbalanced datasets commonly found in fraud detection tasks.
balance_data(df_train)

df_test.to_csv('test_clean.csv', index=False)




