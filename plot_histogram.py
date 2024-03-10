import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# Your data loading and plotting logic remains the same
# Make sure to replace 'train.csv' with the correct path to your dataset
df = pd.read_csv('train.csv')
num_bins, bin_edges = freedman_diaconis_bins(df['C3'])

print("num bins = ", num_bins, "number of edges = ", bin_edges)
# Given min and max values
min_value = 0.251
max_value = 31940.0

# # Define the number of bins
# num_bins = 50

# # Create bin edges for equally spaced bins based on the min and max values
# bin_edges = np.linspace(min_value, max_value, num_bins + 1)

# Create labels for the bins based on their edges
bin_labels = [f"{bin_edges[i]} - {bin_edges[i+1]}" for i in range(len(bin_edges)-1)]

# For demonstration, let's create a sample DataFrame with 'TransactionAmt' within the given range
np.random.seed(42)  # For reproducibility


# Bin the data
df['C3Bin'] = pd.cut(df['C3'], bins=bin_edges, labels=bin_labels, include_lowest=True)

# Plotting
plt.figure(figsize=(150, 8))
df['C3Bin'].value_counts(sort=False).plot(kind='bar')
plt.title('card1 Distribution Across 10 Bins')
plt.xlabel('C3 Bins (Lower Limit - Upper Limit)')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()




# Example usage:
# Assuming `df['your_column_name']` is your DataFrame Series.
# num_bins, bin_edges = freedman_diaconis_bins(df['your_column_name'])

# Print the number of bins and the bin edges
# print("Number of bins:", num_bins)
# print("Bin edges:", bin_edges)


