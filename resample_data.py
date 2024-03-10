#google docs link to find fraud, not fraud rato
#https://docs.google.com/spreadsheets/d/1afoNBKsHJ6Oxsby1TdqzqAKUV0eZlL0x_MQi51yRQzE/edit#gid=1557939618

import pandas as pd

# Script to oversample the 'isFraud' cases in a dataset to help balance the class distribution.
# It reads the dataset containing fraud cases, formats each row, and writes each row multiple times to a new file
# to increase the representation of fraud cases. This approach can help in situations where the original dataset
# has a significantly imbalanced class distribution, potentially improving model performance on minority classes.

# Load the dataset containing only the fraud cases
is_fraud_df = pd.read_csv('train_isfraud.csv')
# Define the name of the new file that will contain the oversampled fraud cases
f_name = 'is_fraud_overSampled.csv'
# Open the new file in write mode
f_is_fraud_overSampled = open(f_name, 'w')

number_rows = len(is_fraud_df)

# Iterate through each row in the dataset
for index, row in is_fraud_df.iterrows():
    # Initialize an empty list to hold formatted row values
    formatted_row_data = []
    for item in row.values:
        # Determine the type and format accordingly
        if isinstance(item, float) and item.is_integer():
            # Item is a float but represents an integer, convert to int then to string
            formatted_item = str(int(item))
        else:
            # For floats that are not integers, and other types, convert directly to string
            formatted_item = str(item)
        formatted_row_data.append(formatted_item)

    # Join the formatted row values into a string
    row_str = ",".join(formatted_row_data)

    # Write the formatted string to the file multiple times as per your requirement
    for i in range(27):
        f_is_fraud_overSampled.write(f"{row_str}\n")


print(f"Status: {f_name} saved sucessfully")
f_is_fraud_overSampled.close()


