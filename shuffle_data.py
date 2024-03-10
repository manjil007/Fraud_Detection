import pandas as pd
from sklearn.utils import shuffle
from preprocess_data import train_attributes
df = pd.read_csv('balancedTrain.csv')

# Assign the list of column names to the DataFrame's columns
df.columns = train_attributes

df_shuffled = shuffle(df)

df_shuffled.to_csv('balancedTrain.shuffled.csv', index=False)

