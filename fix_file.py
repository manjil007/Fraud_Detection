import pandas as pd

df = pd.read_csv('predictions.csv')

# Remove square brackets and convert to int
df['Prediction'] = df['Prediction'].apply(lambda x: int(x.strip('[]')))

# Construct the new filename with tool and alpha values
new_filename = f'cleaned_predictions.csv'

# Save the cleaned DataFrame to a new CSV file
df.to_csv(new_filename, index=False)