
# Fraud Detection Project

This project is designed to detect fraudulent transactions using a machine learning model based on a RandomForest classifier. The process involves data preprocessing, balancing the dataset, model training, and prediction.

## Getting Started

### Prerequisites

- Python 3.x
- Pandas library
- NumPy library
- Matplotlib library
- Scikit-learn library
- Jupyter Notebook or any Python IDE

### Installation

Clone the repository to your local machine:

```bash
git clone <repository-url>
```

Ensure you have the required libraries installed:

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Data Preprocessing

Before training the model, the data must be preprocessed and balanced:

1. **Preprocess the Data**: The script `preprocess_data.py` handles missing values, converts categorical variables to numerical, bins continuous variables, and balances the dataset.
2. **Balance the Data**: Using the script `balance_data.py`, split the training data based on the 'isFraud' column and balance it to address the class imbalance problem.
3. **Oversample 'isFraud' Cases**: The `oversample_isfraud.py` script increases the representation of fraud cases in the dataset.
4. **Combine Data**: Use the `combine_data.sh` bash script to merge the fraud and non-fraud data into a single CSV file.
5. **Shuffle the Data**: Finally, `shuffle_data.py` shuffles the combined dataset to randomize the order of instances.

### Training the Model

After preprocessing, use the RandomForest classifier implemented in `random_forest.py` to train the model on the balanced and shuffled training dataset.

```python
from random_forest import RandomForest

# Initialize and train the RandomForest
r = RandomForest()
r.fit(df_train, 0, 10)
```

### Predicting Fraud Instances

Use the trained model to predict fraud instances on a new dataset.

```python
predictions = r.predict(df_test)
```

### Plotting Histograms

The `plot_histogram.py` script can be used to visualize the distribution of features using histograms, aiding in understanding data distribution and feature selection.

## Contributing

Feel free to fork the repository and submit pull requests to contribute to this project.

## Authors

- **Manjil Pradhan** - Initial work and development.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- Thanks to the machine learning community for the inspiration and resources to develop this project.
