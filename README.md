# Linear Regression from Scratch

This project provides an educational implementation of simple linear regression, built entirely from scratch without using machine learning libraries like scikit-learn. It aims to offer a deep understanding of the mathematical and algorithmic concepts behind this fundamental method.

## Objectives

- Gain an in-depth understanding of linear regression.
- Implement a simple linear regression model in Python.
- Explore and analyze a dataset to extract insights.
- Evaluate the model's performance using standard metrics.

## Features

- **Multiple Implementations**: Simple Linear Regression, Ridge Regression, and Lasso Regression.
- **Exploratory Data Analysis (EDA)**: Data type inspection, descriptive statistics, and visualizations.
- **Model Evaluation**: Metrics such as R², RMSE, and Leave-One-Out Cross-Validation.
- **Visualization**: Graphs to represent relationships between variables and predictions.
- **Flexibility**: Ability to adjust hyperparameters like lambda for Ridge and Lasso.

## Prerequisites

- Python 3.x
- Required libraries: NumPy, Pandas, Matplotlib, Seaborn

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/adam.chgour/LinearRegressionFromScratch.git
cd LinearRegressionFromScratch
pip install -r requirements.txt
```

## Project Structure

```
LinearRegressionFromScratch/
├── linear_regression.py    # Main model implementation
├── analysis/
│   ├── EDA.ipynb           # Exploratory Data Analysis
├── utils.py                # Utility functions
├── tests/                  # Unit tests
└── README.md               # Project documentation
```

## Exploratory Data Analysis (EDA)

An exploratory data analysis was conducted to better understand the dataset. Here is an overview of the steps:

1. **Data Inspection**: Checking data types, missing values, and variable distributions.
2. **Descriptive Statistics**: Means, standard deviations, minimums, and maximums.
3. **Visualizations**: Graphs to explore relationships between variables.

### Example Results

#### Data Types
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   region    1338 non-null   object 
 6   charges   1338 non-null   float64
```

#### Descriptive Statistics
```
Summary Statistics (Numerical Features):
               age          bmi     children       charges
count  1338.000000  1338.000000  1338.000000   1338.000000
mean     39.207025    30.663397     1.094918  13270.422265
std      14.049960     6.098187     1.205493  12110.011237
min      18.000000    15.960000     0.000000   1121.873900
max      64.000000    53.130000     5.000000  63770.428010
```

## Usage

Here is an example of how to use the linear regression model:

```python
from linear_regression import LinearRegression

# Load the data
X_train, X_test, y_train, y_test = ...  # Prepare the data

# Create an instance of the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Display the coefficients
print(f"Slope: {model.slope}, Intercept: {model.intercept}")
```

## Author

Created by Adam CHGOUR. Feel free to reach out for any questions or suggestions!
