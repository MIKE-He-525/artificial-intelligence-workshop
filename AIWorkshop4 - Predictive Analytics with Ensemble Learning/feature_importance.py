import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load housing data
import pandas as pd

# Load the dataset
data = pd.read_csv('boston_house_prices.csv')

data_target = np.array(data.MEDV)
data_feature = np.array(data.iloc[:, :-1])

# Display the first few rows of the dataset
# print(data.head())

print(data_target.shape)
print(data_feature.shape)

# Shuffle the data
# X, y = shuffle(housing_data.data, housing_data.target, random_state=7)
X, y = shuffle(data_feature, data_target, random_state=7)

# Split data into training and testing datasets 
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)

# AdaBoost Regressor model
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), 
        n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)

# Evaluate performance of AdaBoost regressor
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred )
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Extract feature importances
feature_importances = regressor.feature_importances_
feature_names = data.columns

# Normalize the importance values 
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Sort the values and flip them
index_sorted = np.flipud(np.argsort(feature_importances))

# Arrange the X ticks
pos = np.arange(index_sorted.shape[0]) + 0.5

plt.figure(figsize=(10, 6))  # Set figure size for clarity
# Create a bar chart: x-axis = sorted feature names, y-axis = normalized importance
plt.bar(
    pos,  # X-axis positions for bars
    feature_importances[index_sorted],  # Sorted importance values
    align='center'  # Align bars with x-axis ticks
)
# Set x-axis ticks and labels (sorted feature names)
plt.xticks(pos, feature_names[index_sorted], rotation=45, ha='right')  # Rotate labels for readability
# Set chart title and axis labels (consistent with the document's chart title)
plt.title('Feature Importance Using AdaBoost Regressor')
plt.xlabel('Feature Names')
plt.ylabel('Relative Importance (%)')
plt.tight_layout()  # Adjust layout to prevent label cutoff

# Save the chart (required for submission in Workshop 4.6)
plt.savefig('feature_importance_chart.png')
plt.show()

# 10. Print the top 4 most important features (answers Workshop 4.6 Question 4)
print("\nTOP 4 MOST IMPORTANT FEATURES")
print("=" * 40)
for i in range(4):
    feature_idx = index_sorted[i]
    feature_name = feature_names[feature_idx]
    feature_importance = feature_importances[feature_idx]
    print(f"{i+1}. {feature_name}: {round(feature_importance, 2)}%")
print("=" * 40)
