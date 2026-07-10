import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import grid_search
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utilities import visualize_classifier

# Load input data
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Separate input data into three classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

# Split the data into training and testing datasets 
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

# Define the parameter grid 
parameter_grid = [ {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
                   {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
                 ]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print("\n##### Searching optimal parameters for", metric)

    classifier = grid_search.GridSearchCV(
            ExtraTreesClassifier(random_state=0), 
            parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    print("\nGrid scores for the parameter grid:")
    means = classifier.cv_results_['mean_test_score']
    params = classifier.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f with:   %r" % (mean, param))


 # 【Key Amendment】Output the BEST Parameters (as required by the question)
    print("\n### BEST Parameters for", metric, "###")
    print("Best parameter combination:", classifier.best_params_)
    print("Best cross-validation score:", "%f" % classifier.best_score_)
    print("-" * 50)

# Optional: Evaluate the best model on test dataset (to verify performance)
best_classifier = ExtraTreesClassifier(
    n_estimators=classifier.best_params_['n_estimators'],
    max_depth=classifier.best_params_['max_depth'],
    random_state=0
)
best_classifier.fit(X_train, y_train)
y_test_pred = best_classifier.predict(X_test)

print("\n##### Performance of the BEST Model on Test Dataset #####")
class_names = ['Class-0', 'Class-1', 'Class-2']
print(classification_report(y_test, y_test_pred, target_names=class_names))