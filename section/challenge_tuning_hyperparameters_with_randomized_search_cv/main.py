import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins_pipelined.csv')
# Assign X, y variables (X is already preprocessed and y is already encoded)
X, y = df.drop('species', axis=1), df['species']
# Create the param_grid and initialize a model
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 12, 15, 17, 20, 25],
                         'weights': ['distance', 'uniform'],
                         'p': [1, 2, 3, 4, 5]
}
model = KNeighborsClassifier()
# Initialize RandomizedSearchCV and GridSearchCV
randomized = RandomizedSearchCV(model, param_grid,n_iter=20, cv=5)
grid = GridSearchCV(model, param_grid, cv=5)
# Train the GridSearchCV object. During training it finds the best parameters
grid.fit(X,y)
randomized.fit(X,y)
# Print the best estimator and its cross-validation score
print('GridSearchCV:')
print(grid.best_estimator_)
print(grid.best_score_)
print('RandomizedSearchCV:')
print(randomized.best_estimator_)
print(randomized.best_score_)