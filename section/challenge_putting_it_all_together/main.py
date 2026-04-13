import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import make_column_transformer

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins.csv')
# Removing rows with more than 1 null
df = df[df.isna().sum(axis=1) < 2] 
# Assigining X, y variables
X, y = df.drop('species', axis=1), df['species']
# Encode the target
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)
# Make a train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Create the ColumnTransformer for encoding features
ct = make_column_transformer((OneHotEncoder(), ['island','sex']), remainder='passthrough')
# Make a param_grid for the grid search and initialize the GridSearchCV object
param_grid = {'n_neighbors': [1,3,5,7,9,11,13,15],
                         'weights': ['distance', 'uniform'],
                         'p': [1, 2]
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid)
# Make a Pipeline of ct, SimpleImputer, and StandardScaler
pipe = make_pipeline(ct, SimpleImputer(strategy='most_frequent'), StandardScaler(),grid_search)
# Train the model
pipe.fit(X_train, y_train)
# Print score
print(pipe.score(X_test,y_test))
# Print predictions
y_pred = pipe.predict(X_test) # Get encoded predictions
print(label_enc.inverse_transform(y_pred[:5])) # Decode predictions and print 5 first
# Print the best estimator
print(grid_search.best_params_)