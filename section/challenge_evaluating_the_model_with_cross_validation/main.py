import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins_pipelined.csv')
# Assign X, y variables (X is already preprocessed and y is already encoded)
X, y = df.drop('species', axis=1), df['species']
# Initialize a model
model = KNeighborsClassifier(n_neighbors=4)
# Calculate and print the mean of cross validation scores
scores = cross_val_score(model, X, y, cv=3)
print('Cross-val score:', scores.mean())
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Train a model
model.fit(X_train,y_train)
# Print the score using the test set
print('Train-test score:', model.score(X_train, y_train))