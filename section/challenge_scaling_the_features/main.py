import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins_imputed_encoded.csv')
# Assign X,y variables
X, y = df.drop('species', axis=1), df['species']
# Initialize a sclaer and scale the X matrix
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)