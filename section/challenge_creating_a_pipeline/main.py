import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline 

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins.csv')
# Removing rows with more than 1 null
df = df[df.isna().sum(axis=1) < 2] 
# Assigining X, y variables
X, y = df.drop('species', axis=1), df['species']
# Create the ColumnTransformer for encoding
ct = make_column_transformer((OneHotEncoder(), ['sex','island']), remainder='passthrough')
# Make a Pipeline of ct, SimpleImputer, and StandardScaler
pipe = make_pipeline(ct, SimpleImputer(strategy='most_frequent'), StandardScaler())
# Transform X using the pipeline and print transformed X
X_transformed = pipe.fit_transform(X)
print(X_transformed)