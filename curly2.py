import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# For this example, let's focus on a subset of features
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']]

# Fill missing age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)



# Initializing the hypothesis with null values
hypothesis = ["0"] * len(df.columns)

# Iterate through the training examples
for index, row in df.iterrows():
    # Following the Find-S Algorithm
    if row['Survived'] == 1:
        for i, value in enumerate(row):
            if hypothesis[i] == "0":
                hypothesis[i] = "?"
            elif hypothesis[i] != value:
                hypothesis[i] = value

print("Final Hypothesis: ")
print(hypothesis)
