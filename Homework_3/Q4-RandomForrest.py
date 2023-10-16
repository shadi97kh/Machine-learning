import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the dataset
data = pd.read_csv("train.csv")

# 1. Preprocessing
# Impute and OneHotEncode categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Impute and scale numerical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Defining numeric and categorical columns
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Embarked', 'Sex', 'Pclass']

# Final preprocessor object applying transformations to respective columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = data.drop('Survived', axis=1)
y = data['Survived']

# 2. Feature Selection (Using all the features for simplicity in this example)

# 3. Decision Tree Model & Fine-tuning
# Split data to train and validate the model's performance
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Define and train model
model_tree = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', DecisionTreeClassifier(random_state=0, max_depth=5))
                            ])

model_tree.fit(X_train, y_train)

# 4. Decision Tree CV Accuracy
scores_tree = cross_val_score(model_tree, X_train, y_train, cv=5)
average_accuracy_tree = scores_tree.mean()
print(f"Decision Tree CV Accuracy: {average_accuracy_tree}")

# 5. Random Forest Model & Fine-tuning
model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestClassifier(random_state=0, n_estimators=100, max_depth=5))
                          ])

model_rf.fit(X_train, y_train)

# Random Forest CV Accuracy
scores_rf = cross_val_score(model_rf, X_train, y_train, cv=5)
average_accuracy_rf = scores_rf.mean()
print(f"Random Forest CV Accuracy: {average_accuracy_rf}")

# 6. Which algorithm is better?
better_algorithm = "Decision Tree" if average_accuracy_tree > average_accuracy_rf else "Random Forest"
print(f"The better algorithm is: {better_algorithm}")

# 7. Observations and Conclusions
# This is more of a subjective section based on the results you've obtained. As an example:
if better_algorithm == "Decision Tree":
    print("While Decision Tree performed better, further optimizations and testing on unseen data is essential.")
else:
    print("Random Forest outperformed Decision Tree, showcasing its robustness by aggregating multiple decision trees.")