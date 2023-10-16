import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the dataset
data = pd.read_csv("train.csv")

# Preprocessing, as previously described
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Embarked', 'Sex', 'Pclass']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = data.drop('Survived', axis=1)
y = data['Survived']

# Split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Bagging Classifier using DecisionTree as the base estimator
base_est = DecisionTreeClassifier(random_state=0, max_depth=5)
bagging_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', BaggingClassifier(base_est, n_estimators=100, random_state=0))
                               ])

bagging_model.fit(X_train, y_train)

# Evaluating Bagging Classifier using CV
scores_bagging = cross_val_score(bagging_model, X_train, y_train, cv=5)
average_accuracy_bagging = scores_bagging.mean()
print(f"Bagging Classifier CV Accuracy: {average_accuracy_bagging}")