from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler


# 1: LOAD DATASET  
# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features 
y = breast_cancer_wisconsin_original.data.targets 
  
# metadata 
# print(breast_cancer_wisconsin_original.metadata) 
  
# variable information 
# print(breast_cancer_wisconsin_original.variables)

# divide train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

missing_percentage = X.isnull().mean() * 100
columns_to_drop = missing_percentage[missing_percentage > 0].index

# 3: Drop those columns from the DataFrame
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)

# Feature Scaling - better to scale with train and test divided
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 4: Make Predictions
y_pred = model.predict(X_test_scaled)

print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Confusion Matrix
print(f'Confusion Matrix:' + confusion_matrix(y_test, y_pred))

print(f'Classification Report:' + classification_report(y_test, y_pred))
