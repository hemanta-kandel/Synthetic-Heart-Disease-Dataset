from xgboost import XGBRegressor, XGBClassifier
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

model_classifier = XGBClassifier(objective='binary:logistic', n_estimator=100, depth=5)

# Regressor Model is not used for this project, If you can use it for more advanced operations,
model_regressor = XGBRegressor(n_estimator=100, random_state=42) 

data = pd.read_csv('synthetic_heart_disease_dataset.csv')
df = pd.DataFrame(data)

def cleaning():
    all = df[:, :]
    if sum(df[all].isnull()) >= 1:
        return f"Null data: {df.isnull().sum()}"
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].np.mean())
    else:
        return ""
        
# Cleaning function is very important if you are working with messy data. It check if the data exist or not if not it fill with the mean data

# If NAN or empty integer data is founded this work well.  

def heart_attack():
    disease_chance = 0
    i = 0
    while len(df['Age']) > i:
        if df['Weight'] >= 95:
            disease_chance += 1
            if df['Smoking'] != 'None' & df['Alcohol_Intake'] != 'None':
                disease_chance += 1
            else:
                i += 1
        else:
            i += 1

    j = 0

    while len(df['Heart_Diease']) > j:
        if df['Heart_Disease', j] != 0:
            disease_chance += 1
        else:
            j += 1

    print(f"The Heart Disease chancex is {disease_chance}")
    
def model():
    x = np.array([[df.iloc[:, [2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]]])
    y = np.array([df.iloc[:, 0]])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, n_estimator=100, random_state=42)

    heart_disease = model_classifier.predict(x_test)
    accuracy = accuracy_score(heart_disease)
    score_r2 = r2_score(heart_disease)

    print(f"Accuracy: {accuracy}, R2 Score: {score_r2}")
    
    
if __name__ == '__main__':
    cleaning()
    heart_attack()
    model()
