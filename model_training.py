from xgboost import XGBRegressor, XGBClassifier
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

model_classifier = XGBClassifier(objective='binary:logistic', n_estimators=100, max_depth=5)

# Regressor Model is not used for this project, If you can use it for more advanced operations,
model_regressor = XGBRegressor(n_estimator=100, random_state=42) 

data = pd.read_csv('synthetic_heart_disease_dataset.csv')
df = pd.DataFrame(data)

def cleaning():
    if np.sum(df.isna().to_numpy()) >= 1:
        return f"Null data: {df.isnull().sum()}"
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].np.mean())
    else:
        return ""

    return df
        
# Cleaning function is very important if you are working with messy data. It check if the data exist or not if not it fill with the mean data

# If NAN or empty integer data is founded this work well.  

#In this code is trying to find the chances for being having a heart disease. The best practice is to avoid while loop for ml/ai project. Because it can cause for a slow running process and use resources. 
def heart_attack(df):
    disease_chance = np.zeros(len(df), dtype='int')

    disease_chance += np.where(df['Weight'] >= 95, 1,0)

    smoking_plus_drinking = (df['Smoking'] != 'None') & (df['Alcohol_Intake'] != 'None')
    disease_chance += np.where(smoking_plus_drinking, 1,0)

    disease_chance += np.where(df['Heart_Disease'] != 0, 1, 0)

    print(f"The disease chance list is: {disease_chance} ")

    return disease_chance
   
def model(df, disease_chance):
    x = df[['Age', 'Weight', 'Smoking', 'Alcohol_Intake']]
    x = pd.get_dummies(x, columns=['Smoking', 'Alcohol_Intake'], dtype=int)
    y = disease_chance - 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model_classifier.fit(x_train, y_train)

    heart_disease = model_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, heart_disease)
    
    print(f"Accuracy: {accuracy*100:.2f}%, Classification Report: {classification_report(y_test, heart_disease)}")

disease_chance = heart_attack(df)
model(df, disease_chance)

heart_chance = heart_attack(df)
model(df, disease_chance)

if __name__ == '__main__':
    pass
    
