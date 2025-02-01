import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def generate_and_process_data():
    # Load the dataset
    data = pd.read_csv('./data/Churn_Modelling.csv')
    
    # Preview the data structure
    print("Dataset loaded successfully.")
    print(data.head())
    
    # Select relevant features and target
    # Assuming 'Exited' is the target column (adjust as necessary)
    features = data.iloc[:,3:13]
    target = data.iloc[:,13]
    
    # Create dummy variables for the 'Geography' categorical feature
    # drop_first=True avoids the dummy variable trap by dropping one category
    geography = pd.get_dummies(features["Geography"], drop_first=True)

    # Create dummy variables for the 'Gender' categorical feature
    gender = pd.get_dummies(features["Gender"], drop_first=True)
    
    # Concatenate the dummy variables with the original feature set
    features = pd.concat([features, geography, gender], axis=1)
    
    # Drop the original categorical columns as they are now encoded
    features = features.drop(["Geography", "Gender"], axis=1)

    # Split the data into training and testing sets (80% train, 20% test)
    features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=0)
    
    #Feature scaling
    sc=StandardScaler()
    features_train=sc.fit_transform(features_train)
    features_test=sc.transform(features_test)
    
    # Save the scaler for deployment
    joblib.dump(sc, './model/scaler.pkl')

    # Save the processed data
    processed_data = pd.DataFrame(features_train, columns=features.columns)
    processed_data['Target'] = target_train.reset_index(drop=True)
    processed_data.to_csv('./data/processed_data.csv', index=False)
    
    print("Data processing complete. Processed data saved.")
    
    return features_train, features_test, target_train, target_test

if __name__ == "__main__":
    generate_and_process_data()
