import os
from ucimlrepo import fetch_ucirepo

def download_heart_disease_data(output_path="data/dataset.csv"):
    print("Fetching heart disease dataset from UCI...")
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    
    # Combine into a single dataframe
    df = X.copy()
    # The target is 'num', which is the presence of heart disease from 0 (no presence) to 4.
    # We will convert this to a binary classification: 0 for no disease, 1 for disease (1-4).
    df['target'] = y['num'].apply(lambda x: 1 if x > 0 else 0)
    
    # Save to csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset downloaded successfully to {output_path}")

if __name__ == "__main__":
    download_heart_disease_data()
