import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def engineer_features(df):
    df = df.copy()
    # Age Group
    df['age_group'] = pd.cut(df['age'], bins=[0, 45, 65, 100], labels=['Young', 'Middle Age', 'Senior'])
    
    # Risk Factor Count
    # high trestbps (>130), high chol (>240), fbs (>120 is fbs=1 in dataset)
    df['risk_factor_count'] = (
        (df['trestbps'] > 130).astype(int) + 
        (df['chol'] > 240).astype(int) + 
        (df['fbs'] == 1).astype(int)
    )
    return df

def main():
    print("Loading data...")
    df = pd.read_csv('../data/dataset.csv')
    
    print("Engineering features...")
    df = engineer_features(df)
    
    # Save the engineered dataset for analytics dashboard
    df.to_csv('../data/dataset_engineered.csv', index=False)
    
    # Define features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Identify column types
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'age_group']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'risk_factor_count']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    print("Training models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_model = None
    best_f1 = 0
    best_model_name = ""
    best_pipeline = None
    
    results = []
    
    for name, model in models.items():
        # Pipeline with SMOTE for class imbalance (even if slight)
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
        roc = roc_auc_score(y_test, y_proba)
        
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': roc
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model based on F1-score: {best_model_name}")
    
    # Save artifacts
    os.makedirs('../models', exist_ok=True)
    
    # We will save the entire pipeline as disease_model.pkl
    # This is easier for FastAPI than separating scaler and encoder, though the pdf suggests separate.
    # The pdf says "Alternatively, create a complete preprocessing + model pipeline: disease_pipeline.pkl". We will do this.
    joblib.dump(best_pipeline, '../models/disease_pipeline.pkl')
    
    # Save the feature names used for training
    feature_columns = numerical_cols + categorical_cols
    joblib.dump(feature_columns, '../models/feature_columns.pkl')
    
    # Save evaluation results
    pd.DataFrame(results).to_csv('../outputs/model_evaluation.csv', index=False)
    
    print("Model and artifacts saved successfully.")

if __name__ == "__main__":
    main()
