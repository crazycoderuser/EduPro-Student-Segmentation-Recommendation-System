import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_preprocessing():
    """
    Performs data preprocessing for clustering including LabelEncoding for categorical
    columns, StandardScaler scaling for the selected 10 clustering features, and saving
    the fitted scaler object to models/scaler.pkl.
    """
    print("=" * 60)
    print("PHASE 4: Preprocessing (Normalize & Encode)")
    print("=" * 60)

    # 1. Load datasets
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        profiles = pd.read_csv(data_dir / "learner_profiles.csv")
        print("Learner profiles loaded successfully.")
    except Exception as e:
        print(f"Error loading learner profiles: {e}")
        return

    # Check for empty dataframe
    if profiles.empty:
        print("Error: Profiles DataFrame is empty.")
        return

    # STEP 1: ENCODE CATEGORICAL COLUMNS
    print("Encoding categorical columns using LabelEncoder...")
    
    # Initialize LabelEncoders
    le_category = LabelEncoder()
    le_level = LabelEncoder()
    le_gender = LabelEncoder()

    # Fit and transform columns
    profiles["preferred_category_enc"] = le_category.fit_transform(profiles["preferred_category"])
    profiles["preferred_level_enc"] = le_level.fit_transform(profiles["preferred_level"])
    profiles["gender_enc"] = le_gender.fit_transform(profiles["Gender"])

    print("Added encoded columns: preferred_category_enc, preferred_level_enc, gender_enc")

    # STEP 2: SELECT 10 CLUSTERING FEATURES
    cluster_features = [
        'total_courses',           # Engagement: how active?
        'unique_categories',       # Engagement: how broad?
        'avg_rating_enrolled',     # Preference: quality threshold?
        'total_spending',          # Behavioral: lifetime value?
        'avg_spending',            # Behavioral: willingness to pay?
        'diversity_score',         # Behavioral: generalist vs specialist?
        'learning_depth_index',    # Behavioral: beginner vs advanced?
        'preferred_category_enc',  # Preference: dominant domain?
        'preferred_level_enc',     # Preference: preferred difficulty?
        'Age'                      # Demographic context
    ]

    print(f"Selected {len(cluster_features)} clustering features: {cluster_features}")

    # STEP 3: STANDARDIZE
    print("Standardizing clustering features using StandardScaler...")
    X = profiles[cluster_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # VERIFY
    mean_val = np.mean(X_scaled)
    std_val = np.std(X_scaled)
    print(f"Verify scaled mean (should be ≈ 0.0): {mean_val:.6f}")
    print(f"Verify scaled std (should be ≈ 1.0): {std_val:.6f}")

    assert abs(mean_val) < 1e-3, f"Verification failed: Mean is {mean_val}"
    assert abs(std_val - 1.0) < 1e-3, f"Verification failed: Std is {std_val}"
    print("Scaled verification PASSED.")

    # SAVE SCALER AND PROFILES
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved StandardScaler to {scaler_path}")
    
    # Save the updated profiles DataFrame back
    profiles.to_csv(data_dir / "learner_profiles.csv", index=False)
    print(f"Saved updated learner profiles containing encoded columns to {data_dir / 'learner_profiles.csv'}")
    print("=" * 60)
    print("SUCCESS: Preprocessing completed successfully.")
    print("=" * 60)
    
    return X_scaled, profiles

if __name__ == "__main__":
    run_preprocessing()
