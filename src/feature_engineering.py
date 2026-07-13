import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mode

def run_feature_engineering():
    """
    Performs Feature Engineering by extracting engagement, preference, and behavioral
    metrics for each learner on the platform, merging them with demographic data,
    and saving the results to data/learner_profiles.csv.
    """
    print("=" * 60)
    print("PHASE 3: Feature Engineering")
    print("=" * 60)

    # 1. Load datasets
    data_dir = Path("data")
    try:
        users = pd.read_csv(data_dir / "users.csv")
        courses = pd.read_csv(data_dir / "courses.csv")
        transactions = pd.read_csv(data_dir / "transactions.csv")
        print("Data files loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Check for empty dataframes
    if users.empty or courses.empty or transactions.empty:
        print("Error: One or more datasets are empty.")
        return

    # Parse dates in transactions
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])

    # Merge transactions with courses to get category, level, and rating
    merged = transactions.merge(courses, on="CourseID", how="left")

    # Group by UserID to calculate engagement features
    print("Computing engagement features...")
    # total_courses
    total_courses = merged.groupby("UserID")["CourseID"].count().rename("total_courses")
    # unique_categories
    unique_categories = merged.groupby("UserID")["CourseCategory"].nunique().rename("unique_categories")
    # enrollment_span_days
    span_days = merged.groupby("UserID")["TransactionDate"].agg(lambda x: (x.max() - x.min()).days).rename("enrollment_span_days")

    # Combine engagement features
    engagement_df = pd.DataFrame(index=users["UserID"]) # Ensure all 500 users are represented
    engagement_df = engagement_df.join(total_courses, how="left").fillna(0)
    engagement_df = engagement_df.join(unique_categories, how="left").fillna(0)
    engagement_df = engagement_df.join(span_days, how="left").fillna(0)

    # Convert total_courses and unique_categories to int
    engagement_df["total_courses"] = engagement_df["total_courses"].astype(int)
    engagement_df["unique_categories"] = engagement_df["unique_categories"].astype(int)
    engagement_df["enrollment_span_days"] = engagement_df["enrollment_span_days"].astype(int)

    # Compute preference features (mode of category/level, mean course rating)
    print("Computing preference features...")
    
    # Mode category helper
    def get_mode_category(series):
        if series.empty:
            return "Unknown"
        m = series.mode()
        return m.iloc[0] if not m.empty else "Unknown"
    
    # Mode level helper
    def get_mode_level(series):
        if series.empty:
            return "Unknown"
        m = series.mode()
        return m.iloc[0] if not m.empty else "Unknown"

    pref_category = merged.groupby("UserID")["CourseCategory"].agg(get_mode_category).rename("preferred_category")
    pref_level = merged.groupby("UserID")["CourseLevel"].agg(get_mode_level).rename("preferred_level")
    avg_rating = merged.groupby("UserID")["CourseRating"].mean().rename("avg_rating_enrolled")

    # Compute spending features (total, average)
    print("Computing spending features...")
    total_spending = merged.groupby("UserID")["Amount"].sum().rename("total_spending")
    avg_spending = merged.groupby("UserID")["Amount"].mean().rename("avg_spending")

    # Compute diversity score and learning depth index
    print("Computing diversity and depth indices...")
    # diversity_score = unique_categories / total_courses
    diversity_score = (engagement_df["unique_categories"] / (engagement_df["total_courses"] + 1e-9)).rename("diversity_score")

    # learning_depth_index = Advanced_count / (Beginner + Intermediate + Advanced + 1e-9)
    # Get count of courses per level per user
    level_counts = merged.groupby(["UserID", "CourseLevel"]).size().unstack(fill_value=0)
    # Ensure all levels are present
    for lvl in ["Beginner", "Intermediate", "Advanced"]:
        if lvl not in level_counts.columns:
            level_counts[lvl] = 0

    beginner_count = level_counts["Beginner"].rename("Beginner_count")
    intermediate_count = level_counts["Intermediate"].rename("Intermediate_count")
    advanced_count = level_counts["Advanced"].rename("Advanced_count")
    
    depth_index = (level_counts["Advanced"] / (
        level_counts["Beginner"] + level_counts["Intermediate"] + level_counts["Advanced"] + 1e-9
    )).rename("learning_depth_index")

    # MERGE ORDER:
    # 1. Start with engagement DataFrame
    profiles = engagement_df.copy()

    # 2. Left-merge preferred_category, preferred_level, avg_rating
    profiles = profiles.join(pref_category, how="left").fillna({"preferred_category": "Unknown"})
    profiles = profiles.join(pref_level, how="left").fillna({"preferred_level": "Unknown"})
    profiles = profiles.join(avg_rating, how="left").fillna(0.0)

    # 3. Left-merge spending features
    profiles = profiles.join(total_spending, how="left").fillna(0.0)
    profiles = profiles.join(avg_spending, how="left").fillna(0.0)

    # 4. Left-merge diversity_score, learning_depth_index, and course level counts for completeness
    profiles = profiles.join(diversity_score, how="left").fillna(0.0)
    profiles = profiles.join(beginner_count, how="left").fillna(0)
    profiles = profiles.join(intermediate_count, how="left").fillna(0)
    profiles = profiles.join(advanced_count, how="left").fillna(0)
    profiles = profiles.join(depth_index, how="left").fillna(0.0)

    # Convert counts to int
    profiles["Beginner_count"] = profiles["Beginner_count"].astype(int)
    profiles["Intermediate_count"] = profiles["Intermediate_count"].astype(int)
    profiles["Advanced_count"] = profiles["Advanced_count"].astype(int)

    # Reset index to bring UserID as column
    profiles = profiles.reset_index()

    # 5. Left-merge users.csv (adds Age, Gender)
    profiles = profiles.merge(users[["UserID", "Age", "Gender"]], on="UserID", how="left")

    # Output validation
    # No NaN values
    assert profiles.isnull().sum().sum() == 0, "Error: Profiles DataFrame contains NaN values."
    # Shape must be: (500, 14+)
    print(f"Final profiles DataFrame shape: {profiles.shape}")
    assert profiles.shape == (500, 16), f"Error: Unexpected profiles shape: {profiles.shape}"

    # Save to data/learner_profiles.csv
    profiles_path = data_dir / "learner_profiles.csv"
    profiles.to_csv(profiles_path, index=False)
    
    print(f"Saved profiles table to {profiles_path}")
    print("=" * 60)
    print("SUCCESS: Feature Engineering completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    run_feature_engineering()
