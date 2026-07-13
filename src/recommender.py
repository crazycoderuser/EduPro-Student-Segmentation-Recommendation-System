import pandas as pd
import numpy as np
from pathlib import Path

# Module-level paths
data_dir = Path("data")

# Global variables to cache tables
_profiles = None
_courses = None
_transactions = None

def _load_data():
    """
    Loads and caches the clustered profiles, courses, and transactions dataframes
    at the module level.
    """
    global _profiles, _courses, _transactions
    if _profiles is None or _courses is None or _transactions is None:
        try:
            # Look for learner_profiles_clustered.csv, fallback to learner_profiles.csv if not clustered yet
            clustered_path = data_dir / "learner_profiles_clustered.csv"
            if clustered_path.exists():
                _profiles = pd.read_csv(clustered_path)
            else:
                _profiles = pd.read_csv(data_dir / "learner_profiles.csv")
                
            _courses = pd.read_csv(data_dir / "courses.csv")
            _transactions = pd.read_csv(data_dir / "transactions.csv")
        except Exception as e:
            print(f"Error loading and caching files in recommender module: {e}")
            _profiles = pd.DataFrame()
            _courses = pd.DataFrame()
            _transactions = pd.DataFrame()

def get_recommendations(
    user_id: str,
    top_n: int = 5,
    level_filter: list = None,
    category_filter: list = None
) -> pd.DataFrame:
    """
    Generates course recommendations for a learner using cluster-aware collaborative
    filtering. Scores courses based on popularity within the learner's segment and
    overall course rating, filtering out courses the student has already completed.

    Parameters:
    -----------
    user_id : str
        The unique UserID of the student (e.g. 'U0001')
    top_n : int, default 5
        The maximum number of courses to recommend
    level_filter : list, default None
        List of levels to filter courses by (e.g. ['Beginner', 'Intermediate'])
    category_filter : list, default None
        List of categories to filter courses by (e.g. ['Cybersecurity'])

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: [CourseID, CourseCategory, CourseLevel, CourseRating, final_score]
    """
    _load_data()
    
    # Check if datasets were loaded successfully
    if _profiles.empty or _courses.empty or _transactions.empty:
        print("Error: Recommender datasets are empty. Cannot make recommendations.")
        return pd.DataFrame(columns=["CourseID", "CourseCategory", "CourseLevel", "CourseRating", "final_score"])

    # Step 2: Look up user_id in profiles -> get their Cluster ID
    user_row = _profiles[_profiles["UserID"] == user_id]
    if user_row.empty:
        print(f"Warning: UserID '{user_id}' not found in learner profiles. Returning empty recommendations.")
        return pd.DataFrame(columns=["CourseID", "CourseCategory", "CourseLevel", "CourseRating", "final_score"])
    
    cluster_id = user_row.iloc[0]["Cluster"]
    pref_level = user_row.iloc[0]["preferred_level"]

    # Step 3: Get all UserIDs in the same Cluster
    cluster_users = _profiles[_profiles["Cluster"] == cluster_id]["UserID"]

    # Step 4: Get all transactions by those cluster members
    cluster_txn = _transactions[_transactions["UserID"].isin(cluster_users)]
    
    if cluster_txn.empty:
        return pd.DataFrame(columns=["CourseID", "CourseCategory", "CourseLevel", "CourseRating", "final_score"])

    # Step 5: Count enrollments per CourseID within the cluster
    cluster_enrollments = cluster_txn.groupby("CourseID").size().rename("popularity")

    # Step 6: Merge with courses.csv to get rating, category, level
    courses_scored = _courses.merge(cluster_enrollments, on="CourseID", how="left").fillna({"popularity": 0})

    # Step 7: Compute final_score (float, 0.0–1.0)
    max_enroll = cluster_enrollments.max() if not cluster_enrollments.empty else 1
    if max_enroll == 0:
        max_enroll = 1
        
    courses_scored["popularity_score"] = courses_scored["popularity"] / max_enroll
    # Normalize CourseRating from 3.0-5.0 scale to 0.0-1.0
    courses_scored["rating_score"] = (courses_scored["CourseRating"] - 3.0) / 2.0
    courses_scored["rating_score"] = courses_scored["rating_score"].clip(0.0, 1.0)
    
    # Calculate level match score
    if pd.notna(pref_level) and pref_level != "Unknown":
        courses_scored["level_match"] = (courses_scored["CourseLevel"] == pref_level).astype(float)
    else:
        courses_scored["level_match"] = 0.0

    # final_score = 0.4 * popularity_score + 0.4 * rating_score + 0.2 * level_match
    courses_scored["final_score"] = 0.4 * courses_scored["popularity_score"] + 0.4 * courses_scored["rating_score"] + 0.2 * courses_scored["level_match"]

    # Step 8: Remove courses already taken by user_id
    user_taken = _transactions[_transactions["UserID"] == user_id]["CourseID"].tolist()
    courses_scored = courses_scored[~courses_scored["CourseID"].isin(user_taken)]

    # Step 9: Apply level_filter if provided
    if level_filter is not None and len(level_filter) > 0:
        courses_scored = courses_scored[courses_scored["CourseLevel"].isin(level_filter)]

    # Apply category_filter if provided
    if category_filter is not None and len(category_filter) > 0:
        courses_scored = courses_scored[courses_scored["CourseCategory"].isin(category_filter)]

    # Step 10: Sort by final_score DESC, return top_n rows
    recommendations = courses_scored.sort_values(by="final_score", ascending=False).head(top_n)
    
    # Format and return the final dataframe columns
    return recommendations[["CourseID", "CourseCategory", "CourseLevel", "CourseRating", "final_score"]].reset_index(drop=True)

def evaluate_recommendations():
    """
    Samples 100 random users and computes recommendation match rates
    for CourseCategory and CourseLevel preferences. Prints the results.
    """
    _load_data()
    
    if _profiles.empty:
        print("Error: No profiles dataset available for evaluation.")
        return
        
    # Sample 100 random users (or all users if less than 100 exist)
    sample_size = min(100, len(_profiles))
    sampled_users = _profiles.sample(n=sample_size, random_state=42)
    
    category_matches = 0
    level_matches = 0
    total_recs = 0
    
    for _, user in sampled_users.iterrows():
        uid = user["UserID"]
        pref_category = user["preferred_category"]
        pref_level = user["preferred_level"]
        
        # Get top 5 recommendations
        recs = get_recommendations(uid, top_n=5)
        
        if recs.empty:
            continue
            
        for _, rec in recs.iterrows():
            total_recs += 1
            if rec["CourseCategory"] == pref_category:
                category_matches += 1
            if rec["CourseLevel"] == pref_level:
                level_matches += 1
                
    category_match_rate = category_matches / total_recs if total_recs > 0 else 0.0
    level_match_rate = level_matches / total_recs if total_recs > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("PHASE 6: Recommender System Evaluation")
    print("=" * 60)
    print(f"Sampled Users Evaluated:  {sample_size}")
    print(f"Total Recommendations:    {total_recs}")
    print(f"Category Match Rate:      {category_match_rate:.4f}")
    print(f"Level Match Rate:         {level_match_rate:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    _load_data()
    # Print sample recommendations for U0001
    sample_user = "U0001"
    print(f"\nSample recommendations for user '{sample_user}':")
    recs = get_recommendations(sample_user, top_n=5)
    print(recs)
    
    # Evaluate recommender system
    evaluate_recommendations()
