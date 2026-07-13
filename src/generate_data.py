import numpy as np
import pandas as pd
import random
from pathlib import Path

def generate_edupro_dataset():
    """
    Generates synthetic dataset for EduPro platform including Users, Courses,
    and Transactions tables according to specified schema and proportions.
    """
    print("=" * 60)
    print("PHASE 1: Programmatic Synthetic Data Generation")
    print("=" * 60)

    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Create target directory
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate users.csv: exactly 500 rows, columns: UserID, Age, Gender
    print("Generating users.csv...")
    user_ids = [f"U{i:04d}" for i in range(1, 501)]
    ages = np.random.randint(18, 55, size=500)
    genders = np.random.choice(["Male", "Female", "Other"], size=500, p=[0.48, 0.48, 0.04])
    users_df = pd.DataFrame({
        "UserID": user_ids,
        "Age": ages,
        "Gender": genders
    })
    
    # Save users table
    users_path = data_dir / "users.csv"
    users_df.to_csv(users_path, index=False)
    print(f"Saved {len(users_df)} users to {users_path}")

    # 2. Generate courses.csv: exactly 100 rows, columns: CourseID, CourseCategory, CourseType, CourseLevel, CourseRating
    print("Generating courses.csv...")
    course_ids = [f"C{i:04d}" for i in range(1, 101)]
    categories = ["Data Science", "Web Developmen", "Cybersecurity", "Business", "Design", "AI/ML", "Cloud Computing"]
    course_categories = np.random.choice(categories, size=100)
    course_types = np.random.choice(["Free", "Paid", "Certificate"], size=100, p=[0.30, 0.50, 0.20])
    course_levels = np.random.choice(["Beginner", "Intermediate", "Advanced"], size=100, p=[0.40, 0.35, 0.25])
    course_ratings = np.round(np.random.uniform(3.0, 5.0, size=100), 2)

    courses_df = pd.DataFrame({
        "CourseID": course_ids,
        "CourseCategory": course_categories,
        "CourseType": course_types,
        "CourseLevel": course_levels,
        "CourseRating": course_ratings
    })
    
    # Save courses table
    courses_path = data_dir / "courses.csv"
    courses_df.to_csv(courses_path, index=False)
    print(f"Saved {len(courses_df)} courses to {courses_path}")

    # 3. Generate transactions.csv: ~3000 raw transactions, deduped on (UserID, CourseID)
    print("Generating transactions.csv...")
    # Setup popularity weights for users and courses to introduce realistic duplicate counts
    user_popularity = np.random.exponential(scale=1.0, size=500)
    user_popularity /= user_popularity.sum()

    course_popularity = np.random.exponential(scale=1.0, size=100)
    course_popularity /= course_popularity.sum()

    # Sample user and course IDs with weights
    u_samples = np.random.choice(user_ids, size=3000, p=user_popularity)
    c_samples = np.random.choice(course_ids, size=3000, p=course_popularity)

    # Date range: 2023-01-01 to 2024-12-31
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-12-31')
    delta_days = (end_date - start_date).days
    random_days = np.random.randint(0, delta_days + 1, size=3000)
    dates = start_date + pd.to_timedelta(random_days, unit='D')

    raw_tx_df = pd.DataFrame({
        "UserID": u_samples,
        "CourseID": c_samples,
        "TransactionDate": dates
    })

    # Deduplicate on UserID and CourseID, keeping the first occurrence
    tx_df = raw_tx_df.drop_duplicates(subset=["UserID", "CourseID"]).copy()
    
    # Left merge courses table to check type and set transaction amounts
    tx_df = tx_df.merge(courses_df[["CourseID", "CourseType"]], on="CourseID", how="left")

    # Set seed for reproducible pricing
    np.random.seed(42)
    amounts = []
    for course_type in tx_df["CourseType"]:
        if course_type == "Free":
            amounts.append(0.00)
        else:
            amounts.append(round(np.random.uniform(9.99, 199.99), 2))
    
    tx_df["Amount"] = amounts
    
    # Select and reorder columns
    tx_df = tx_df[["UserID", "CourseID", "TransactionDate", "Amount"]]
    
    # Save transactions table
    tx_path = data_dir / "transactions.csv"
    tx_df.to_csv(tx_path, index=False)
    print(f"Saved {len(tx_df)} unique transactions to {tx_path}")
    print("=" * 60)
    print("SUCCESS: Data generation completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    generate_edupro_dataset()
