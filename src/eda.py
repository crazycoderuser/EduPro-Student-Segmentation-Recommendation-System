import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def run_eda():
    """
    Performs Exploratory Data Analysis on the EduPro dataset and generates
    summary metrics and diagnostic plots in outputs/eda_overview.png.
    """
    print("=" * 60)
    print("PHASE 2: Exploratory Data Analysis (EDA)")
    print("=" * 60)

    # 1. Load datasets
    data_dir = Path("data")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

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

    # Merge datasets
    # transactions: UserID, CourseID, TransactionDate, Amount
    # users: UserID, Age, Gender
    # courses: CourseID, CourseCategory, CourseType, CourseLevel, CourseRating
    merged_df = transactions.merge(users, on="UserID", how="left")
    merged_df = merged_df.merge(courses, on="CourseID", how="left")

    # 2. Print metrics to console
    total_records = len(merged_df)
    unique_users = merged_df["UserID"].nunique()
    unique_courses = merged_df["CourseID"].nunique()

    print("\n--- Platform Metrics Summary ---")
    print(f"Total Enrollment Records: {total_records}")
    print(f"Unique Users Enrolled: {unique_users}")
    print(f"Unique Courses Enrolled: {unique_courses}")

    # Null counts per column
    print("\n--- Missing Value Counts per Column ---")
    print(merged_df.isnull().sum())

    # Descriptive stats for Amount and CourseRating
    print("\n--- Descriptive Statistics for Transaction Amount ---")
    print(merged_df["Amount"].describe())

    print("\n--- Descriptive Statistics for Course Rating ---")
    print(merged_df["CourseRating"].describe())

    # Top 3 categories by enrollment count
    category_counts = merged_df["CourseCategory"].value_counts()
    print("\n--- Top 3 Course Categories by Enrollment Count ---")
    for category, count in category_counts.head(3).items():
        print(f"  {category}: {count} enrollments")

    # 3. Generate and save diagnostic plots (2x2 grid)
    print("\nGenerating diagnostic plots in outputs/eda_overview.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EduPro Platform — Exploratory Data Analysis", fontsize=16, fontweight='bold')

    # Plot 1 (top-left): Bar chart — enrollments per CourseCategory
    category_order = category_counts.index
    sns.countplot(
        data=merged_df,
        x="CourseCategory",
        order=category_order,
        color="steelblue",
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Enrollments per Course Category", fontsize=12, fontweight='semibold')
    axes[0, 0].set_xlabel("Course Category")
    axes[0, 0].set_ylabel("Enrollment Count")
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2 (top-right): Pie chart — CourseLevel distribution
    level_counts = merged_df["CourseLevel"].value_counts()
    # Align levels order if possible for neat display
    level_order = ["Beginner", "Intermediate", "Advanced"]
    level_pie_counts = [level_counts.get(lvl, 0) for lvl in level_order]
    axes[0, 1].pie(
        level_pie_counts,
        labels=level_order,
        autopct='%1.1f%%',
        colors=['#66B2FF', '#FF6666', '#99FF99'],
        startangle=90
    )
    axes[0, 1].set_title("Course Level Distribution", fontsize=12, fontweight='semibold')

    # Plot 3 (bottom-left): Histogram — Amount for paid courses only (Amount > 0)
    paid_transactions = merged_df[merged_df["Amount"] > 0]
    axes[1, 0].hist(
        paid_transactions["Amount"],
        bins=30,
        color="coral",
        edgecolor="black",
        alpha=0.7
    )
    axes[1, 0].set_title("Spending Distribution (Paid Courses Only)", fontsize=12, fontweight='semibold')
    axes[1, 0].set_xlabel("Transaction Amount ($)")
    axes[1, 0].set_ylabel("Frequency")

    # Plot 4 (bottom-right): Histogram — enrollments per UserID
    enrollments_per_user = merged_df.groupby("UserID").size()
    axes[1, 1].hist(
        enrollments_per_user,
        bins=20,
        color="mediumseagreen",
        edgecolor="black",
        alpha=0.7
    )
    axes[1, 1].set_title("Enrollments per Learner Distribution", fontsize=12, fontweight='semibold')
    axes[1, 1].set_xlabel("Number of Enrolled Courses")
    axes[1, 1].set_ylabel("Frequency")

    # Layout adjustment and saving
    plt.tight_layout()
    plot_path = outputs_dir / "eda_overview.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved EDA summary plots to {plot_path}")
    print("=" * 60)
    print("SUCCESS: Exploratory Data Analysis completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
