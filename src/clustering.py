import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def run_clustering():
    """
    Trains K-Means clustering on the scaled profiles, evaluates optimal K from 2 to 10,
    saves the selection plot and final 2D PCA cluster visualization, maps cluster IDs
    to human-readable segments dynamically based on feature centroids, and saves
    the final model and clustered dataset.
    """
    print("=" * 60)
    print("PHASE 5: K-Means Clustering & Segment Analysis")
    print("=" * 60)

    # 1. Load scaled profiles
    data_dir = Path("data")
    models_dir = Path("models")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    try:
        profiles = pd.read_csv(data_dir / "learner_profiles.csv")
        scaler = joblib.load(models_dir / "scaler.pkl")
        print("Dataset and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Select clustering features
    cluster_features = [
        'total_courses', 'unique_categories', 'avg_rating_enrolled', 'total_spending', 
        'avg_spending', 'diversity_score', 'learning_depth_index', 
        'preferred_category_enc', 'preferred_level_enc', 'Age'
    ]
    
    # Scale features
    X = profiles[cluster_features].fillna(0)
    X_scaled = scaler.transform(X)

    # STEP 1: FIND OPTIMAL K
    print("\nEvaluating K values from 2 to 10...")
    k_values = list(range(2, 11))
    inertias = []
    silhouettes = []

    print(f"{'K':^5} | {'Inertia':^12} | {'Silhouette Score':^18}")
    print("-" * 43)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, km.labels_)
        inertias.append(inertia)
        silhouettes.append(sil)
        print(f"{k:^5} | {inertia:^12.4f} | {sil:^18.4f}")

    # Save selection curves to outputs/cluster_selection.png
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (SSE)', color=color)
    ax1.plot(k_values, inertias, marker='o', color=color, linewidth=2, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_values, silhouettes, marker='s', color=color, linewidth=2, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('K-Means Clustering: Elbow & Silhouette Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    selection_path = outputs_dir / "cluster_selection.png"
    plt.savefig(selection_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster selection plot to {selection_path}")

    # STEP 2: TRAIN FINAL MODEL (OPTIMAL K = 4)
    optimal_k = 4
    print(f"\nTraining final K-Means model with K = {optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    profiles["Cluster"] = kmeans.fit_predict(X_scaled)

    # STEP 3: ANALYZE AND NAME SEGMENTS DYNAMICALLY
    # We analyze the centroids to map clusters to correct traits:
    # - Casual Learners: lowest course count / spending (index of min total_courses)
    # - Career Climbers: highest total spending among the remaining clusters
    # - Deep Specialists: highest learning depth index among the remaining clusters
    # - Curious Explorers: remaining cluster (high diversity score)
    
    means = profiles.groupby("Cluster")[cluster_features].mean()
    
    # Identify Casual Learners
    casual_cluster = means["total_courses"].idxmin()
    
    # Filter out Casual Learners for next steps
    active_clusters = [c for c in range(optimal_k) if c != casual_cluster]
    
    # Identify Career Climbers
    climber_cluster = means.loc[active_clusters, "total_spending"].idxmax()
    
    # Identify remaining two clusters
    other_clusters = [c for c in active_clusters if c != climber_cluster]
    
    # Identify Deep Specialists (higher learning depth index)
    if means.loc[other_clusters[0], "learning_depth_index"] > means.loc[other_clusters[1], "learning_depth_index"]:
        specialist_cluster = other_clusters[0]
        explorer_cluster = other_clusters[1]
    else:
        specialist_cluster = other_clusters[1]
        explorer_cluster = other_clusters[0]

    cluster_names = {
        explorer_cluster: "Curious Explorers",
        climber_cluster: "Career Climbers",
        specialist_cluster: "Deep Specialists",
        casual_cluster: "Casual Learners"
    }

    profiles["Segment"] = profiles["Cluster"].map(cluster_names)
    print("\nMapped Cluster IDs to segments based on feature analysis:")
    for cid, name in cluster_names.items():
        print(f"  Cluster {cid} -> {name}")

    # STEP 4: PCA VISUALIZATION (2D SCATTER)
    print("\nPerforming PCA dimension reduction to 2D...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    profiles["pca_x"] = X_pca[:, 0]
    profiles["pca_y"] = X_pca[:, 1]

    plt.figure(figsize=(10, 8))
    # Map colors systematically
    segment_palette = {
        "Curious Explorers": "#3498db",
        "Career Climbers": "#2ecc71",
        "Deep Specialists": "#9b59b6",
        "Casual Learners": "#95a5a6"
    }
    
    sns.scatterplot(
        data=profiles,
        x="pca_x",
        y="pca_y",
        hue="Segment",
        palette=segment_palette,
        style="Segment",
        s=80,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.8
    )
    plt.title("EduPro Student Segmentation (PCA 2D Projection)", fontsize=14, fontweight='bold')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Learner Segments", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    vis_path = outputs_dir / "cluster_visualization.png"
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster visualization plot to {vis_path}")

    # STEP 5: SAVE ARTIFACTS
    model_path = models_dir / "kmeans_model.pkl"
    joblib.dump(kmeans, model_path)
    print(f"Saved KMeans model to {model_path}")

    clustered_path = data_dir / "learner_profiles_clustered.csv"
    profiles.to_csv(clustered_path, index=False)
    print(f"Saved clustered dataset to {clustered_path}")

    # REPORT METRICS
    final_silhouette = silhouette_score(X_scaled, profiles["Cluster"])
    print(f"\nFinal Silhouette Score: {final_silhouette:.4f}")
    print("\nLearner Segment Value Counts:")
    print(profiles["Segment"].value_counts())
    print("=" * 60)
    print("SUCCESS: Clustering pipeline completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    run_clustering()
