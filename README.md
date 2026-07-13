# EduPro: Student Segmentation & Recommendation System

## Project Overview
EduPro is an end-to-end machine learning system designed to personalize online education experiences by segmenting student learning behavior. The platform processes learners, courses, and transactions to analyze engagement patterns, course categories, and difficulty levels. Using unsupervised learning, students are partitioned into cohesive clusters which are then used by a collaborative course recommendation engine to suggest highly relevant training modules.

## Architecture
- **Data Generation Layer**: Programmatically generates synthetic data for learners, course offerings, and enrollment transactions using numpy and faker.
- **Exploratory Data Analysis Layer**: Visualizes category enrollment, level distribution, transaction prices, and user activity profiles.
- **Feature Engineering Layer**: Extracts user-level metrics summarizing engagement span, diversity score, depth indices, and average spending.
- **Preprocessing Layer**: Encodes categorical values and standardizes features using StandardScaler for stable clustering.
- **Clustering Layer**: Uses KMeans (K=4) to segments learners into behavior-based profiles backed by silhouette score selection.
- **Personalized Recommender Layer**: Scores and filters course recommendations based on segment popularities and overall course ratings.

## Learner Segments

| Segment | Description | Key Traits |
| --- | --- | --- |
| Curious Explorers | Students exploring broad domains with wide interests | High diversity score, moderate spending, generalist learning |
| Career Climbers | Professional students aiming for growth and certification | High spending, certificate/paid course focus, target-oriented |
| Deep Specialists | Users targeting vertical subject areas in depth | Low diversity, high learning depth index, specialized expertise |
| Casual Learners | Passive learners with minimal course investments | Low spending, low course count, casual platform usage |

## Quick Start
```bash
git clone <repo>
pip install -r requirements.txt
python src/generate_data.py
streamlit run app/streamlit_app.py
```

## Project Structure
```text
edupro/
├── requirements.txt
├── README.md
├── data/
│   ├── users.csv
│   ├── courses.csv
│   ├── transactions.csv
│   ├── learner_profiles.csv
│   └── learner_profiles_clustered.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_recommendation.ipynb
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── clustering.py
│   └── recommender.py
├── models/
│   ├── kmeans_model.pkl
│   └── scaler.pkl
├── outputs/
│   ├── eda_overview.png
│   ├── cluster_selection.png
│   └── cluster_visualization.png
└── app/
    └── streamlit_app.py
```

## Tech Stack

| Library | Version | Purpose |
| --- | --- | --- |
| Python | 3.10+ | Core language execution |
| pandas | >=2.0.0 | Dataframe structured manipulation |
| numpy | >=1.24.0 | Numerical operations and array math |
| matplotlib | >=3.7.0 | Static data visualizations |
| seaborn | >=0.12.0 | High-level statistical charts |
| scikit-learn | >=1.3.0 | KMeans modeling and preprocessing |
| scipy | >=1.11.0 | Calculations for mode finding |
| streamlit | >=1.28.0 | Front-end web platform dashboard |
| plotly | >=5.15.0 | Dynamic interactive visualizations |
| faker | >=18.0.0 | Generating synthetic demographics |
| joblib | >=1.3.0 | Scaling and clustering serialization |

## Evaluation Results
- **Silhouette Score**: 0.2925

## Security & Environment Variables
If you ever integrate real databases or third-party APIs (such as external user registries or LLM services) into this dashboard, always manage credentials using environment variables via a `.env` file (which is ignored by Git).

> [!WARNING]
> **Git History Warning:** If any API keys, passwords, database URLs, or credentials were previously hardcoded and committed, those old values still exist in the Git history. You must rotate any previously exposed secrets immediately to prevent unauthorized access.
