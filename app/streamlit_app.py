import streamlit as pd_st # Avoid namespace clashes
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
import uuid
from pathlib import Path

# Helper to load local .env variables at startup (avoids external dependencies)
def load_env():
    for path in [Path(".env"), Path(__file__).resolve().parent / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if path.exists() and path.is_file():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
            break

# Load environment configurations
load_env()

# Production Environment Verification Check
DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ("true", "1", "t", "yes")
CRITICAL_ENV_VARS = ["DATABASE_URL", "AI_RECOMMENDER_API_KEY"]
missing_vars = [var for var in CRITICAL_ENV_VARS if not os.getenv(var)]

if missing_vars:
    if not DEBUG_MODE:
        st.error(f"❌ Production Launch Blocked: Missing critical environment variables: {', '.join(missing_vars)}")
        st.stop()
    else:
        # Fallback values for local development
        os.environ.setdefault("DATABASE_URL", "sqlite:///data/edupro.db")
        os.environ.setdefault("AI_RECOMMENDER_API_KEY", "dev_mock_key_12345")

# Configuration

# Add project root to sys.path to resolve src imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.recommender import get_recommendations

# Curated real-world online learning resources map for course recommendations
REAL_WORLD_RESOURCES = {
    "AI/ML": {
        "websites": [
            {"title": "Fast.ai: Deep Learning", "url": "https://course.fast.ai/", "desc": "A legendary free course teaching deep learning from the top down."},
            {"title": "DeepLearning.AI (Coursera)", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "desc": "Andrew Ng's classic, world-famous introduction to machine learning."},
            {"title": "Kaggle Learn", "url": "https://www.kaggle.com/learn", "desc": "Free, hands-on micro-courses for machine learning practitioners."},
            {"title": "Google AI Education", "url": "https://ai.google/education/", "desc": "Curated guide, tutorials, and training pathways for AI developers."}
        ],
        "videos": [
            {"title": "Andrej Karpathy: Zero to Hero", "url": "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUbFy1A", "desc": "The ultimate video guide to building neural networks from scratch."},
            {"title": "3Blue1Brown: Neural Networks", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi", "desc": "Beautifully intuitive visual explanations of backpropagation and deep learning."},
            {"title": "Lex Fridman Podcast: AI Series", "url": "https://www.youtube.com/@lexfridman", "desc": "In-depth conversations with the world's leading artificial intelligence researchers."}
        ],
        "articles": [
            {"title": "Hugging Face NLP Course", "url": "https://huggingface.co/learn/nlp-course", "desc": "Official guide to modern transformers, datasets, and tokenizers."},
            {"title": "Machine Learning Mastery", "url": "https://machinelearningmastery.com/", "desc": "Practical tutorials with clean, working Python/Scikit-Learn code."},
            {"title": "Distill.pub Journal", "url": "https://distill.pub/", "desc": "Highly interactive, visual explanations of complex ML concepts."}
        ]
    },
    "Data Science": {
        "websites": [
            {"title": "Kaggle Playground", "url": "https://www.kaggle.com/", "desc": "The ultimate playground for data science competitions, datasets, and notebooks."},
            {"title": "Mode Analytics SQL Tutorial", "url": "https://mode.com/sql-tutorial/", "desc": "The absolute best interactive guide to mastering SQL for analytics."},
            {"title": "DataQuest Platforms", "url": "https://www.dataquest.io/", "desc": "Hands-on, browser-based coding platform to learn Python and data science."}
        ],
        "videos": [
            {"title": "Alex The Analyst: Boot Camp", "url": "https://www.youtube.com/playlist?list=PLUaB-1hjhk8G759M2e_TfH3505XWnJb7a", "desc": "Complete visual path covering SQL, Excel, PowerBI, Tableau, and Python."},
            {"title": "Ken Jee: Learn Data Science", "url": "https://www.youtube.com/watch?v=KxryzSO1Fjs", "desc": "A practical, realistic blueprint for breaking into data science."},
            {"title": "Luke Barousse Channel", "url": "https://www.youtube.com/@LukeBarousse", "desc": "Excellent tutorials focusing on data analytics tools, portfolio building, and AI integration."}
        ],
        "articles": [
            {"title": "Towards Data Science", "url": "https://towardsdatascience.com/", "desc": "Popular Medium publication covering concepts, math, and code implementation."},
            {"title": "Python Data Science Handbook", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/", "desc": "Free online version of Jake VanderPlas's essential reference book."},
            {"title": "KDNuggets Blog", "url": "https://www.kdnuggets.com/", "desc": "One of the oldest and most respected blogs on analytics, big data, and data mining."}
        ]
    },
    "Web Developmen": {
        "websites": [
            {"title": "freeCodeCamp.org", "url": "https://www.freecodecamp.org/", "desc": "Thousands of hours of interactive coding challenges and certifications."},
            {"title": "The Odin Project", "url": "https://www.theodinproject.com/", "desc": "Full-stack path using JavaScript or Ruby on Rails, entirely project-based."},
            {"title": "W3Schools Academy", "url": "https://www.w3schools.com/", "desc": "Comprehensive, easy-to-use tutorial website covering basic web programming."}
        ],
        "videos": [
            {"title": "Traversy Media Channel", "url": "https://www.youtube.com/user/TechGuyWeb", "desc": "Amazing crash courses on HTML, CSS, React, Node, and more."},
            {"title": "Kevin Powell: CSS Demystified", "url": "https://www.youtube.com/@KevinPowell", "desc": "The internet's best resource for mastering responsive CSS layouts."},
            {"title": "Web Dev Simplified (Kyle)", "url": "https://www.youtube.com/@WebDevSimplified", "desc": "Clear, practical explanations of web concepts and tools for junior developers."}
        ],
        "articles": [
            {"title": "MDN Web Docs", "url": "https://developer.mozilla.org/", "desc": "The authoritative, standard reference manual for HTML, CSS, and JS."},
            {"title": "JavaScript.info", "url": "https://javascript.info/", "desc": "A deep-dive, modern tutorial covering basic to advanced Javascript concepts."},
            {"title": "CSS-Tricks", "url": "https://css-tricks.com/", "desc": "A goldmine of articles, snippets, and guides on CSS and frontend development."}
        ]
    },
    "Web Development": {
        "websites": [
            {"title": "freeCodeCamp.org", "url": "https://www.freecodecamp.org/", "desc": "Thousands of hours of interactive coding challenges and certifications."},
            {"title": "The Odin Project", "url": "https://www.theodinproject.com/", "desc": "Full-stack path using JavaScript or Ruby on Rails, entirely project-based."},
            {"title": "W3Schools Academy", "url": "https://www.w3schools.com/", "desc": "Comprehensive, easy-to-use tutorial website covering basic web programming."}
        ],
        "videos": [
            {"title": "Traversy Media Channel", "url": "https://www.youtube.com/user/TechGuyWeb", "desc": "Amazing crash courses on HTML, CSS, React, Node, and more."},
            {"title": "Kevin Powell: CSS Demystified", "url": "https://www.youtube.com/@KevinPowell", "desc": "The internet's best resource for mastering responsive CSS layouts."},
            {"title": "Web Dev Simplified (Kyle)", "url": "https://www.youtube.com/@WebDevSimplified", "desc": "Clear, practical explanations of web concepts and tools for junior developers."}
        ],
        "articles": [
            {"title": "MDN Web Docs", "url": "https://developer.mozilla.org/", "desc": "The authoritative, standard reference manual for HTML, CSS, and JS."},
            {"title": "JavaScript.info", "url": "https://javascript.info/", "desc": "A deep-dive, modern tutorial covering basic to advanced Javascript concepts."},
            {"title": "CSS-Tricks", "url": "https://css-tricks.com/", "desc": "A goldmine of articles, snippets, and guides on CSS and frontend development."}
        ]
    },
    "Cybersecurity": {
        "websites": [
            {"title": "TryHackMe Labs", "url": "https://tryhackme.com/", "desc": "Hands-on cyber security training through byte-sized, gamified labs."},
            {"title": "Hack The Box Academy", "url": "https://www.hackthebox.com/", "desc": "An advanced, gamified cyber security training platform to practice hacking."},
            {"title": "PortSwigger Web Academy", "url": "https://portswigger.net/web-security", "desc": "The absolute best place to learn web application hacking for free."},
            {"title": "Cybrary Security Platform", "url": "https://www.cybrary.it/", "desc": "Interactive training, labs, and certification preparation for IT security."}
        ],
        "videos": [
            {"title": "John Hammond Channel", "url": "https://www.youtube.com/@JohnHammond010", "desc": "In-depth walkthroughs of real-world cybersecurity challenges and CTFs."},
            {"title": "NetworkChuck CCNA/Linux", "url": "https://www.youtube.com/@NetworkChuck", "desc": "High-energy tutorials on networking, Linux, and security essentials."},
            {"title": "LiveOverflow: IT Security", "url": "https://www.youtube.com/@LiveOverflow", "desc": "Fascinating educational hacking content, exploring binary exploitation and security concepts."}
        ],
        "articles": [
            {"title": "OWASP Top 10 Risks", "url": "https://owasp.org/www-project-top-ten/", "desc": "The standard awareness document for web application security."},
            {"title": "Krebs on Security", "url": "https://krebsonsecurity.com/", "desc": "In-depth investigative journalism on high-profile security breaches."},
            {"title": "The Hacker News", "url": "https://thehackernews.com/", "desc": "Leading news source for info-security, cybersecurity threats, and technology trends."}
        ]
    },
    "Business": {
        "websites": [
            {"title": "Wharton Foundations", "url": "https://www.coursera.org/specializations/wharton-business-foundations", "desc": "An introduction to marketing, finance, accounting, and operations."},
            {"title": "HubSpot Academy", "url": "https://academy.hubspot.com/", "desc": "Free certifications for inbound marketing, sales, and customer service."},
            {"title": "Google Primer App", "url": "https://www.yourprimer.com/", "desc": "Fast, easy-to-digest business and marketing lessons on your mobile device."}
        ],
        "videos": [
            {"title": "YC Startup School", "url": "https://www.youtube.com/@ycombinator", "desc": "Advice from top startup founders and investors on building products."},
            {"title": "Crash Course Business", "url": "https://www.youtube.com/playlist?list=PL8dPuuaLjXtM5-jWvjTkHkPzV5of1F5wN", "desc": "Fast-paced overview of soft skills, entrepreneurship, and management."},
            {"title": "Slidebean: Startups 101", "url": "https://www.youtube.com/@Slidebean", "desc": "Deconstructs startup metrics, financial modeling, and slide deck pitches."}
        ],
        "articles": [
            {"title": "Harvard Business Review", "url": "https://hbr.org/", "desc": "The premier publication for business research and leadership strategies."},
            {"title": "Investopedia", "url": "https://www.investopedia.com/", "desc": "A comprehensive encyclopedia for financial terms, concepts, and market updates."},
            {"title": "McKinsey & Company Insights", "url": "https://www.mckinsey.com/featured-insights", "desc": "In-depth reports and insights on economics, business strategy, and technology trends."}
        ]
    },
    "Design": {
        "websites": [
            {"title": "Figma Community", "url": "https://www.figma.com/community", "desc": "Templates, UI kits, and design files created by global creators."},
            {"title": "Dribbble Portfolio Showcase", "url": "https://dribbble.com/", "desc": "A showcase for UI/UX designers, illustrators, and graphic artists."},
            {"title": "Awwwards Showcase", "url": "https://www.awwwards.com/", "desc": "Design awards website that recognizes and promotes the best innovative web design."}
        ],
        "videos": [
            {"title": "The Futur with Chris Do", "url": "https://www.youtube.com/@thefutur", "desc": "Business of design, branding, and typography masterclasses."},
            {"title": "Jesse Showalter UI/UX", "url": "https://www.youtube.com/@JesseShowalter", "desc": "Practical UI/UX design workflows, tools, and portfolio tips."},
            {"title": "Satori Graphics Channel", "url": "https://www.youtube.com/@SatoriGraphics", "desc": "High-quality graphic design tutorials focusing on vector illustration and design theory."}
        ],
        "articles": [
            {"title": "Nielsen Norman Group UX", "url": "https://www.nngroup.com/articles/", "desc": "Evidence-based research and guidelines for user experience design."},
            {"title": "UX Collective", "url": "https://uxdesign.cc/", "desc": "A curated publication featuring perspectives on UX, product, and design."},
            {"title": "Smashing Magazine Design", "url": "https://www.smashingmagazine.com/category/design", "desc": "Superb articles and e-books on design principles, UI usability, and design systems."}
        ]
    },
    "Cloud Computing": {
        "websites": [
            {"title": "AWS Skills Builder", "url": "https://explore.skillbuilder.aws/", "desc": "Official free training paths and labs provided directly by Amazon."},
            {"title": "Microsoft Learn for Cloud", "url": "https://learn.microsoft.com/en-us/training/", "desc": "Comprehensive documentation and interactive labs for Azure."},
            {"title": "Google Cloud Skills Boost", "url": "https://www.cloudskillsboost.google/", "desc": "Hands-on learning platform for Google Cloud technologies and architecture."}
        ],
        "videos": [
            {"title": "TechWorld with Nana", "url": "https://www.youtube.com/@TechWorldwithNana", "desc": "Extremely clear visual guides on DevOps, Kubernetes, and Cloud Concepts."},
            {"title": "AWS Practitioner (freeCodeCamp)", "url": "https://www.youtube.com/watch?v=SOTamWGuqXs", "desc": "A complete, free video prep course to pass AWS foundational certs."},
            {"title": "Adrian Cantrill's Cloud Tutorials", "url": "https://www.youtube.com/@CantrillIO", "desc": "Deep, structured tutorials on AWS architecture, networking, and cloud design."}
        ],
        "articles": [
            {"title": "AWS Architecture Center", "url": "https://aws.amazon.com/architecture/", "desc": "Reference blueprints, diagrams, and whitepapers for cloud architecture."},
            {"title": "The Twelve-Factor App", "url": "https://12factor.net/", "desc": "A methodology for building modern, cloud-native software-as-a-service apps."},
            {"title": "Kubernetes Official Blog", "url": "https://kubernetes.io/blog/", "desc": "Official updates, architectures, and stories from the Kubernetes community."}
        ]
    }
}


# Helper functions for UI styling and banners
def inject_custom_css():
    st.markdown("""<style>
/* Import modern, premium fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Apply fonts globally to content elements (skips span/div to avoid breaking Streamlit system icons) */
html, body, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label {
font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
}

/* Block padding adjustments */
.block-container {
padding-top: 1.5rem !important;
padding-bottom: 3rem !important;
max-width: 1300px;
}

/* Premium sidebar styling */
[data-testid="stSidebar"] {
background-color: var(--secondary-background-color);
border-right: 1px solid rgba(128, 128, 128, 0.1);
}

/* Style st.metric cards to look like premium dashboard widgets */
div[data-testid="stMetric"] {
background-color: var(--secondary-background-color);
border: 1px solid rgba(128, 128, 128, 0.15);
border-radius: 16px;
padding: 20px 24px;
box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease, border-color 0.3s ease;
}
div[data-testid="stMetric"]:hover {
transform: translateY(-4px);
box-shadow: 0 10px 30px rgba(52, 152, 219, 0.12);
border-color: rgba(52, 152, 219, 0.5);
}
div[data-testid="stMetricValue"] > div {
font-size: 2.2rem !important;
font-weight: 800 !important;
letter-spacing: -0.5px;
}

/* Expanders styled as modern card wrappers */
div[data-testid="stExpander"] {
background-color: var(--secondary-background-color);
border: 1px solid rgba(128, 128, 128, 0.15) !important;
border-radius: 14px !important;
box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
margin-bottom: 14px;
transition: all 0.2s ease-in-out;
}
div[data-testid="stExpander"]:hover {
border-color: #3498db !important;
box-shadow: 0 6px 16px rgba(52, 152, 219, 0.08);
}

/* Form buttons and CTA links */
div.stButton > button, div.stLinkButton > a {
border-radius: 10px !important;
font-weight: 600 !important;
padding: 10px 20px !important;
transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div.stButton > button:hover, div.stLinkButton > a:hover {
transform: scale(1.02);
box-shadow: 0 4px 15px rgba(52, 152, 219, 0.25) !important;
}

/* Tabs customization */
button[data-baseweb="tab"] {
font-size: 1.05rem !important;
font-weight: 600 !important;
color: #888888;
padding: 12px 20px !important;
transition: color 0.2s ease;
}
button[data-baseweb="tab"][aria-selected="true"] {
color: #3498db !important;
}

/* Sidebar navigation radio styling */
div[data-testid="stSidebarNav"] {
padding-bottom: 2rem;
}
</style>""", unsafe_allow_html=True)

def render_banner(title, subtitle, icon="🎓"):
    st.markdown(f"""<div style="
background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
padding: 35px 30px;
border-radius: 20px;
margin-bottom: 30px;
box-shadow: 0 10px 30px rgba(0,0,0,0.1);
color: white;
position: relative;
overflow: hidden;
">
<!-- Decorative circle overlay -->
<div style="
position: absolute; 
width: 250px; 
height: 250px; 
background: rgba(255,255,255,0.06); 
border-radius: 50%; 
top: -50px; 
right: -50px; 
filter: blur(40px);
"></div>

<div style="display: flex; align-items: center; gap: 20px; position: relative; z-index: 2;">
<div style="
font-size: 3rem; 
background: rgba(255,255,255,0.15); 
width: 70px; 
height: 70px; 
display: flex; 
align-items: center; 
justify-content: center; 
border-radius: 16px;
backdrop-filter: blur(10px);
">{icon}</div>
<div>
<h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; color: white; border: none; padding-bottom: 0;">{title}</h1>
<p style="margin: 6px 0 0 0; opacity: 0.85; font-size: 1.05rem; font-weight: 400; line-height: 1.4;">{subtitle}</p>
</div>
</div>
</div>""", unsafe_allow_html=True)

# Configuration
st.set_page_config(
    page_title="EduPro Intelligence Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply UI custom styles
inject_custom_css()


# DATA LOADING (cached)
@st.cache_data
def load_data():
    """
    Loads and caches the EduPro data tables.
    """
    data_dir = Path("data")
    try:
        profiles = pd.read_csv(data_dir / "learner_profiles_clustered.csv")
        courses = pd.read_csv(data_dir / "courses.csv")
        transactions = pd.read_csv(data_dir / "transactions.csv")
        return profiles, courses, transactions
    except Exception as e:
        correlation_id = str(uuid.uuid4())[:8]
        # Log deep exception context server-side only
        sys.stderr.write(f"[Correlation-ID: {correlation_id}] Failed to load datasets: {str(e)}\n")
        st.error(f"❌ An error occurred while loading application datasets. Please contact support. (Correlation ID: {correlation_id})")
        # Return empty dataframes as fallback to prevent crash
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

profiles, courses, transactions = load_data()

# Check if data exists
if profiles.empty or courses.empty or transactions.empty:
    st.error("EduPro datasets could not be loaded. Please ensure you ran 'python src/generate_data.py', 'python src/feature_engineering.py', and 'python src/clustering.py' first.")
    st.stop()

# SIDEBAR NAVIGATION
st.sidebar.markdown("# 🎓 EduPro Intelligence")
st.sidebar.markdown("Student Segmentation & Recommendations")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Pages:",
    ["🏠 Platform Overview", "👤 Learner Profile Explorer", "📊 Segment Comparison Dashboard", "🎯 Course Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Platform Metrics Badges")
st.sidebar.metric("Total Learners", len(profiles))
st.sidebar.metric("Segment Categories", profiles["Segment"].nunique())
st.sidebar.metric("Available Courses", len(courses))

# PAGE 1 — 🏠 Platform Overview
if page == "🏠 Platform Overview":
    render_banner("Platform Overview Dashboard", "Aggregate enrollment statistics, dynamic segment distributions, and platform spending insights.", "🏠")
    
    # 4 KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Learners Enrolled", len(profiles))
    with col2:
        st.metric("Segment Groups Identified", profiles["Segment"].nunique())
    with col3:
        avg_courses = round(profiles["total_courses"].mean(), 2)
        st.metric("Avg Courses per Learner", f"{avg_courses}")
    with col4:
        avg_spending = round(profiles["total_spending"].mean(), 2)
        st.metric("Avg Spending per Learner", f"${avg_spending}")

    st.markdown("---")
    
    # 2-column chart row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Distribution of Learner Segments")
        segment_counts = profiles["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Learner Count"]
        
        # Color mapping for consistency
        color_palette = {
            "Curious Explorers": "#3498db",
            "Career Climbers": "#2ecc71",
            "Deep Specialists": "#9b59b6",
            "Casual Learners": "#95a5a6"
        }
        
        fig_pie = px.pie(
            segment_counts,
            values="Learner Count",
            names="Segment",
            color="Segment",
            color_discrete_map=color_palette,
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_col2:
        st.subheader("Average Spending per Segment")
        avg_spending_seg = profiles.groupby("Segment")["total_spending"].mean().reset_index()
        avg_spending_seg.columns = ["Segment", "Avg Total Spending ($)"]
        
        fig_bar = px.bar(
            avg_spending_seg,
            x="Segment",
            y="Avg Total Spending ($)",
            color="Segment",
            color_discrete_map=color_palette,
            labels={"Avg Total Spending ($)": "Avg Spending ($)"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# PAGE 2 — 👤 Learner Profile Explorer
elif page == "👤 Learner Profile Explorer":
    render_banner("Learner Profile Explorer", "Detailed learning habits, activity profiles, and category preferences for individual students.", "👤")
    
    # Selection dropdown
    user_list = sorted(profiles["UserID"].tolist())
    selected_user = st.selectbox("Select UserID:", user_list)
    
    # Retrieve user data
    user_data = profiles[profiles["UserID"] == selected_user].iloc[0]
    segment = user_data["Segment"]
    
    # Display segment badge colored by cluster
    if segment == "Curious Explorers":
        st.info(f"Student Segment Classification: **{segment}**")
    elif segment == "Career Climbers":
        st.success(f"Student Segment Classification: **{segment}**")
    elif segment == "Deep Specialists":
        st.warning(f"Student Segment Classification: **{segment}**")
    else:
        st.error(f"Student Segment Classification: **{segment}**")

    # 2-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics & Basic Engagement")
        st.write(f"**Age:** {user_data['Age']}")
        st.write(f"**Gender:** {user_data['Gender']}")
        st.write(f"**Total Courses Enrolled:** {user_data['total_courses']}")
        st.write(f"**Unique Categories Explored:** {user_data['unique_categories']}")
        st.write(f"**Platform Enrollment Duration:** {user_data['enrollment_span_days']} days")
        
    with col2:
        st.subheader("Learning Preferences & Behavior")
        st.write(f"**Preferred Category:** {user_data['preferred_category']}")
        st.write(f"**Preferred Difficulty Level:** {user_data['preferred_level']}")
        st.write(f"**Avg Course Rating Enrolled:** {user_data['avg_rating_enrolled']:.2f} ⭐")
        st.write(f"**Total Spending:** ${user_data['total_spending']:.2f}")
        st.write(f"**Avg Course Cost:** ${user_data['avg_spending']:.2f}")
        
        # Diversity note
        div_score = user_data['diversity_score']
        st.write(f"**Diversity Score:** {div_score:.2f}")
        st.caption("*(0.0 = pure specialist in 1 category, 1.0 = pure generalist)*")
        
        # Depth index percentage
        depth_pct = user_data['learning_depth_index'] * 100
        st.write(f"**Learning Depth Index:** {depth_pct:.1f}%")
        st.progress(user_data['learning_depth_index'])

# PAGE 3 — 📊 Segment Comparison Dashboard
elif page == "📊 Segment Comparison Dashboard":
    render_banner("Segment Comparison Dashboard", "Compare behavioral profiles and statistics across segmented learner groups.", "📊")
    
    # Segment summary table (highlight max values in green)
    st.subheader("Segment Summary Statistics")
    
    summary_df = profiles.groupby("Segment").agg({
        "UserID": "count",
        "total_spending": "mean",
        "total_courses": "mean",
        "diversity_score": "mean",
        "learning_depth_index": "mean"
    }).reset_index()
    
    summary_df.columns = ["Segment", "Learner Count", "Avg Spending ($)", "Avg Courses", "Avg Diversity", "Avg Depth Index"]
    
    # Highlight max values using Pandas Styling (renders in Streamlit)
    styled_df = summary_df.style.highlight_max(
        subset=["Learner Count", "Avg Spending ($)", "Avg Courses", "Avg Diversity", "Avg Depth Index"],
        color="#d4edda"  # soft light green
    ).format({
        "Avg Spending ($)": "{:.2f}",
        "Avg Courses": "{:.2f}",
        "Avg Diversity": "{:.3f}",
        "Avg Depth Index": "{:.3f}"
    })
    
    st.write(styled_df)
    st.caption("*Green cells highlight the maximum value in each metric column.*")
    
    st.markdown("---")
    
    # 2 Plotly charts
    chart_col1, chart_col2 = st.columns(2)
    color_palette = {
        "Curious Explorers": "#3498db",
        "Career Climbers": "#2ecc71",
        "Deep Specialists": "#9b59b6",
        "Casual Learners": "#95a5a6"
    }

    with chart_col1:
        st.subheader("Spending Distribution per Segment")
        fig_box = px.box(
            profiles,
            x="Segment",
            y="total_spending",
            color="Segment",
            color_discrete_map=color_palette,
            labels={"total_spending": "Total Spending ($)"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with chart_col2:
        st.subheader("Average Enrolled Courses per Segment")
        fig_avg_courses = px.bar(
            summary_df,
            x="Segment",
            y="Avg Courses",
            color="Segment",
            color_discrete_map=color_palette,
            labels={"Avg Courses": "Avg Courses count"}
        )
        st.plotly_chart(fig_avg_courses, use_container_width=True)

# PAGE 4 — 🎯 Course Recommendations
elif page == "🎯 Course Recommendations":
    render_banner("Personalized Course Recommendations", "AI-driven local suggestions and real-world web/video/article resources for learning.", "🎯")
    
    # 2-column top row
    top_col1, top_col2 = st.columns(2)
    
    with top_col1:
        # Choose a learner
        user_list = sorted(profiles["UserID"].tolist())
        selected_user = st.selectbox("Choose a Learner:", user_list)
        
    with top_col2:
        # Filter by level
        level_choices = ["Beginner", "Intermediate", "Advanced"]
        selected_levels = st.multiselect("Filter by Course Level:", level_choices, default=level_choices)
        
    # Optional category filter
    categories_choices = sorted(courses["CourseCategory"].unique().tolist())
    selected_categories = st.multiselect("Filter by Course Category (Optional):", categories_choices)
    
    st.markdown("---")
    
    # Fetch learner's segment info
    user_row = profiles[profiles["UserID"] == selected_user].iloc[0]
    learner_segment = user_row["Segment"]
    
    st.info(f"Learner Profile: **{selected_user}** | Segment Classification: **{learner_segment}**")
    
    # Generate and display recommendations
    if not selected_levels:
        st.warning("Please select at least one course level from the filter above.")
    else:
        st.subheader("Top Recommended Courses by Difficulty Level")
        
        # Create tabs dynamically for each selected level
        level_tabs = st.tabs([f"🎓 {lvl} Level" for lvl in selected_levels])
        
        for lvl, tab in zip(selected_levels, level_tabs):
            with tab:
                # Query recommendations specifically for this level!
                recs = get_recommendations(
                    user_id=selected_user,
                    top_n=5,
                    level_filter=[lvl],
                    category_filter=selected_categories if selected_categories else None
                )
                
                if recs.empty:
                    st.info(f"No recommended {lvl} courses matching your filters.")
                else:
                    # Display recommendations using expanders
                    for idx, row in recs.iterrows():
                        course_id = row["CourseID"]
                        category = row["CourseCategory"]
                        level = row["CourseLevel"]
                        rating = row["CourseRating"]
                        score = row["final_score"]
                        
                        # Normalize category for clean UI display
                        clean_category = "Web Development" if category == "Web Developmen" else category
                        
                        expander_title = f"📘 {course_id} | {clean_category} | ⭐ {rating:.2f}"
                        with st.expander(expander_title):
                            st.write(f"**Course ID:** {course_id}")
                            st.write(f"**Domain Category:** {clean_category}")
                            st.write(f"**Difficulty Level:** {level}")
                            st.write(f"**Course Rating:** {rating:.2f} ⭐")
                            st.write(f"**Relevance Recommendation Score:** {score:.3f}")
                            st.progress(score)
                            
                            # Fetch curated real-world internet resources
                            resources = REAL_WORLD_RESOURCES.get(category) or REAL_WORLD_RESOURCES.get(clean_category)
                            if resources:
                                st.markdown("---")
                                st.markdown(f"#### 🌐 Real-World Learning Resources for *{clean_category}*")
                                res_col1, res_col2, res_col3 = st.columns(3)
                                
                                with res_col1:
                                    st.markdown("**💻 Top Platforms & Websites**")
                                    for item in resources["websites"]:
                                        st.link_button(item['title'], item['url'])
                                        st.caption(item['desc'])
                                        st.write("")
                                        
                                with res_col2:
                                    st.markdown("**🎥 YouTube Playlists & Videos**")
                                    for item in resources["videos"]:
                                        st.link_button(item['title'], item['url'])
                                        st.caption(item['desc'])
                                        st.write("")
                                        
                                with res_col3:
                                    st.markdown("**📄 Articles & Reference Guides**")
                                    for item in resources["articles"]:
                                        st.link_button(item['title'], item['url'])
                                        st.caption(item['desc'])
                                        st.write("")
