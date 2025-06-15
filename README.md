 Risk Analytics & Predictive Modeling – Car Insurance (South Africa)

This project explores car insurance data to identify low-risk segments and optimize premium pricing using exploratory data analysis (EDA), statistical modeling, and machine learning. The workflow is version-controlled using Git and DVC to ensure reproducibility and compliance with industry standards.
📌 Objectives

    Understand key risk factors in insurance claims.

    Identify low-risk customers for optimized marketing and pricing.

    Apply statistical and machine learning techniques to predict risk.

    Implement version-controlled, reproducible analysis pipelines with DVC.

📁 Project Structure

.
├── data/                        # Raw and processed data
├── notebooks/                  # Jupyter notebooks for EDA & modeling
├── src/                        # Scripts for preprocessing and modeling
├── .dvc/                       # DVC config and tracked files
├── .github/workflows/ci.yml   # GitHub Actions workflow
├── requirements.txt            # Python dependencies
├── README.md                   # Project description
└── dvc.yaml                    # DVC pipeline definition (optional)

🧪 Exploratory Data Analysis (EDA)

EDA helped uncover:

    🧮 Loss Ratio Analysis: Claims vs Premium across regions and customer segments.

    📊 Distributions & Outliers: Detected using histograms and boxplots.

    🗺️ Trends by Geography: Provinces with higher risk profiles.

    ⏱️ Temporal Trends: Seasonality in claims over 18 months.

    🚘 High-risk Vehicle Types: Certain models are consistently riskier.

All analysis is documented in:
📓 notebooks/1EDA_Stats.ipynb
📦 DVC Setup

To ensure reproducibility and data versioning:
Step-by-Step Setup

    Initialize DVC

dvc init

Add Data

dvc add data/MachineLearningRating_v3.txt

Configure Remote Storage

dvc remote add -d localstorage /path/to/your/storage

Push Data

dvc push

Track Changes with Git

    git add data/MachineLearningRating_v3.txt.dvc .dvc/config
    git commit -m "Track data with DVC"

⚙️ CI/CD with GitHub Actions

    Automatically runs linting and tests on every push to main.

    Ensures environment consistency with Python 3.10 and pinned dependencies.

    CI file: .github/workflows/ci.yml

📚 Requirements

Install dependencies with:

pip install -r requirements.txt

✅ Deliverables

    EDA Notebook

    Statistical Summary

    Predictive Model

    DVC-tracked data

    GitHub Actions Pipeline

📅 Timeline
Milestone	Date
Challenge Start	11 June 2025
Interim Report	13 June 2025
Final Submission	17 June 2025
👥 Team

    Yohannes Alemu – Marketing Analytics Enginee
