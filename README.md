# Credit Risk Probability Model for Alternative Data

## An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model


## Overview

As an Analytics Engineer at Bati Bank, this project focuses on developing an innovative Credit Scoring Model. This model will leverage alternative data from a successful e-commerce platform to enable a "buy-now-pay-later" service for customers. The core challenge is to transform behavioral data, specifically Recency, Frequency, and Monetary (RFM) patterns, into a reliable proxy for credit risk. This allows for the development and deployment of a predictive model that assigns a risk probability score, informs loan approvals, and determines optimal loan terms.

### Business Need

EthioMart, a leading financial service provider, is partnering with an e-commerce company to offer buy-now-pay-later services. Traditional credit scoring relies heavily on historical loan performance. However, this project innovates by utilizing alternative data – customer transaction patterns – to assess creditworthiness. The goal is to:

1.  **Define a proxy variable** to categorize users as high-risk (bad) or low-risk (good) when a direct "default" label is unavailable.
2.  **Select observable features** that are strong predictors of this defined default variable.
3.  **Develop a model** that assigns a credit risk probability to new customers.
4.  **Create a model** that derives a credit score from these risk probability estimates.
5.  **Build a model** to predict the optimal loan amount and duration.

This solution will empower Bati Bank to expand its lending services to a broader customer base, leveraging modern data analytics for informed financial decisions.

## Data and Features

The dataset for this challenge is the [Xente Challenge dataset](https://www.kaggle.com/datasets/atwine/xente-challenge).

**Data Fields:**

  * **`TransactionId`**: Unique transaction identifier.
  * **`BatchId`**: Unique number for a batch of transactions.
  * **`AccountId`**: Unique customer identifier on the platform.
  * **`SubscriptionId`**: Unique customer subscription identifier.
  * **`CustomerId`**: Unique identifier linked to Account.
  * **`CurrencyCode`**: Country currency.
  * **`CountryCode`**: Numerical geographical code of country.
  * **`ProviderId`**: Source provider of the item bought.
  * **`ProductId`**: Item name being bought.
  * **`ProductCategory`**: Broader product categories.
  * **`ChannelId`**: Channel used (web, Android, IOS, pay later, checkout).
  * **`Amount`**: Value of the transaction (positive for debits, negative for credits).
  * **`Value`**: Absolute value of the amount.
  * **`TransactionStartTime`**: Transaction start time.
  * **`PricingStrategy`**: Xente's pricing structure category for merchants.
  * **`FraudResult`**: Fraud status of transaction (1 - yes or 0 - No).

## Learning Outcomes

### Skills

  * Advanced use of scikit-learn
  * Feature Engineering
  * ML Model building and fine-tuning
  * CI/CD deployment of ML models
  * Python logging
  * Unit testing
  * Model management
  * MLOps with CML and MLflow

In the "Credit Scoring Business Understanding" section of this README, answer the following:

#### Credit Scoring Business Understanding

1.  **How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?**
    The Basel II Accord mandates that financial institutions hold adequate capital against the risks they face, including credit risk. This necessitates robust, quantifiable risk measurement frameworks. For credit scoring models, this translates into a strong need for interpretability and thorough documentation. Regulators require banks to justify how their models arrive at credit decisions. An interpretable model allows the bank to explain to regulators (and customers) why a particular credit decision was made, demonstrating adherence to fair lending practices and risk management principles. Well-documented models ensure transparency, reproducibility, and auditability, which are critical for regulatory compliance and internal governance. It also facilitates model validation and monitoring, ensuring that the model remains accurate and reliable over time.

2.  **Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?**
    Creating a proxy variable for "default" is necessary because direct default labels, especially in new or alternative data contexts like e-commerce transactions, are often unavailable. A proxy allows us to approximate the concept of credit risk using observable behavioral data (like RFM). For instance, a customer with very low transaction frequency and monetary value, and high recency of their last purchase, might be "proxied" as high-risk or disengaged.
    However, using a proxy variable introduces several business risks:

      * **Proxy Error:** The proxy may not perfectly capture actual default behavior, leading to misclassification (e.g., labeling a low-risk customer as high-risk, or vice versa).
      * **Incorrect Business Decisions:** If the proxy is flawed, loan approval decisions based on its predictions could be inaccurate, leading to either unnecessary rejection of creditworthy customers (lost revenue) or approving high-risk customers who later default (financial loss).
      * **Bias Introduction:** The proxy definition itself might inadvertently introduce biases if the "disengagement" patterns correlate with demographic or socioeconomic factors in an undesirable way.
      * **Loss of Trust:** Frequent false positives (denying credit to good customers) can lead to customer dissatisfaction and damage the bank's reputation.
      * **Regulatory Scrutiny:** Regulators may question the validity and fairness of a proxy-based credit scoring system, especially if it leads to disparate impact on certain groups.

3.  **What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?**
    In a regulated financial context, the choice between simple and complex models involves significant trade-offs:

      * **Simple Models (e.g., Logistic Regression with WoE):**
          * **Pros:** Highly interpretable (coefficients directly show feature impact), easier to understand and explain to regulators/stakeholders, lower computational cost, often more stable with concept drift, easier to audit and validate. Weight of Evidence (WoE) transformation further enhances interpretability by showing the monotonic relationship between features and the target.
          * **Cons:** May have lower predictive power, especially for complex, non-linear relationships in data. May not capture all nuances of risk.
      * **Complex Models (e.g., Gradient Boosting, Random Forest):**
          * **Pros:** Often achieve higher predictive accuracy by capturing intricate non-linear relationships and interactions in the data. Can handle large datasets and a high number of features effectively.
          * **Cons:** Less interpretable ("black box" nature), making it challenging to explain individual predictions and satisfy regulatory requirements for model transparency. Higher computational cost for training and inference. More susceptible to overfitting if not tuned properly. Debugging and auditing can be more difficult.
            The trade-off is often between **interpretability/explainability and predictive performance**. In highly regulated industries like finance, interpretability is often prioritized due to regulatory demands and the need for clear accountability. While complex models might offer marginal performance gains, the regulatory burden and operational challenges associated with their "black-box" nature often lead to a preference for simpler, more transparent models, or the use of explainability tools (like SHAP/LIME) to shed light on complex model decisions.

-----
### Knowledge

  * Reasoning with business context
  * Data exploration
  * Predictive analysis
  * Machine learning
  * Hyperparameter tuning
  * Model comparison & selection

### Communication

  * Reporting on statistically complex issues


```
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                    # Raw data goes here
│   └── processed/              # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb           # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Script for feature engineering
│   ├── train.py                # Script for model training
│   ├── predict.py              # Script for inference
│   └── api/
│       ├── main.py             # FastAPI application
│       └── pydantic_models.py  # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

### Task 1: Understanding Credit Risk

Focus on understanding the concept of Credit Risk, particularly in the context of the Basel II Capital Accord.

**Key References:**
 * [🖥️ Updated Homepage Screenshot](image/Updated_homepage.jpg) 
  * [📊 Project Architecture Diagram](image/Project%20Architecture.drawio.png) 
* [📌 Risk Officer - Credit Risk](image/687474~1.PNG)  
  
    
      * [📈 Severity Feature Importance](image/severity_feature_importance.png)

 
  

### Task 2: Exploratory Data Analysis (EDA)

Explore the dataset to uncover patterns, identify data quality issues, and form hypotheses to guide feature engineering. All exploratory work should be done in `notebooks/1.0-eda.ipynb`.

**Steps:**

1.  **Overview of the Data:** Understand dataset structure (rows, columns, data types).
2.  **Summary Statistics:** Analyze central tendency, dispersion, and shape of distributions.
3.  **Distribution of Numerical Features:** Visualize distributions (histograms, density plots) to identify patterns, skewness, outliers.
4.  **Distribution of Categorical Features:** Analyze frequency and variability of categories.
5.  **Correlation Analysis:** Understand relationships between numerical features (heatmap).
6.  **Identifying Missing Values:** Determine missing data and plan imputation strategies.
7.  **Outlier Detection:** Use box plots to identify outliers.

### Task 3: Feature Engineering

Build a robust, automated, and reproducible data processing script (`src/data_processing.py`) that transforms raw data into a model-ready format. Use `sklearn.pipeline.Pipeline` to chain transformations.

**Steps:**

1.  **Create Aggregate Features:**
      * Total Transaction Amount, Average Transaction Amount, Transaction Count, Standard Deviation of Transaction Amounts per customer.
2.  **Extract Features:**
      * Transaction Hour, Day, Month, Year.
3.  **Encode Categorical Variables:**
      * One-Hot Encoding, Label Encoding.
4.  **Handle Missing Values:**
      * Imputation (mean, median, mode, KNN), or removal.
5.  **Normalize/Standardize Numerical Features:**
      * Normalization (scales to [0, 1]), Standardization (scales to mean 0, std 1).
6.  **Feature Engineering using `xverse` and `woe` libraries:** Explore advanced techniques like Weight of Evidence (WoE) and Information Value (IV).
      * [xverse library](https://pypi.org/project/xverse/)
      * [woe library](https://pypi.org/project/woe/)
      * [Weight of Evidence (WOE) and Information Value (IV) Explained](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)

### Task 4: Proxy Target Variable Engineering

Create a "credit risk" column by programmatically identifying a group of "disengaged" customers as high-risk proxies. High-risk groups are those with a high likelihood of default.

**Steps:**

1.  **Calculate RFM Metrics:** For each `CustomerId`, calculate Recency, Frequency, and Monetary (RFM) values from transaction history, using a defined snapshot date.
2.  **Cluster Customers:** Use K-Means clustering (with `random_state` for reproducibility) to segment customers into 3 distinct groups based on scaled RFM profiles.
3.  **Define and Assign the "High-Risk" Label:** Analyze clusters to identify the least engaged/highest-risk segment (low frequency, low monetary value). Create a new binary target column `is_high_risk` (1 for high-risk, 0 for others).
4.  **Integrate the Target Variable:** Merge `is_high_risk` back into the main processed dataset for model training.

### Task 5: Model Training and Tracking

Develop a structured model training process with experiment tracking, model versioning, and unit testing. Add `mlflow` and `pytest` to `requirements.txt`.

**Steps:**

1.  **Model Selection and Training:**
      * Split data into training and testing sets.
      * Choose at least two models from: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting Machines (GBM).
      * Train the chosen models.
2.  **Hyperparameter Tuning:** Improve performance using Grid Search or Random Search.
3.  **Model Evaluation:** Assess model performance using:
      * Accuracy, Precision, Recall (Sensitivity), F1 Score, ROC-AUC.
4.  **MLflow Model Registry:** Identify your best model and register it in the MLflow Model Registry.
5.  **Unit Tests:** Write at least two unit tests for a helper function in `tests/test_data_processing.py`.

### Task 6: Model Deployment and Continuous Integration

Package the trained model into a containerized API and set up a CI/CD pipeline. Add `fastapi`, `uvicorn`, and a linter (`flake8` or `black`) to `requirements.txt`.

**Steps:**

1.  **Create the API (`src/api/main.py`):**
      * Build a REST API using FastAPI.
      * Load your best model from the MLflow registry.
      * Create a `/predict` endpoint that accepts new customer data and returns risk probability.
      * Use Pydantic models (`src/api/pydantic_models.py`) for data validation.
2.  **Containerize the Service:**
      * Write a `Dockerfile` to set up the environment and run the FastAPI application.
      * Write a `docker-compose.yml` file for easy service build and execution.
3.  **Configure CI/CD (`.github/workflows/ci.yml`):**
      * Create a GitHub Actions workflow that triggers on every push to the main branch.
      * Include steps to run a code linter (e.g., `flake8`) and `pytest` for unit tests.
      * The build must fail if linter or tests fail.

## Tutorials Schedule

  * **Wednesday (Morning)**: Introduction to the challenge, Introduction to Credit Risk Analysis and Modeling.
  * **Thursday (Morning)**: Feature Engineering, Weight of Evidence (WoE), and Information Value (IV).
  * **Thursday (Afternoon)**: Model Training, Hyperparameter Tuning, and Evaluation.
  * **Friday (Morning)**: Model Serving and Deployment with Docker.
  * **Friday (Afternoon)**: Q\&A.

## Deliverables

### Interim Submission (Sunday, June 29, 2025)

  * Link to your GitHub repository showing work for Task 1 and progress in other tasks.
  * A 1-2 page PDF review report of your reading and understanding of Task 1 and progress.

### Final Submission (Tuesday, July 01, 2025)

  * A blog post entry (e.g., on Medium) or a PDF report, outlining your process and exploration results. Focus on data, model selection, and performance after fine-tuning.
  * Link to your GitHub code, including screenshots demonstrating implemented features and results.

## References

### Credit Risk

  * [Investopedia: Credit Risk](https://www.investopedia.com/terms/c/creditrisk.asp)
  * [Investopedia: Credit Spread](https://investopedia.com/terms/c/creditspread.asp)
  * [ClearTax: Credit Risk Glossary](https://cleartax.in/glossary/credit-risk/)
  * [Corporate Finance Institute: Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
  * [Risk Officer: Credit Risk](https://www.risk-officer.com/Credit_Risk.htm)
  * **Publications (Ethiopian Context):**
      * [Credit Risk Determinants in Selected Ethiopian Commercial Banks: A Panel Data Analysis](https://drive.google.com/drive/folders/1pAXmJ_SI46D4Ex-nV0pDGvpxa7HD5erW?usp=drive_link)
      * [Factors Affecting Credit Risk Exposure of Commercial Banks in Ethiopia: An Empirical Analysis](https://drive.google.com/drive/folders/1pAXmJ_SI46D4Ex-nV0pDGvpxa7HD5erW?usp=drive_link)
      * [Credit Risk Analysis of Ethiopian Banks: A Fixed Effect Panel Data Model.](https://drive.google.com/drive/folders/1pAXmJ_SI46D4Ex-nV0pDGvpxa7HD5erW?usp=drive_link)
  * [Scorecard Development](https://shichen.name/scorecard/)

### MLOps

  * [Auto-sklearn Documentation](https://www.google.com/search?q=https://automl.github.io/auto-sklearn/main/)
  * [Hyperparameter Optimization with Random Search and Grid Search](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)
  * [Tune Hyperparameters with GridSearchCV](https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/)

### Kaggle Kernels

  * [Xente Challenge Dataset on Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge)

### Feature Engineering in Credit Scoring

  * [xverse library](https://pypi.org/project/xverse/)
  * [woe library](https://pypi.org/project/woe/)
  * [JGFuentesC/woe\_credit\_scoring GitHub](https://github.com/JGFuentesC/woe_credit_scoring)
  * [Weight of Evidence (WOE) and Information Value (IV) Explained](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
  * [Scorecard Development](https://shichen.name/scorecard/)

### Related Optional References

  * [Credit Risk Determinants in Selected Ethiopian Commercial Banks: A Panel Data Analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8860138/)
  * [Factors Affecting Credit Risk Exposure of Commercial Banks in Ethiopia: An Empirical Analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8860138/)
  * [Credit Risk Analysis of Ethiopian Banks: A Fixed Effect Panel Data Model.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8860138/)

-----

## What the Images Show (Credit Risk Probability Model for Alternative Data)

Based on a general search for "Credit Risk Probability Model for Alternative Data" images, the visuals typically illustrate the following:

  * **Data Integration:** Diagrams showing the flow of various "alternative data" sources (e-commerce transactions, social media, telecom data, utility payments, web Browse behavior) being collected and integrated with traditional financial data.
  * **Feature Engineering:** Visuals often depict the transformation of raw, unstructured alternative data into structured features suitable for modeling. This might include RFM calculation, aggregation, and encoding.
  * **Credit Scoring Pipeline:** Flowcharts demonstrating the end-to-end process: data collection -\> preprocessing -\> feature engineering -\> model training -\> model evaluation -\> model deployment (e.g., via API) -\> monitoring.
  * **Model Architectures:** Simplified representations of machine learning models (e.g., a "black box" for complex models, or a decision tree structure for more interpretable ones).
  * **Risk Scorecards/Distributions:** Graphs showing the distribution of credit scores or probability of default, often with thresholds to distinguish between good and bad borrowers.
  * **Interpretability Visuals:** Conceptual images of SHAP or LIME plots, illustrating how feature contributions are explained for individual predictions or overall model behavior.
  * **Deployment & MLOps:** Icons or diagrams representing Docker containers, cloud platforms (AWS, Azure, GCP), CI/CD pipelines (GitHub Actions, GitLab CI), and MLflow dashboards for tracking experiments and managing models.
  * **Business Impact:** Visualizations highlighting the benefits such as reduced default rates, increased loan approvals for underserved populations, and improved financial inclusion.

These images collectively depict the complex yet systematic process of leveraging non-traditional data to build robust and transparent credit risk assessment systems, emphasizing the importance of data pipelines, advanced analytics, and MLOps practices.