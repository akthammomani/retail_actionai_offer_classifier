# **Retail ActionAI: Next-Best-Offer Classifier**

This project is part of **Machine Learning: Fundamentals and Applications (AAI-510)** course in the [Applied Artificial Intelligence M.S. program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at the **University of San Diego**.

> **Project Status:** Ongoing



## **Introduction**

**Problem statement**

Retail ActionAI has one clear business goal: **spend every promotion dollar where it actually changes behaviour.**  
Using Instacart's historical baskets, we want to answer—**for each shopper-item pair right now**:

> *“Will this person put this item in their very next order?”*

If the answer is **very likely**, a coupon is wasted.  
If it's **unlikely but within reach**, a smart offer—like a discount or upsell—might win the sale.  
So the core technical task is a **"buy-next" prediction** at the *(user, product)* level.

**Justification for this formulation**

* **Granular ROI**: Modeling individual user-product pairs helps us apply offers with precision.
* **Behavior-Aware Offers**: Based on user engagement and purchase intent, we tailor offers (coupon, upsell, or none).
* **Operational Flexibility**: The business layer can evolve rules (e.g., budget caps, product exclusions) without retraining the model.

**Proposed approach**

| Layer | Role |
|-------|------|
| **Feature Engineering** | Turn raw basket logs into rich shopper, product, time, and user-product features |
| **Binary Classifier** | Predict `p_buy = P(item appears in shopper's next order)` |
| **NBO Assignment Logic** | Use `p_buy` and behavioral thresholds to choose between **"None"**, **"Coupon"**, or **"Upsell"** |


**Notebook roadmap** 

| Stage | What We Do | Key Artefacts |
|-------|------------|---------------|
| **Data Wrangling** | Load Instacart data, optimize dtypes, join look-ups &rarr; `prior_full` | Cleaned basket log |
| **Feature Engineering** | Create:<br>• Shopper features<br>• Product features<br>• Time features<br>• User x Product interactions | 4 feature blocks |
| **Modeling Dataset** | Build labeled matrix with `bought_next`, clip outliers, balance classes | `candidates` dataset |
| **Model Training + NBO Engine** | Calibrate a model, score each row, apply NBO rules | Model + business logic layer |

With a clean, feature-rich dataset and a well-framed target (`bought_next`), we're ready to build the engine that turns raw basket data into profit-maximising next-best offers (NBO).


## **Methods Used**

* Data Wrangling & Multi-table Joins  
* Exploratory Data Analysis (EDA) & Visualization  
* Feature Engineering (reorder ratios, gap ratios, ..etc)  
* Handling Imbalanced Classes  
* Multiple models (Logistic Regression, RF, XGBoost, LightGBM and imbalanced-learn models)  
* Hyperparameter Tuning with **Optuna**  
* SHAP for Feature Importance  
* Streamlit App Development & Deployment  



## **Technologies**

| Area | Stack |
|------|-------|
| **Core Language** | Python |
| **Modeling** | Logistic Regression, RF, XGBoost, LightGBM and imbalanced-learn models |
| **Tuning** | Optuna |
| **Explainability** | SHAP |
| **App** | Streamlit + a touch of HTML/CSS for polish |
| **DevOps** | GitHub, GitHub Actions (CI linting), Streamlit Cloud |



## **Repository Contents**




## **Key Features**

* **One-Click Recommendation**: Drop in customer stats or a bulk CSV and see the recommended action.  
* **Transparent Explanations**: Top SHAP factors displayed so marketers know *why* a shopper was flagged.  
* **Batch or Real-Time**: CLI scripts support nightly scoring; the Streamlit app handles ad-hoc what-ifs (TBD)
* **Lightweight Footprint**: Whole pipeline runs on a modest laptop; Cloud deploy is free-tier-friendly (TBD)



## **How It Works**

1. **Join & Clean**: Merge six Instacart tables (`orders`, `order_products__prior`, etc.) on `order_id` and `product_id`.  
2. **Engineer Features**: Compute recency, frequency, avg basket size, reorder ratio, and gap ratio per shopper.  
3. **Label**: Rule-based triage into **Send Coupon**, **Upsell**, or **No Action** using recency and basket metrics.  
4. **Model**: Train multiple models: Logistic Regression, Random Forest, XGBoost, LightGBM.  
5. **Explain**: Generate SHAP values to highlight drivers.  
6. **Serve**: Expose predictions through a Streamlit web UI.



## **Dataset**

* **Name:** Instacart Market Basket Analysis  
* **Link:** <https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis>  
* **Size:** ~3.4 M orders, 206 K customers, 30 K products  
* **Schema:** Orders, order-product lines, product-aisle-department metadata.  
* **Usage:** Joined on `order_id` and `product_id` to build a shopper-level panel for modeling.



## **Future Work**

* Incorporate basket dollar values when available to predict incremental revenue uplift.  
* Experiment with sequence models (Transformer, GRU) for fine-grained next-action timing.  
* Add an **A/B simulation module** to estimate ROI of each action class.  
* Containerize with Docker and add a REST endpoint.


## **Installation**


## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## **Acknowledgments**

Thank you to Professor Wesley Pasfield for your guidance and support throughout this project/class. Your insights have been greatly appreciated.



