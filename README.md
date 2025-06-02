# **Retail ActionAI: Next-Best-Offer Classifier**

This project is part of **Machine Learning: Fundamentals and Applications (AAI-510)** in the [Applied Artificial Intelligence M.S. program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at the **University of San Diego**.

> **Project Status:** Ongoing



## **Introduction**

Grocery chains lose revenue every day by blasting the same coupons to everyone or by waiting too long to re-engage lapsed shoppers.  
**Retail ActionAI** turns raw basket history into a simple answer to one question:

*“What's the smartest action we can take for this customer right now?”*  

The app classifies each shopper into one of three actionable buckets—**Send Coupon**, **Upsell**, or **No Action**—so marketing teams can spend less and convert more.



## **Objectives**

1. **Data-Driven Segmentation**: Translate Instacart basket logs into shopper-level features (recency, frequency, basket size, reorder ratio).  
2. **Multi-Class Classification**: Train multiple models (Logistic Regression, RF, XGBoost, LightGBM and imbalanced-learn models) that predict the next-best action with strong recall on high-value segments.  
3. **Explainability First**: Utilize SHAP explanations so analysts can see *why* a customer was routed to a specific action.  
4. **App-Ready Delivery**: Wrap the model in a Streamlit interface that anyone can use—upload a CSV or type a customer ID and get an instant recommendation.

---

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



