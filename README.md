# **Retail ActionAI: Next-Best-Offer Classifier**

This project is part of **Machine Learning: Fundamentals and Applications (AAI-510)** course in the [Applied Artificial Intelligence M.S. program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at the **University of San Diego**.

> **Project Status:** Completed



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

## **Dataset**

* **Name:** Instacart Market Basket Analysis  
* **Link:** <https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis>  
* **Size:** ~3.4 M orders, 206 K customers, 30 K products  
* **Schema:** Orders, order-product lines, product-aisle-department metadata.  
* **Usage:** Joined on `order_id` and `product_id` to build a shopper-level panel for modeling.

## **Methods Used**

* Data Wrangling & Multi-table Joins  
* Exploratory Data Analysis (EDA) & Visualization  
* Feature Engineering (reorder ratios, gap ratios, ..etc)  
* Handling Imbalanced Classes  
* Multiple models (Logistic Regression, RF, XGBoost, LightGBM and imbalanced-learn models)  
* SHAP for Feature Importance  


## **Technologies**

| Area | Stack |
|------|-------|
| **Core Language** | Python |
| **Modeling** |  RF, XGBoost, LightGBM and imbalanced-learn models |
| **Explainability** | SHAP |

## **Data Wrangling**

This step polishes the raw Instacart tables so they join cleanly, fit in memory, and feed the model reliable inputs. Below is the game-plan and what each sub-section tackles:

* **Data types:** keep integers small and consistent by:
  * Cast join keys to compact ints (`int32`, `int16`, `int8`) so merges don't up-cast to `int64`.
  * Down-cast the one float column (`days_since_prior_order`) to `float32` to save RAM.

* **Remove duplicates:**  This will guarantee one factual row per record: 
  * `df.drop_duplicates()` on every table—cheap insurance against accidental repeats.
  * Leaner tables speed up the big merge and later group-bys.

* **Joining the tables**: build a single, product-rich history view: 
  * Merge `products `&rarr;` aisles `&rarr; `departments` into a **product lookup**.  
  * Join that lookup to `order_products__prior`, then attach the `orders` header. The result **`prior_full`** has one row per product in every past basket, plus aisle, department, order time, etc.   

* **Missing data**: Here we will plug the only nulls to keep the column fully numeric.
  * `days_since_prior_order` is missing on a shopper's first-ever order. we'll 
  fill with **0** (no prior gap) and convert to `int16`.

* **Object to category conversion:** We'll trim memory for small text columns by: 
  *  Convert low-cardinality strings (`aisle`, `department`, `eval_set`) to `category`.  
  * Leave `product_name` as plain text; ~50 K unique values mean categorising it bloats rather than helps. 

## **Features Engineering & Exploratory Data Analysis (EDA)**

This section answers the big “what, when, and how often” questions hidden in our cleaned Instacart history table.  
W'll walk through four lenses—shopper, product, time, and user-product to guide feature design and model focus.

* **Shopper profile**:  
  * **Avg. order size**  
  * **Avg. number of reorders per cart**  
  * **Avg. days between orders**  
  * **Avg. order hour** and **day of week**  
  * How many total orders does each customer place?

* **Product snapshot**:  
  * **Total orders per product**  
  * **Times product was reordered** and **reorder probability**  
  * Top aisles and departments by volume  
  * Which products are frequently ordered or usually reordered?  
  * Ratio of reordered versus newly ordered items.

* **Temporal patterns**:
  * When do people place their orders? (hour-of-day and day-of-week heatmaps)  
  * Cart size distribution over time.

* **User x Product dynamics**:  
  * **Avg. add-to-cart position** (normalized)  
  * **Times a user bought a product** and **reorder frequency**  
  * Orders and days since the last purchase of that item  
  * Streaks of consecutive product orders.

* **Data health checks**:  
  * Recency/frequency histograms to spot long tails.  
  * Flag outliers or ultra-rare categories that could skew the model.

By the end of this EDA pass we'll know which signals are strongest, where class imbalance or sparsity lives, and which features to engineer first for our next-best-offer model.


## **Modeling & Decision Engine - Notebook Plan**  

This notebook takes the cleaned **`candidates`** matrix and turns it into a calibrated “buy-next” model plus the business rules that convert probabilities into offers.  
No code below—just the play-by-play of what each section will do.

---

**Quick Feature Checks**

* **Pearson correlation** on numeric columns  
* **Mutual information** on numeric + categorical columns &rarr; rank useful predictors; drop MI &asymp; 0 if we need to slim width.  
* Result: a leaner, high-signal feature set.

**Train/Test Split & CatBoost Encoding**

* 80 / 20 split, stratified on `bought_next`.  
* Low-cardinality categoricals (`aisle`, `department`, `peak_dow`) encoded with **CatBoostEncoder** (target-aware, leakage-safe).  
* Numeric features stay as-is; long-tail columns were winsorised earlier.

**Modeling**

Train five imbalance-aware tree ensembles:  

| model | imbalance trick |
|-------|-----------------|
| Random Forest | `class_weight='balanced'` |
| LightGBM | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight &asymp; 15` |
| Balanced Bagging (LightGBM base) | under-samples negatives in each bag |
| Easy Ensemble (LightGBM base) | AdaBoost on balanced bags |

Metrics collected: **ROC-AUC**, F1 and recall.  

**Next-Best-Offer Logic**

*For each user-product candidate in production:*  

* Get `p_buy` from the calibrated model.  
* Apply business rule thresholds:
  - If `p_buy > 0.8` &rarr; assign **"None"** (likely buyer; no incentive needed).
  - If `0.3 < p_buy ≤ 0.8`:
    - If user is **engaged** (`user_total_orders > 10` and `avg_basket_size > 5`) &rarr; assign **"Upsell"**.
    - Else &rarr; assign **"Coupon"**.
  - If `p_buy ≤ 0.3` &rarr; assign **"Coupon"** (low intent, needs incentive).
* This rule-based approach blends behavioral segmentation with predictive modeling.

**Save Artifacts**

* Calibrated model (`.pkl`)  
* CatBoost encoder (`.pkl`)  
* Final feature list  

With these pieces we can deploy a real-time API that scores the active cart, runs the EV maths, and surfaces the smartest next-best offer.


## Discussion & Conclusions

**Problem Recap**

Our goal was to answer a key question for every user-product pair:

> *“Will this person put this item in their very next order?”*

By accurately predicting this, we aim to **spend promotional budget only where it influences behavior** — avoiding unnecessary discounts for likely buyers and targeting incentives to those on the edge.


**What We Achieved**

- Built a calibrated **buy-next model** that predicts purchase intent at the user-product level.
- Created a rule-based **Next-Best-Offer engine** that classifies each row into `Coupon`, `Upsell`, or `None`.
- Achieved a balanced NBO distribution:
  - ~70% of offers need no action
  - ~22% receive a coupon
  - ~7% are good candidates for upsell

This logic allows us to act efficiently and scalably, aligning offers with user behavior.


**Recommendations**

- **Deploy as a batch pipeline** to power CRM, app, or email personalization.
- **Use real-time scoring** to refresh offers when carts change.
- **Incorporate business data** like product margins, time-based recency, and customer segments for smarter targeting.
- **Enhance with LLMs** for personalized explanations, dynamic copy, or segment tagging.


With this system in place, Retail Action-AI can intelligently deliver the right offer, to the right shopper, at the right time — maximizing return on every promotion dollar.


## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## **Acknowledgments**

Thank you to Professor Wesley Pasfield for your guidance and support throughout this project/class. Your insights have been greatly appreciated.



