# ✈️ Flight Delay Analytics & Prediction

A data-driven machine learning project that analyzes flight delays across airlines, airports, months, and delay causes, and builds a **Random Forest classification model** to identify flight groups with significant delay rates.

The project combines **exploratory data analysis, feature engineering, class-imbalance handling, machine learning, and performance evaluation** to understand the major factors behind flight delays and predict whether an airport-flight group is likely to experience a significant delay rate.

---

## 📌 Project Overview

Flight delays are influenced by several operational and external factors, including:

- Airline/carrier issues
- Weather conditions
- National Airspace System (NAS) congestion
- Security-related delays
- Late arrival of aircraft
- Seasonal and airport-specific effects

This project analyzes **179,338 flight records** covering **2015–2023**, involving multiple airlines and hundreds of airports.

The analysis is divided into two major machine learning tasks:

1. **Classification** - Predict whether a flight group has a significant delay rate.
2. **Regression** - The notebook includes a Random Forest regression section intended to estimate total arrival delay in minutes.

---

## 🎯 Objectives

The main objectives of the project are:

- Analyze historical flight delay patterns.
- Identify seasonal trends in flight delays.
- Compare delay performance across airlines.
- Identify airports with high total arrival delays.
- Analyze the major causes of flight delays.
- Engineer meaningful features from operational flight data.
- Handle class imbalance using **SMOTE**.
- Build a **Random Forest Classifier** for delay prediction.
- Evaluate the model using precision, recall, F1-score, confusion matrix, and ROC-AUC.
- Develop a Random Forest regression workflow for predicting arrival delay.

---

## 📊 Dataset

The project uses an `Airline_Delay_Cause.csv` dataset.

### Dataset Size

| Property | Value |
|---|---:|
| Records | 179,338 |
| Original Features | 21 |
| Time Period | 2015–2023 |
| Airport Codes | 396 |
| Main Target for Classification | `is_delayed` |
| Regression Target | `arr_delay` |

### Dataset Features

| Feature | Description |
|---|---|
| `year` | Year of the flight record |
| `month` | Month of the flight record |
| `carrier` | Airline/carrier code |
| `carrier_name` | Full airline name |
| `airport` | Airport code |
| `airport_name` | Full airport name |
| `arr_flights` | Number of arrival flights |
| `arr_del15` | Number of flights arriving 15+ minutes late |
| `carrier_ct` | Number of delays attributed to carrier issues |
| `weather_ct` | Number of delays attributed to weather |
| `nas_ct` | Number of delays attributed to NAS |
| `security_ct` | Number of security-related delays |
| `late_aircraft_ct` | Number of delays caused by late aircraft |
| `arr_cancelled` | Number of cancelled flights |
| `arr_diverted` | Number of diverted flights |
| `arr_delay` | Total arrival delay in minutes |
| `carrier_delay` | Total carrier-related delay |
| `weather_delay` | Total weather-related delay |
| `nas_delay` | Total NAS-related delay |
| `security_delay` | Total security-related delay |
| `late_aircraft_delay` | Total delay caused by late aircraft |

---

# 🔍 Exploratory Data Analysis

The notebook performs extensive exploratory analysis to understand the structure and behavior of flight delays.

### 1. Data Quality Analysis

The project checks:

- Dataset dimensions
- Data types
- Missing values
- Duplicate records
- Descriptive statistics
- Feature distributions

The dataset contains **179,338 records**, with approximately **0.19% missing values in most numerical operational fields** and approximately **0.33% missing values in `arr_del15`**.

No duplicate rows were detected.

---

## 📈 2. Yearly Delay Analysis

The project analyzes flight activity and delay behavior across years.

The dataset spans **2015 to 2023**.

The analysis examines:

- Total number of arrival flights by year
- Average arrival delay by year
- Total delay minutes by year
- Year-over-year delay trends

For example, the yearly arrival-flight volume and average delay are calculated to identify long-term changes in operational performance.

---

## 📅 3. Monthly Analysis

Monthly patterns are analyzed to identify seasonal effects.

The notebook evaluates:

- Total arrival delay by month
- Number of flights delayed by 15+ minutes
- Average delay by month
- Total flight volume by month

This helps reveal seasonal periods where delays become more frequent or severe.

---

## ✈️ 4. Carrier Analysis

Airlines are compared using:

- Flight frequency
- Total arrival delay
- Number of delayed flights
- Weather-related delays
- Carrier-specific operational performance

This allows comparison of delay behavior across different carriers.

---

## 🛫 5. Airport Analysis

Airport-level analysis identifies locations with high levels of delay.

The project calculates:

- Average arrival delay by airport
- Total arrival delay by airport
- Cancellation rate
- Diversion rate
- Delay causes by airport

The notebook identifies major airports such as **ORD, DFW, ATL, DEN, EWR, LAX, SFO, LGA, CLT, and MCO** among those with the highest total arrival delay.

---

# 🧩 Feature Engineering

Feature engineering is performed before machine learning to convert raw operational data into more useful predictive features.

### Cyclical Month Features

Because month is a cyclical variable, the project transforms it using sine and cosine encoding:

```python
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

This allows the model to understand that December and January are close to each other in the yearly cycle.

### Delay Rate

A delay-rate feature is created as:

```text
arr_del15_rate = arr_del15 / arr_flights
```

This represents the proportion of arrival flights delayed by at least 15 minutes.

### Binary Classification Target

The main classification target is:

```text
is_delayed = 1  if arr_del15_rate > 5%
is_delayed = 0  otherwise
```

Therefore, the classification problem asks:

> **Does this airport/carrier/month group have more than 5% of its arrival flights delayed by 15+ minutes?**

---

# 🤖 Machine Learning - Classification

## Random Forest Classifier

The classification model uses a **Random Forest Classifier**.

### Input Features

The classification model uses:

- `arr_flights`
- `month_sin`
- `month_cos`
- `carrier_ct`
- `weather_ct`
- `nas_ct`
- `security_ct`
- `late_aircraft_ct`
- `carrier`
- `airport`

Categorical variables are converted using **One-Hot Encoding**.

Numerical missing values are handled using **median imputation**.

---

## ⚖️ Handling Class Imbalance

The classification target is imbalanced, with substantially more observations belonging to the delayed class.

The project addresses this using **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data.

The Random Forest model also uses:

```text
class_weight = "balanced"
```

This helps improve recognition of the minority class rather than optimizing only for overall accuracy.

---

## 🌲 Model Configuration

The classifier uses:

```text
RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

---

# 📊 Classification Results

The stored notebook results show the following performance on the test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 92% |
| Class 0 Precision | 53% |
| Class 0 Recall | 86% |
| Class 0 F1-score | 66% |
| Class 1 Precision | 99% |
| Class 1 Recall | 92% |
| Class 1 F1-score | 95% |
| Weighted F1-score | 93% |
| ROC-AUC | **0.964** |

The ROC-AUC of approximately **0.964** indicates strong separation between the two delay-status classes on the held-out test data.

The notebook also generates:

- Confusion matrix
- ROC curve
- Classification report

---

# 📉 Regression Analysis

The notebook also contains a **Random Forest Regression** section intended to predict:

```text
arr_delay
```

where `arr_delay` represents total arrival delay in minutes for a flight-group observation.

Additional engineered features include:

- Delay rates
- Cause-specific delay rates
- Average delay per flight
- Cyclical month features
- Flight volume
- Carrier
- Airport

The regression model uses:

```text
RandomForestRegressor(
    n_estimators=600,
    max_depth=25,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

The notebook also evaluates a second Random Forest configuration using **200 trees**.

### Evaluation Metrics

The regression section calculates:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

### ⚠️ Implementation Note

The current notebook contains a target-assignment issue in the regression train/test split:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

Here, `y` is the **classification target (`is_delayed`)**, while the intended regression target is `Y = df["arr_delay"]`.

Therefore, the stored regression metrics in the notebook should **not be interpreted as actual `arr_delay` prediction performance**.

To correctly evaluate regression on arrival-delay minutes, the split should use:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)
```

This correction should be made before using the regression results for final reporting.

---

# 🛠️ Technologies Used

### Programming & Analysis

- Python
- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- Random Forest
- SMOTE
- One-Hot Encoding
- Median Imputation

### Development Environment

- Jupyter Notebook

---

# 📂 Project Structure

```text
Flight-Delay-Analytics-Prediction/
│
├── final_notebook.ipynb
├── Airline_Delay_Cause.csv
├── 22112114_socbiz_report.pdf
└── README.md
```

> Note: `Airline_Delay_Cause.csv` is required to execute the notebook but is not included in the provided ZIP repository.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Flight-Delay-Analytics-Prediction.git
cd Flight-Delay-Analytics-Prediction
```

## 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

Or:

```bash
pip install -r requirements.txt
```

## 3. Add the Dataset

Place the dataset in the project root:

```text
Flight-Delay-Analytics-Prediction/
└── Airline_Delay_Cause.csv
```

The notebook expects this exact filename:

```python
pd.read_csv("Airline_Delay_Cause.csv")
```

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
final_notebook.ipynb
```

and execute the cells sequentially.

---

# 🔄 End-to-End Workflow

```text
Raw Flight Data
       │
       ▼
Data Loading
       │
       ▼
Data Quality Checks
       │
       ├── Missing Values
       ├── Duplicate Detection
       └── Descriptive Statistics
       │
       ▼
Exploratory Data Analysis
       │
       ├── Year Analysis
       ├── Month Analysis
       ├── Carrier Analysis
       └── Airport Analysis
       │
       ▼
Feature Engineering
       │
       ├── Month Sine/Cosine
       ├── Delay Rate
       ├── Cause-Specific Rates
       └── Average Delay per Flight
       │
       ├───────────────────────┐
       ▼                       ▼
Classification             Regression
       │                       │
       ▼                       ▼
Train/Test Split           Train/Test Split
       │                       │
       ▼                       ▼
Imputation + Encoding     Imputation + Encoding
       │                       │
       ▼                       ▼
SMOTE                     Random Forest
       │                       │
       ▼                       ▼
Random Forest             Performance Metrics
       │
       ▼
Predictions
       │
       ▼
Confusion Matrix + ROC-AUC
```

---

# 💡 Key Business Insights

The analysis provides several useful operational insights:

### Airline Performance

Different carriers show substantially different delay patterns, making airline-level benchmarking possible.

### Airport Bottlenecks

Major airports account for a large share of total delay minutes, highlighting potential operational bottlenecks at high-volume hubs.

### Seasonal Effects

Flight delays vary across months, indicating that seasonal conditions can play an important role in operational performance.

### Delay Root Causes

The dataset separates delays into:

- Carrier
- Weather
- NAS
- Security
- Late aircraft

This enables organizations to move beyond simply measuring delay volume and investigate **why delays occur**.

### Late Aircraft Effect

Late arrival of a previous aircraft can propagate through subsequent operations, making it an important contributor to overall delay.

---

# 📌 Key Takeaways

- **179K+ flight records** were analyzed across 2015–2023.
- Multiple airlines and hundreds of airports were evaluated.
- Extensive EDA was performed across time, carriers, and airports.
- Cyclical feature engineering was applied to monthly seasonality.
- Delay-rate features were created for machine learning.
- **SMOTE + class-weighted Random Forest** was used to handle class imbalance.
- The classification model achieved approximately **92% test accuracy**.
- The classification model achieved approximately **0.964 ROC-AUC**.
- The project provides both operational analysis and predictive modeling.
- The regression implementation should be corrected before its stored metrics are interpreted as `arr_delay` prediction results.

---

# 📜 Project Report

A detailed project report is included in the repository:

```text
22112114_socbiz_report.pdf
```

It provides additional visual analysis, trends, model discussion, and project documentation.

---

# 🔮 Future Improvements

Potential improvements to the project include:

- Correcting and fully validating the regression pipeline.
- Using a proper time-based train/test split to better simulate future prediction.
- Comparing Random Forest with XGBoost, LightGBM, or Gradient Boosting.
- Performing hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
- Adding feature-importance and SHAP-based model interpretability.
- Building an interactive dashboard using Power BI or Streamlit.
- Adding airport-level and carrier-level prediction interfaces.
- Deploying the trained classification model as an API.
- Creating automated model monitoring and retraining pipelines.

---

# 👨‍💻 Author

**Aryan Kumar**

B.Tech - Mechanical Engineering  
National Institute of Technology, Agartala

---

## ⭐ Project Focus

**Data Analytics | Exploratory Data Analysis | Machine Learning | Classification | Regression | Feature Engineering | Aviation Analytics**

> This project demonstrates how historical aviation data can be transformed into actionable insights and machine-learning-based delay prediction.
