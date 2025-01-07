# Car Price Prediction

![Project Banner](https://github.com/Parmod2310/PRJ-CAR-PRICE-PREDICTION/blob/24c295716ec1a969cc1b260cfe8c565f3c38c1e7/image/pexels-heru-vision-289677440-13387407.jpg)

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Project Steps](#project-steps)
   1. [Import Libraries](#1️⃣-import-libraries)
   2. [Data Loading & EDA](#2️⃣-data-loading--eda)
   3. [Data Preprocessing](#3️⃣-data-preprocessing)
   4. [Train-Test Split](#4️⃣-train-test-split)
   5. [Model Implementation](#5️⃣-model-implementation)
   6. [Hyperparameter Tuning](#6️⃣-hyperparameter-tuning)
   7. [Model Persistence](#7️⃣-model-persistence)
5. [Results](#results)
6. [Directory Structure](#directory-structure)
7. [How to Run](#how-to-run)
8. [Key Learnings](#key-learnings)
9. [Future Enhancements](#future-enhancements)
10. [License](#license)
11. [Contact](#contact)


---
## 🚗 Project Overview
This project leverages machine learning to predict car prices based on various attributes. The workflow includes **automated/manual Exploratory Data Analysis (EDA)**, **data preprocessing**, **regression modeling**, **evaluation**, and **model persistence**. The dataset used is `audi.csv`, with `price` as the target variable.

---

## 📦 Installation
To set up the environment, use the following commands to install required libraries:

```bash
pip install ydata-profiling
pip install catboost
pip install -r requirements.txt
```

If profiling throws an error, install from source:
```bash
pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
```

---

## 📂 Dataset
### Dataset: `audi.csv`
- **Target Variable:** `price`
- Includes features such as `model`, `year`, `mileage`, `fuel type`, `transmission`, etc.

---

## 🔧 Project Steps

### 1️⃣ **Import Libraries**
Essential Python libraries used:
- `numpy`, `pandas`, `matplotlib`
- `sklearn`, `ydata_profiling`, `pickle`

### 2️⃣ **Data Loading & EDA**
#### Automated EDA
Generate detailed profiling reports:
```python
from ydata_profiling import ProfileReport
display(ProfileReport(df))
```
#### Manual EDA
- Dataset summary using `.info()` and `.describe()`
- Null value analysis with `.isna().sum()`

### 3️⃣ **Data Preprocessing**
- **Encoding:**
  - `Label Encoding`: For `model`, `fuel type`
  - `One-Hot Encoding`: For `transmission`
- **Feature Scaling:** Normalize features using `StandardScaler`

### 4️⃣ **Train-Test Split**
Divide the data into training and testing subsets:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

### 5️⃣ **Model Implementation**
#### Models:
- **Random Forest Regressor**
- **Linear Regression**
- **Extra Trees Regressor**
- **CatBoost Regressor**

Each model is evaluated using:
- `R2 Score`
- `Mean Absolute Error`

### 6️⃣ **Hyperparameter Tuning**
Optimize models using `RandomizedSearchCV`:
```python
from sklearn.model_selection import RandomizedSearchCV
```

### 7️⃣ **Model Persistence**
Save models for future use:
```python
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
```
Load and use saved models:
```python
model = pickle.load(open('model.pkl', 'rb'))
```

---

## 📊 Results
**Evaluation Metrics:**
- **R2 Score**: Measure of variance explained by the model.
- **Mean Absolute Error**: Average difference between actual and predicted values.

### Predicted vs Actual Values:
![Prediction Graph](https://github.com/Parmod2310/PRJ-CAR-PRICE-PREDICTION/blob/24c295716ec1a969cc1b260cfe8c565f3c38c1e7/image/predicted_vs_actual_values.png)


---

## 🗂️ Directory Structure
```plaintext
📂 Car Price Prediction
├── audi.csv             # Dataset file
├── model.pkl            # Saved ML model
├── CAR PRICE PREDICTION Part 1.ipynb # Jupyter notebook for data preprocessing & exploration
├── CAR PRICE PREDICTION Part 2.ipynb # Jupyter notebook for model building & evaluation
├── README.md            # Project documentation
├── requirements.txt     # Dependencies
├── catboost_info/       # Folder containing CatBoost-specific info
│   ├── learn/           # Folder containing CatBoost learning logs
│   │   ├── events.out.tfevents  # Event file generated during CatBoost training
│   ├── catboost_training.json # CatBoost model training details in JSON format
│   ├── learn_error.tsv  # Error logs during training
│   └── time_left.tsv    # Time remaining during CatBoost training
└── image/               # Folder containing images (e.g., Project Banner, Prediction Graph)
    ├── pexels-heru-vision-289677440-13387407.jpg
    └── predicted_vs_actual_values.png

```

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone <repo-link>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

---

## 📈 Key Learnings
- Automated/manual EDA techniques.
- Encoding and feature scaling.
- Multiple regression model implementation.
- Model evaluation and hyperparameter tuning.
- Saving/loading models with `pickle`.

---

## 🌟 Future Enhancements
- Add visualizations for feature importance.
- Explore advanced models like Gradient Boosting.
- Deploy the model with **Flask** or **Streamlit**.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🧑‍💻 Contact
For queries or feedback, feel free to reach out:
- **Email:** p921035@gmail.com
- **GitHub:** [Parmod2310](https://github.com/Parmod2310)
- **LinkedIn:** [Parmod2310](https://www.linkedin.com/in/parmod2310/)

---

## 🙏 Thank You for Visiting!
We appreciate you taking the time to explore this project. Your support and interest mean a lot. If you have any questions, suggestions, or feedback, please feel free to reach out. Together, we can continue to innovate and make a positive impact.

Stay curious, keep learning, and create something amazing!
