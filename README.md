# 📊 **Property Assessment Challenge for Cook County, Illinois**

## 🏆 **Project Overview**
This project, completed as part of the **FIN 550 - Big Data Analytics** course, involved predicting residential property values in Cook County, Illinois. Out of approximately 100 students in the class, this project achieved **1st Place** by minimizing prediction error (Mean Squared Error, MSE) to 15.09 billions.

---

## 📝 **Problem Statement**
The objective of this project was to build a predictive model to assess property values based on features provided in historical property data. The predictions were compared to actual property values to evaluate model accuracy.

### **Dataset Overview**
1. `predict_property_data.csv` – 10,000 properties for assessment.
2. `historic_property_data.csv` – 50,000 properties with actual sales prices.
3. `codebook.csv` – Descriptions of all variables.

The goal was to minimize the **Mean Squared Error (MSE)** between the predicted and actual property values.

---

## ⚙️ **Tools and Methodology**

### **Tools**
- **Programming Language:** R
- **Libraries Used:** 
  - `xgboost` – Extreme Gradient Boosting
  - `randomForest` – Ensemble learning for regression
  - `caret` – Preprocessing and model evaluation
  - `tidyverse` – Data wrangling and visualization

---

### **Methodology**

#### **Preprocessing**
1. **Handling Missing Values**  
   - Imputed missing values using median for numeric variables and mode for categorical variables.  
2. **Scaling**  
   - Standardized numerical features to have a mean of 0 and a standard deviation of 1.  
3. **One-Hot Encoding**  
   - Converted categorical variables into dummy variables to make them suitable for modeling.

#### **Modeling Approaches**
1. **Random Forest**
   - Used for its robustness and ability to handle high-dimensional structured data.
   - Performed hyperparameter tuning to optimize the number of trees and maximum depth.

2. **XGBoost**
   - Selected for its ability to capture complex relationships and boost model performance.
   - Performed grid search to fine-tune learning rate, max depth, and number of estimators.

#### **Model Evaluation**
- Evaluated both models using **Mean Squared Error (MSE)** on a validation set.
- Selected the model with the lowest MSE for final predictions.

---

## 🏅 **Results**
The final model achieved the **lowest Mean Squared Error (MSE)** among all teams, earning 1st Place in a class of ~100 students.

- **Final Model:** XGBoost (after hyperparameter tuning)
