
---

## 📌 README – Customer Churn Prediction

```markdown
# 📉 Customer Churn Prediction

## 📌 Project Overview
Customer Churn Prediction aims to identify customers who are likely to stop using a service.  
This project uses machine learning classification models to predict churn based on customer behavior and service usage data.

## 🎯 Objectives
- Understand factors contributing to customer churn
- Perform exploratory data analysis (EDA)
- Build and evaluate churn prediction models
- Help businesses take proactive retention actions

## 📂 Dataset
- **Name:** Telco Customer Churn Dataset
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## 🛠️ Technologies Used
- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- (Optional) XGBoost

## ⚙️ Project Workflow
1. Load and inspect dataset  
2. Perform basic EDA to understand churn patterns  
3. Data cleaning and preprocessing  
4. Encode categorical variables  
5. Feature scaling  
6. Train-test split  
7. Train models:
   - Logistic Regression
   - Random Forest
   - XGBoost (optional)
8. Evaluate using Accuracy, Recall, ROC-AUC  
9. Generate business insights  

## 📊 Model Evaluation Metrics
- Accuracy
- Recall
- ROC-AUC Score
- Confusion Matrix

## 📈 Business Insights
- Month-to-month contract customers churn more
- Higher monthly charges increase churn risk
- Long-tenure customers are more loyal
- Early churn prediction helps improve customer retention

## 🚀 How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
python customer.py
