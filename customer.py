
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False

# =========================================================
# 1. Load Dataset
# =========================================================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully")
print(df.head())

# =========================================================
# 2. Exploratory Data Analysis (EDA)
# =========================================================
print("\nDataset Info:")
print(df.info())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Churn distribution
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Churn vs Contract
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn vs Contract Type")
plt.xticks(rotation=15)
plt.show()

# Churn vs Monthly Charges
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Churn vs Tenure
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True)
plt.title("Tenure Distribution by Churn")
plt.show()

# Payment Method vs Churn
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.title("Payment Method vs Churn")
plt.xticks(rotation=45)
plt.show()

# =========================================================
# 3. Data Cleaning
# =========================================================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df.drop("customerID", axis=1, inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = label_encoder.fit_transform(df[col])

# =========================================================
# 4. Correlation Heatmap
# =========================================================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# =========================================================
# 5. Feature Scaling
# =========================================================
X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# 6. Train-Test Split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================================================
# 7. Logistic Regression
# =========================================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print("Recall:", recall_score(y_test, lr_preds))
print("ROC-AUC:", roc_auc_score(y_test, lr_probs))

# =========================================================
# 8. Random Forest
# =========================================================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Recall:", recall_score(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_probs))

# =========================================================
# 9. XGBoost (Optional)
# =========================================================
if xgb_available:
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train, y_train)

    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    print("\nXGBoost Results")
    print("Accuracy:", accuracy_score(y_test, xgb_preds))
    print("Recall:", recall_score(y_test, xgb_preds))
    print("ROC-AUC:", roc_auc_score(y_test, xgb_probs))
else:
    print("\nXGBoost not installed. Skipping XGBoost model.")

# =========================================================
# 10. Confusion Matrix (Random Forest)
# =========================================================
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================================
# 11. Business Insights
# =========================================================
print("\nBusiness Insights:")
print("1. Month-to-month contract customers show higher churn.")
print("2. Higher monthly charges increase churn probability.")
print("3. Customers with longer tenure are more loyal.")
print("4. Electronic check users tend to churn more.")
print("5. Churn prediction enables proactive retention strategies.")

print("\nCustomer Churn Prediction Project Completed Successfully ✅")
