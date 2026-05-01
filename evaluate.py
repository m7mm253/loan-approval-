# =============================
# 1. IMPORT LIBRARIES
# =============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

# =============================
# 2. CONFIGURATION & LOADING
# =============================
DATA_FILE = 'loan_data_cleaned_final.csv'
MODEL_FILE = 'loan_model.pkl'

print("🚀 Starting Evaluation Process...")

try:
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Missing {DATA_FILE} or {MODEL_FILE}")

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    target_col = 'Loan_Status' 
    
    # تنظيف البيانات من الأعمدة غير الضرورية
    cols_to_drop = [target_col, 'Loan_ID']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # الاحتفاظ بالقيم الحقيقية
    y = df[target_col]

    # معالجة الـ Features (Dummies)
    X = pd.get_dummies(X)

    # مطابقة الأعمدة مع ما رآه الموديل أثناء التدريب
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# 3. FIXING LABEL MISMATCH (The Error Fix)
# =============================
    y_pred_raw = model.predict(X_test)
    
    # توحيد التنسيق: تحويل y_test من (Y, N) إلى (1, 0)
    y_test_numeric = y_test.map({'Y': 1, 'N': 0}).astype(int)
    
    # توحيد y_pred: التأكد إنها أرقام (لو كانت طالعة Y/N نحولها لـ 1/0)
    if hasattr(y_pred_raw[0], 'lower'): # لو كانت نصوص
        y_pred_numeric = pd.Series(y_pred_raw).map({'Y': 1, 'N': 0}).astype(int)
    else:
        y_pred_numeric = y_pred_raw.astype(int)

    print("✅ Model and Data alignment complete!")

# =============================
# 4. PRINT RESULTS
# =============================
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    
    print("\n" + "="*50)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy  : {accuracy:.2%}")
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"ROC-AUC   : {roc_auc_score(y_test_numeric, y_prob):.2%}")
    
    print("="*50)

    print("\n📄 Classification Report:")
    # بنستخدم أرقام في التقرير عشان نضمن عدم حدوث خطأ التوع
    print(classification_report(y_test_numeric, y_pred_numeric, target_names=['Denied (N)', 'Approved (Y)']))

    # Confusion Matrix Visual
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test_numeric, y_pred_numeric)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['N', 'Y'], yticklabels=['N', 'Y'])
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

except Exception as e:
    print(f"❌ An error occurred: {e}")