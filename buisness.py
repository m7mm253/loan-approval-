import pandas as pd
import joblib

# 1. تحميل الملفات
model = joblib.load('loan_model.pkl')
df = pd.read_csv('loan_data_cleaned_final.csv')

# 2. تجهيز البيانات (حذف الـ ID والـ Target عشان نعرف نتوقع)
target_col = 'Loan_Status'
X = df.drop(columns=[target_col, 'Loan_ID'], errors='ignore')
X = pd.get_dummies(X)

# التأكد من مطابقة الأعمدة للموديل
if hasattr(model, 'feature_names_in_'):
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

# 3. حساب احتمالية التعثر (Risk Score)
# الموديل بيطلع احتمالية (من 0 لـ 1)
df['risk_score'] = model.predict_proba(X)[:, 1] 

# ==========================================
# 4. تطبيق الـ Actionable Insights
# ==========================================

def apply_strategy(row):
    score = row['risk_score']
    education = row['Education']
    area = row['Property_Area']
    
    # أ- تطبيق الـ Tiered Interest Rate
    if score < 0.2:
        rate = "9% (Prime)"
        action = "Strong Approve"
    elif score < 0.5:
        rate = "13% (Standard)"
        action = "Approve"
    elif score < 0.7:
        rate = "18% (High Risk)"
        action = "Approve with Collateral"
    else:
        rate = "N/A"
        action = "Reject (High Risk Cost)"

    # ب- تمييز الـ Strategic Growth Segment (Semi-urban graduates)
    if education == 'Graduate' and area == 'Semiurban' and score < 0.5:
        action = "⭐ VIP Approved (Growth Segment)"
        rate = "8.5% (Special Discount)"

    return pd.Series([action, rate])

# تطبيق الاستراتيجية على كل العملاء
df[['Decision', 'Interest_Rate']] = df.apply(apply_strategy, axis=1)

# ==========================================
# 5. عرض النتائج النهائية للبيزنس
# ==========================================
print("\n" + "!"*40)
print("🏦 STRATEGIC LOAN DECISIONS REPORT")
print("!"*40)

# عرض عينة من القرارات الجديدة
output_cols = ['Education', 'Property_Area', 'risk_score', 'Decision', 'Interest_Rate']
print(df[output_cols].head(15))

# حفظ النتائج في ملف اكسيل للمدير
df.to_csv('final_loan_decisions_with_strategy.csv', index=False)
print("\n✅ تم حفظ ملف القرارات النهائي باسم: final_loan_decisions_with_strategy.csv")