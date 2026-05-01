import streamlit as st
import pandas as pd
import plotly.express as px

# 1. إعدادات الصفحة والعنوان
st.set_page_config(page_title="Loan Strategy Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("🏦 AI-Driven Loan Strategic Dashboard")
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.header("Filter Options")
st.sidebar.info("هذا النظام يقوم بتحويل توقعات الموديل إلى قرارات استراتيجية بناءً على المخاطر.")

# 2. تحميل البيانات
# تأكد أن اسم الملف مطابق للملف الموجود عندك في الفولدر
FILE_NAME = 'final_loan_decisions_with_strategy.csv'

try:
    df = pd.read_csv(FILE_NAME)
    
    # --- صف الإحصائيات العلوية (KPIs) ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_apps = len(df)
    approved = len(df[df['Decision'] != 'Reject (Risk > Profit)'])
    vip_count = len(df[df['Decision'].str.contains('VIP', na=False)])
    avg_risk = df['risk_score'].mean()

    col1.metric("Total Applications", f"{total_apps:,}")
    col2.metric("Approved Loans", f"{approved:,}", f"{(approved/total_apps)*100:.1f}%")
    col3.metric("VIP Customers ⭐", f"{vip_count:,}")
    col4.metric("Avg Risk Score", f"{avg_risk:.2f}")

    st.divider()

    # --- الرسومات البيانية ---
    row1_col1, row1_col2 = st.columns([1, 1])

    with row1_col1:
        st.subheader("📊 Distribution of Decisions")
        # رسمة الدائرة لنسب القرارات
        fig_pie = px.pie(df, names='Decision', hole=0.5, 
                         color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1_col2:
        st.subheader("💰 Interest Rates by Property Area")
        # رسمة الأعمدة لتوزيع أسعار الفائدة حسب المناطق
        fig_bar = px.histogram(df, x="Property_Area", color="Interest_Rate", 
                               barmode="group", text_auto=True,
                               color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- جدول البيانات التفاعلي ---
    st.divider()
    st.subheader("🔍 Search & Explore Customer Data")
    
    # محرك بحث بسيط
    search_query = st.text_input("بحث بالتعليم أو منطقة العقار (e.g. Graduate, Urban)...")
    
    display_df = df[['Education', 'Property_Area', 'risk_score', 'Decision', 'Interest_Rate']]
    
    if search_query:
        display_df = display_df[
            display_df['Education'].str.contains(search_query, case=False, na=False) | 
            display_df['Property_Area'].str.contains(search_query, case=False, na=False)
        ]

    # عرض الجدول مع تلوين درجات المخاطرة
    st.dataframe(
        display_df.style.background_gradient(subset=['risk_score'], cmap='RdYlGn_r'),
        use_container_width=True
    )

    st.success(f"✅ تم تحميل {len(df)} سجل بنجاح من ملف {FILE_NAME}")

except FileNotFoundError:
    st.error(f"❌ لم يتم العثور على ملف '{FILE_NAME}'. تأكد من تشغيل ملف 'buisness.py' أولاً.")
except Exception as e:
    st.error(f"⚠️ حدث خطأ غير متوقع: {e}")