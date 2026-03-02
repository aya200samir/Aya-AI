import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import re
import tempfile
import os
import joblib
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go
import shap
from sklearn.model_selection import train_test_split

# ------------------------------
# إعداد الصفحة
# ------------------------------
st.set_page_config(page_title="GDPR Compliance AI", layout="wide")
st.title("📋 نظام تقييم الامتثال للائحة العامة لحماية البيانات (GDPR)")

# ------------------------------
# دوال مساعدة
# ------------------------------
def clean_text(text):
    """إزالة المسافات الزائدة"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_retention_period(text):
    """استخراج مدة الاحتفاظ من النص"""
    text = text.lower()
    years = re.findall(r'(\d+)\s*year', text)
    if years:
        return int(years[0])
    months = re.findall(r'(\d+)\s*month', text)
    if months:
        return int(months[0]) / 12.0
    return 1.0  # افتراضي

# ------------------------------
# تحميل المواد القانونية (GDPR Articles)
# ------------------------------
GDPR_ARTICLES = [
    {
        "id": "Art. 5",
        "title": "Principles relating to processing of personal data",
        "text": "Personal data shall be processed lawfully, fairly and in a transparent manner. Collected for specified, explicit and legitimate purposes. Adequate, relevant and limited to what is necessary."
    },
    {
        "id": "Art. 32",
        "title": "Security of processing",
        "text": "Implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk."
    },
    {
        "id": "Art. 17",
        "title": "Right to erasure ('right to be forgotten')",
        "text": "The data subject shall have the right to obtain erasure of personal data without undue delay."
    },
    {
        "id": "Art. 37",
        "title": "Designation of the data protection officer",
        "text": "Designate a data protection officer if processing is carried out by a public authority or requires regular and systematic monitoring of data subjects on a large scale."
    }
]

# ------------------------------
# دوال استخراج الكيانات من النص (باستخدام regex)
# ------------------------------
def extract_entities(text):
    """استخراج الكيانات الأساسية من النص باستخدام تعبيرات منتظمة (بدون نماذج ثقيلة)"""
    entities = {}
    
    # اسم الشركة (أول سطر يحتوي على Ltd, Limited, etc)
    company_match = re.search(r'([A-Z][A-Za-z\s]+(Ltd|Limited|LLC|Inc))', text)
    entities['company_name'] = company_match.group(1) if company_match else "Unknown"
    
    # مسؤول حماية البيانات DPO
    dpo_match = re.search(r'Data Protection Officer[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', text, re.IGNORECASE)
    entities['dpo'] = dpo_match.group(1) if dpo_match else None
    
    # أنواع البيانات (كلمات مفتاحية)
    data_keywords = ['name', 'address', 'email', 'phone', 'payment', 'credit card', 'bank', 'ip address', 'location']
    found_data = []
    for kw in data_keywords:
        if re.search(kw, text, re.IGNORECASE):
            found_data.append(kw)
    entities['data_types'] = found_data
    
    # مدة الاحتفاظ
    retention_match = re.search(r'retain.*?for (\d+)\s*(year|month|day)', text, re.IGNORECASE)
    if retention_match:
        num = retention_match.group(1)
        unit = retention_match.group(2)
        entities['retention_period'] = f"{num} {unit}"
    else:
        entities['retention_period'] = "Not specified"
    
    # نقل البيانات خارج الاتحاد الأوروبي
    entities['data_transfer_outside_eea'] = bool(re.search(r'transfer.*(outside|third country|international)', text, re.IGNORECASE))
    
    # الحق في النسيان
    entities['right_to_be_forgotten'] = bool(re.search(r'right to (be forgotten|erasure)', text, re.IGNORECASE))
    
    return entities

# ------------------------------
# دوال معالجة PDF
# ------------------------------
def extract_text_from_pdf(pdf_path):
    """استخراج النص من ملف PDF"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return clean_text(text)

# ------------------------------
# طبقة التشابه الدلالي (Semantic Matcher)
# ------------------------------
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(company_text, model, articles=GDPR_ARTICLES):
    """حساب التشابه بين نص الشركة ومواد القانون"""
    article_texts = [art['text'] for art in articles]
    article_embeddings = model.encode(article_texts)
    company_emb = model.encode([company_text])
    similarities = np.dot(article_embeddings, company_emb.T).flatten()
    top_indices = np.argsort(similarities)[-3:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'article': articles[idx]['id'],
            'title': articles[idx]['title'],
            'similarity': float(similarities[idx])
        })
    return results

# ------------------------------
# محرك القواعد (Rule Engine)
# ------------------------------
def check_violations(entities):
    """تطبيق القواعد الصريحة"""
    violations = []
    
    # القاعدة 1: وجود DPO
    if entities.get('dpo') is None:
        violations.append({
            'rule_id': 'R001',
            'violation': 'No Data Protection Officer designated',
            'severity': 3,
            'article': 'Art. 37'
        })
    
    # القاعدة 2: كثرة أنواع البيانات
    if len(entities.get('data_types', [])) > 10:
        violations.append({
            'rule_id': 'R002',
            'violation': 'Excessive data collection',
            'severity': 2,
            'article': 'Art. 5'
        })
    
    # القاعدة 3: نقل البيانات خارج المنطقة الاقتصادية الأوروبية
    if entities.get('data_transfer_outside_eea'):
        violations.append({
            'rule_id': 'R003',
            'violation': 'Data transfer outside EEA without adequacy decision',
            'severity': 3,
            'article': 'Art. 44-49'
        })
    
    # القاعدة 4: الحق في النسيان
    if not entities.get('right_to_be_forgotten'):
        violations.append({
            'rule_id': 'R004',
            'violation': 'Right to be forgotten not mentioned',
            'severity': 2,
            'article': 'Art. 17'
        })
    
    # القاعدة 5: مدة الاحتفاظ غير محددة
    if entities.get('retention_period') == 'Not specified':
        violations.append({
            'rule_id': 'R005',
            'violation': 'Data retention period not specified',
            'severity': 2,
            'article': 'Art. 5(1)(e)'
        })
    
    return violations

# ------------------------------
# نموذج تقييم المخاطر (Risk Scorer) باستخدام XGBoost
# ------------------------------
@st.cache_resource
def load_risk_model():
    """توليد بيانات تدريب افتراضية وتدريب نموذج XGBoost"""
    np.random.seed(42)
    n_samples = 1000
    data = []
    for _ in range(n_samples):
        has_dpo = np.random.choice([0, 1], p=[0.3, 0.7])
        num_data_types = np.random.randint(1, 20)
        retention_years = np.random.choice([1, 3, 5, 10, 0.5], p=[0.2,0.3,0.3,0.1,0.1])
        has_transfer = np.random.choice([0, 1], p=[0.6, 0.4])
        has_forgotten = np.random.choice([0, 1], p=[0.1, 0.9])
        similarity_art5 = np.random.uniform(0.3, 1.0)
        num_violations = np.random.poisson(1.5)
        
        # درجة المخاطرة المحاكاة
        risk = ( (1-has_dpo)*30 + num_data_types*2 + (retention_years>5)*10 + 
                has_transfer*15 + (1-has_forgotten)*20 + (1-similarity_art5)*30 + num_violations*5 )
        risk = min(risk, 100)
        data.append([has_dpo, num_data_types, retention_years, has_transfer, 
                     has_forgotten, similarity_art5, num_violations, risk])
    
    columns = ['has_dpo', 'num_data_types', 'retention_years', 'has_transfer',
               'has_forgotten', 'similarity_art5', 'num_violations', 'risk_score']
    df = pd.DataFrame(data, columns=columns)
    
    X = df.drop('risk_score', axis=1)
    y = df['risk_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    return model

def prepare_features(entities, semantic_results, violations):
    """تحويل الكيانات والنتائج إلى ميزات رقمية"""
    features = {}
    features['has_dpo'] = 1 if entities.get('dpo') else 0
    features['num_data_types'] = len(entities.get('data_types', []))
    
    retention = entities.get('retention_period', 'Not specified')
    if retention != 'Not specified':
        num = re.findall(r'\d+', retention)
        if num:
            features['retention_years'] = float(num[0])
        else:
            features['retention_years'] = 1.0
    else:
        features['retention_years'] = 5.0
    
    features['has_transfer'] = 1 if entities.get('data_transfer_outside_eea') else 0
    features['has_forgotten'] = 1 if entities.get('right_to_be_forgotten') else 0
    
    if semantic_results:
        features['similarity_art5'] = semantic_results[0]['similarity']
    else:
        features['similarity_art5'] = 0.5
    
    features['num_violations'] = len(violations)
    
    return pd.DataFrame([features])

def predict_risk(model, features_df):
    risk = model.predict(features_df)[0]
    return max(0, min(100, risk))

# ------------------------------
# كاشف الشذوذ (Anomaly Detector) باستخدام Isolation Forest
# ------------------------------
@st.cache_resource
def load_anomaly_model():
    """تدريب نموذج Isolation Forest على بيانات افتراضية"""
    np.random.seed(42)
    n_samples = 500
    X_train = np.random.randn(n_samples, 7)  # 7 ميزات
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    return model

# ------------------------------
# البحث عن سوابق قضائية باستخدام FAISS (بيانات افتراضية)
# ------------------------------
@st.cache_resource
def load_case_matcher():
    """إنشاء قاعدة بيانات سوابق افتراضية"""
    cases = [
        {'company': 'British Airways', 'fine': 20400000, 'reason': 'Data breach affecting 500k customers', 'violations': ['Art. 32']},
        {'company': 'Marriott', 'fine': 18400000, 'reason': 'Data breach affecting 339 million guests', 'violations': ['Art. 5', 'Art. 32']},
        {'company': 'Google', 'fine': 50000000, 'reason': 'Lack of transparency, insufficient consent', 'violations': ['Art. 5', 'Art. 6', 'Art. 7']},
        {'company': 'H&M', 'fine': 35200000, 'reason': 'Illegal monitoring of employees', 'violations': ['Art. 5', 'Art. 6']},
        {'company': 'TIM', 'fine': 27800000, 'reason': 'Aggressive marketing calls without consent', 'violations': ['Art. 6', 'Art. 21']}
    ]
    for case in cases:
        case['text'] = f"{case['company']} {case['reason']} Violations: {', '.join(case['violations'])}"
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([case['text'] for case in cases])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    return {'cases': cases, 'index': index, 'model': model}

def find_similar_cases(matcher, description, top_k=3):
    """البحث عن أقرب القضايا"""
    query_emb = matcher['model'].encode([description])
    distances, indices = matcher['index'].search(query_emb.astype(np.float32), top_k)
    similar = []
    for i, idx in enumerate(indices[0]):
        if idx < len(matcher['cases']):
            case = matcher['cases'][idx]
            similar.append({
                'company': case['company'],
                'fine': case['fine'],
                'reason': case['reason'],
                'distance': float(distances[0][i])
            })
    return similar

# ------------------------------
# تفسير SHAP
# ------------------------------
def explain_risk(model, features_df, feature_names):
    """تفسير النتيجة باستخدام SHAP"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
    
    impacts = list(zip(feature_names, shap_vals[0]))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    explanation = []
    for name, impact in impacts[:3]:
        explanation.append({
            'feature': name,
            'impact': impact,
            'value': features_df.iloc[0][name]
        })
    return explanation

# ------------------------------
# التطبيق الرئيسي (Streamlit)
# ------------------------------
def main():
    # تحميل النماذج (مرة واحدة)
    with st.spinner("جاري تحميل النماذج..."):
        semantic_model = load_semantic_model()
        risk_model = load_risk_model()
        anomaly_model = load_anomaly_model()
        case_matcher = load_case_matcher()
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("🔧 الإعدادات")
        uploaded_file = st.file_uploader("ارفع ملف PDF لسياسة الخصوصية", type=['pdf'])
        st.markdown("---")
        st.markdown("**معلومات إضافية**")
        company_name = st.text_input("اسم الشركة (اختياري)", "")
        company_size = st.selectbox("حجم الشركة", ["صغيرة (<50)", "متوسطة (50-250)", "كبيرة (>250)"])
    
    # معالجة الملف إذا تم رفعه
    if uploaded_file is not None:
        with st.spinner("جارٍ تحليل المستند..."):
            # حفظ الملف المؤقت
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            # استخراج النص
            text = extract_text_from_pdf(tmp_path)
            
            # استخراج الكيانات
            entities = extract_entities(text)
            if company_name:
                entities['company_name'] = company_name
            
            # التشابه الدلالي
            semantic_results = compute_similarity(text, semantic_model)
            
            # تطبيق القواعد
            violations = check_violations(entities)
            
            # تجهيز الميزات
            features_df = prepare_features(entities, semantic_results, violations)
            
            # التنبؤ بالمخاطر
            risk_score = predict_risk(risk_model, features_df)
            
            # كشف الشذوذ
            # نحتاج لمصفوفة ميزات بنفس أبعاد التدريب (7 ميزات)
            anomaly_features = features_df.values.reshape(1, -1)
            anomaly_pred = anomaly_model.predict(anomaly_features)[0]  # -1 شاذ
            
            # البحث عن سوابق
            case_description = f"Violations: {[v['violation'] for v in violations]}. Similar articles: {semantic_results}"
            similar_cases = find_similar_cases(case_matcher, str(case_description))
            
            # تفسير SHAP
            feature_names = features_df.columns.tolist()
            explanation = explain_risk(risk_model, features_df, feature_names)
            
            # حذف الملف المؤقت
            os.unlink(tmp_path)
        
        # عرض النتائج
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("📊 درجة المخاطرة")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "مؤشر المخاطر"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if risk_score > 70 else "orange" if risk_score > 40 else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            if anomaly_pred == -1:
                st.warning("⚠️ هذا النمط غير معتاد ويشبه شركات تم تغريمها سابقاً!")
            
            st.subheader("📌 المعلومات المستخلصة")
            st.json(entities)
        
        with col2:
            st.subheader("⚖️ المخالفات المكتشفة")
            if violations:
                df_viol = pd.DataFrame(violations)
                st.dataframe(df_viol, use_container_width=True)
            else:
                st.success("لم يتم العثور على مخالفات صريحة.")
            
            st.subheader("📜 التشابه مع مواد القانون")
            for res in semantic_results:
                st.markdown(f"**{res['article']}**: {res['title']} (تشابه: {res['similarity']:.2f})")
            
            st.subheader("🔍 تفسير النتيجة")
            for exp in explanation:
                st.write(f"- **{exp['feature']}**: تأثير {exp['impact']:.2f} (القيمة الحالية: {exp['value']})")
            
            st.subheader("⚡ سوابق قضائية مشابهة")
            for case in similar_cases:
                st.markdown(f"**{case['company']}**: غرامة €{case['fine']:,} - {case['reason']} (المسافة: {case['distance']:.2f})")
        
        # توصيات
        st.markdown("---")
        st.subheader("📝 توصيات لتحسين الامتثال")
        
        recommendations = []
        for v in violations:
            recommendations.append(f"- **{v['violation']}** (المادة {v['article']}): يجب معالجتها.")
        
        if risk_score > 70:
            recommendations.append("- **خطر مرتفع جداً**: يوصى بمراجعة قانونية شاملة فوراً.")
        elif risk_score > 40:
            recommendations.append("- **خطر متوسط**: ينصح باتخاذ إجراءات تصحيحية خلال 3 أشهر.")
        else:
            recommendations.append("- **مستوى مخاطر منخفض**: حافظ على الامتثال وراجع التحديثات الدورية.")
        
        for rec in recommendations:
            st.markdown(rec)
    
    else:
        st.info("👈 يرجى رفع ملف PDF لبدء التحليل.")

if __name__ == "__main__":
    main()
