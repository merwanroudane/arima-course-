import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

# إعداد الصفحة
st.set_page_config(page_title="محاضرة السلاسل الزمنية", layout="wide", initial_sidebar_state="expanded")

# CSS مخصص
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(120deg, #2E86AB 0%, #A23B72 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .concept-box {
        background-color: #f0f8ff;
        border-right: 5px solid #2E86AB;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .definition-box {
        background-color: #fff9e6;
        border-right: 5px solid #FFB703;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .example-box {
        background-color: #f0fff4;
        border-right: 5px solid #06D6A0;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .warning-box {
        background-color: #ffe6e6;
        border-right: 5px solid #EF476F;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .formula-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        direction: ltr;
        text-align: center;
        margin: 10px 0;
    }
    h2 {
        color: #2E86AB;
        border-bottom: 3px solid #FFB703;
        padding-bottom: 10px;
    }
    h3 {
        color: #A23B72;
    }
</style>
""", unsafe_allow_html=True)

# القائمة الجانبية
st.sidebar.title("📚 محتويات المحاضرة")
st.sidebar.markdown("---")

sections = {
    "🏠 المقدمة": "intro",
    "📊 السلاسل الزمنية": "timeseries",
    "⚖️ الاستقرارية": "stationarity",
    "✅ السيرورات المستقرة": "stationary",
    "❌ السيرورات غير المستقرة": "non_stationary",
    "🔍 اختبارات الاستقرارية": "tests",
    "🔄 تحويل السلاسل": "transformation",
    "📝 الملخص": "summary"
}

selected_section = st.sidebar.radio("اختر القسم:", list(sections.keys()))

# الصفحة الرئيسية
st.markdown("""
<div class="main-header">
    <h1>📈 محاضرة شاملة في السلاسل الزمنية</h1>
    <h2>Time Series Comprehensive Lecture</h2>
    <p style="font-size: 18px;">الاستقرارية والسيرورات المستقرة وغير المستقرة</p>
    <p style="font-size: 16px;">Stationarity, Stationary and Non-Stationary Processes</p>
</div>
""", unsafe_allow_html=True)

# ======================= المقدمة =======================
if sections[selected_section] == "intro":
    st.header("🏠 المقدمة - Introduction")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="concept-box">
            <h3>🎯 أهداف المحاضرة</h3>
            <ul>
                <li>فهم مفهوم السلاسل الزمنية</li>
                <li>إتقان مفهوم الاستقرارية</li>
                <li>التمييز بين السيرورات المستقرة وغير المستقرة</li>
                <li>معرفة طرق اختبار الاستقرارية</li>
                <li>تعلم كيفية تحويل السلاسل غير المستقرة</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-box">
            <h3>🎯 Lecture Objectives</h3>
            <ul>
                <li>Understanding Time Series Concept</li>
                <li>Mastering Stationarity Concept</li>
                <li>Distinguishing Stationary vs Non-Stationary</li>
                <li>Learning Stationarity Tests</li>
                <li>Learning Transformation Methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="definition-box">
        <h3>📌 أهمية دراسة السلاسل الزمنية</h3>
        <p><strong>العربية:</strong> تُستخدم السلاسل الزمنية في العديد من المجالات مثل الاقتصاد، المالية، الأرصاد الجوية، 
        الطب، والهندسة. فهم خصائص هذه السلاسل ضروري لبناء نماذج تنبؤية دقيقة.</p>
        <p><strong>English:</strong> Time series are used in many fields such as economics, finance, meteorology, 
        medicine, and engineering. Understanding their properties is essential for building accurate predictive models.</p>
    </div>
    """, unsafe_allow_html=True)

# ======================= السلاسل الزمنية =======================
elif sections[selected_section] == "timeseries":
    st.header("📊 السلاسل الزمنية - Time Series")

    st.markdown("""
    <div class="definition-box">
        <h3>📖 التعريف - Definition</h3>
        <p><strong>العربية:</strong> السلسلة الزمنية هي مجموعة من المشاهدات أو القياسات المأخوذة على فترات زمنية منتظمة أو غير منتظمة.</p>
        <p><strong>English:</strong> A time series is a sequence of observations or measurements taken at regular or irregular time intervals.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>الصيغة الرياضية - Mathematical Formula</h4>
        Y = {Y₁, Y₂, Y₃, ..., Yₜ, ..., Yₙ}
        <br>
        حيث t يمثل الزمن (where t represents time)
    </div>
    """, unsafe_allow_html=True)

    # مثال توضيحي
    st.subheader("📈 أمثلة توضيحية - Illustrative Examples")

    # توليد بيانات للأمثلة
    np.random.seed(42)
    time_points = np.arange(0, 100)

    # سلسلة عشوائية
    random_series = np.random.randn(100)

    # سلسلة مع اتجاه
    trend_series = 0.5 * time_points + np.random.randn(100) * 5

    # سلسلة موسمية
    seasonal_series = 10 * np.sin(2 * np.pi * time_points / 12) + np.random.randn(100) * 2

    # سلسلة مع اتجاه وموسمية
    complex_series = 0.3 * time_points + 10 * np.sin(2 * np.pi * time_points / 12) + np.random.randn(100) * 3

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('سلسلة عشوائية - Random',
                        'سلسلة مع اتجاه - Trend',
                        'سلسلة موسمية - Seasonal',
                        'سلسلة معقدة - Complex'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    fig.add_trace(go.Scatter(x=time_points, y=random_series, mode='lines',
                             name='عشوائية', line=dict(color='#2E86AB')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_points, y=trend_series, mode='lines',
                             name='اتجاه', line=dict(color='#A23B72')), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_points, y=seasonal_series, mode='lines',
                             name='موسمية', line=dict(color='#06D6A0')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_points, y=complex_series, mode='lines',
                             name='معقدة', line=dict(color='#FFB703')), row=2, col=2)

    fig.update_layout(height=600, showlegend=False, title_text="أنواع السلاسل الزمنية - Types of Time Series")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="concept-box">
        <h3>🔍 مكونات السلسلة الزمنية - Time Series Components</h3>
        <ol>
            <li><strong>الاتجاه (Trend):</strong> الحركة طويلة المدى في البيانات</li>
            <li><strong>الموسمية (Seasonality):</strong> الأنماط المتكررة على فترات منتظمة</li>
            <li><strong>الدورية (Cyclicity):</strong> التقلبات طويلة المدى غير المنتظمة</li>
            <li><strong>العشوائية (Randomness):</strong> التقلبات غير المنتظمة والعشوائية</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ======================= الاستقرارية =======================
elif sections[selected_section] == "stationarity":
    st.header("⚖️ الاستقرارية - Stationarity")

    st.markdown("""
    <div class="definition-box">
        <h3>📖 التعريف الأساسي - Basic Definition</h3>
        <p><strong>العربية:</strong> السلسلة الزمنية مستقرة إذا كانت خصائصها الإحصائية (المتوسط، التباين، التباين المشترك) 
        لا تتغير مع الزمن.</p>
        <p><strong>English:</strong> A time series is stationary if its statistical properties (mean, variance, covariance) 
        do not change over time.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="concept-box">
            <h3>📋 أنواع الاستقرارية</h3>
            <h4>1. الاستقرارية الضعيفة (Weak Stationarity)</h4>
            <p>تُسمى أيضاً الاستقرارية من الدرجة الثانية</p>
            <p><strong>الشروط:</strong></p>
            <ul>
                <li>المتوسط ثابت: E(Yₜ) = μ</li>
                <li>التباين ثابت: Var(Yₜ) = σ²</li>
                <li>التباين المشترك يعتمد فقط على الفجوة الزمنية</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-box">
            <h3>📋 Types of Stationarity</h3>
            <h4>2. الاستقرارية القوية (Strict Stationarity)</h4>
            <p>Also called Strong Stationarity</p>
            <p><strong>Conditions:</strong></p>
            <ul>
                <li>التوزيع الاحتمالي كاملاً لا يتغير مع الزمن</li>
                <li>Distribution remains unchanged over time</li>
                <li>شرط أقوى وأصعب تحقيقاً</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>الشروط الرياضية للاستقرارية الضعيفة</h4>
        <h4>Mathematical Conditions for Weak Stationarity</h4>
        <p>1. E(Yₜ) = μ  (constant mean)</p>
        <p>2. Var(Yₜ) = E[(Yₜ - μ)²] = σ²  (constant variance)</p>
        <p>3. Cov(Yₜ, Yₜ₊ₖ) = γₖ  (depends only on lag k, not on t)</p>
    </div>
    """, unsafe_allow_html=True)

    # مقارنة بصرية
    st.subheader("📊 مقارنة بصرية - Visual Comparison")

    # سلسلة مستقرة
    stationary = np.random.randn(200)

    # سلسلة غير مستقرة (اتجاه)
    non_stationary_trend = np.cumsum(np.random.randn(200)) * 0.5

    # سلسلة غير مستقرة (تباين متغير)
    non_stationary_var = np.random.randn(200) * (1 + np.arange(200) / 50)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('مستقرة - Stationary',
                        'غير مستقرة (اتجاه) - Non-stationary (Trend)',
                        'غير مستقرة (تباين) - Non-stationary (Variance)'),
        horizontal_spacing=0.08
    )

    fig.add_trace(go.Scatter(y=stationary, mode='lines',
                             name='مستقرة', line=dict(color='#06D6A0', width=2)), row=1, col=1)
    fig.add_hline(y=np.mean(stationary), line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Scatter(y=non_stationary_trend, mode='lines',
                             name='اتجاه', line=dict(color='#EF476F', width=2)), row=1, col=2)

    fig.add_trace(go.Scatter(y=non_stationary_var, mode='lines',
                             name='تباين', line=dict(color='#FFB703', width=2)), row=1, col=3)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ أهمية الاستقرارية - Importance of Stationarity</h3>
        <p><strong>العربية:</strong> معظم نماذج السلاسل الزمنية (مثل ARIMA) تتطلب أن تكون البيانات مستقرة. 
        السلاسل غير المستقرة تجعل التنبؤ صعباً وغير موثوق.</p>
        <p><strong>English:</strong> Most time series models (like ARIMA) require data to be stationary. 
        Non-stationary series make forecasting difficult and unreliable.</p>
    </div>
    """, unsafe_allow_html=True)

# ======================= السيرورات المستقرة =======================
elif sections[selected_section] == "stationary":
    st.header("✅ السيرورات المستقرة - Stationary Processes")

    st.markdown("""
    <div class="definition-box">
        <h3>📖 ما هي السيرورة المستقرة؟</h3>
        <p><strong>العربية:</strong> هي عملية عشوائية تحافظ على خصائصها الإحصائية ثابتة عبر الزمن.</p>
        <p><strong>English:</strong> A stochastic process that maintains constant statistical properties over time.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔹 أمثلة على السيرورات المستقرة")

    # White Noise
    st.markdown("""
    <div class="concept-box">
        <h3>1. الضوضاء البيضاء - White Noise (WN)</h3>
        <p><strong>التعريف:</strong> سلسلة من المتغيرات العشوائية المستقلة والموزعة بشكل متطابق</p>
        <p><strong>Definition:</strong> A sequence of independent and identically distributed random variables</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>خصائص الضوضاء البيضاء - White Noise Properties</h4>
        <p>εₜ ~ WN(0, σ²)</p>
        <p>E(εₜ) = 0</p>
        <p>Var(εₜ) = σ²</p>
        <p>Cov(εₜ, εₛ) = 0  for t ≠ s</p>
    </div>
    """, unsafe_allow_html=True)

    # رسم White Noise
    np.random.seed(123)
    white_noise = np.random.randn(300)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=white_noise, mode='lines',
                             name='White Noise', line=dict(color='#2E86AB')))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title='مثال على الضوضاء البيضاء - White Noise Example',
                      height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Random Walk
    st.markdown("""
    <div class="concept-box">
        <h3>2. المشي العشوائي - Random Walk</h3>
        <p><strong>ملاحظة مهمة:</strong> المشي العشوائي هو سيرورة <strong>غير مستقرة</strong> ولكن يمكن تحويله لسيرورة مستقرة!</p>
        <p><strong>Important Note:</strong> Random Walk is <strong>non-stationary</strong> but can be transformed to stationary!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>صيغة المشي العشوائي - Random Walk Formula</h4>
        <p>Yₜ = Yₜ₋₁ + εₜ</p>
        <p>حيث εₜ ~ WN(0, σ²)</p>
        <br>
        <h4>الفرق الأول (مستقر) - First Difference (Stationary)</h4>
        <p>ΔYₜ = Yₜ - Yₜ₋₁ = εₜ  (White Noise - مستقر!)</p>
    </div>
    """, unsafe_allow_html=True)

    # AR Process
    st.markdown("""
    <div class="concept-box">
        <h3>3. النموذج الانحداري الذاتي - Autoregressive Model (AR)</h3>
        <p><strong>العربية:</strong> سيرورة تعتمد قيمتها الحالية على قيمها السابقة</p>
        <p><strong>English:</strong> A process where current value depends on previous values</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>نموذج AR(1)</h4>
        <p>Yₜ = φYₜ₋₁ + εₜ</p>
        <p>شرط الاستقرارية - Stationarity Condition: |φ| < 1</p>
        <br>
        <h4>نموذج AR(p) العام</h4>
        <p>Yₜ = φ₁Yₜ₋₁ + φ₂Yₜ₋₂ + ... + φₚYₜ₋ₚ + εₜ</p>
    </div>
    """, unsafe_allow_html=True)


    # رسم AR(1)
    def generate_ar1(phi, n=300):
        y = np.zeros(n)
        epsilon = np.random.randn(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + epsilon[t]
        return y


    ar_stationary = generate_ar1(0.7)
    ar_near_unit = generate_ar1(0.95)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('AR(1) مستقر φ=0.7 - Stationary',
                                        'AR(1) قريب من غير مستقر φ=0.95 - Near Non-stationary'))

    fig.add_trace(go.Scatter(y=ar_stationary, mode='lines',
                             name='φ=0.7', line=dict(color='#06D6A0')), row=1, col=1)
    fig.add_trace(go.Scatter(y=ar_near_unit, mode='lines',
                             name='φ=0.95', line=dict(color='#FFB703')), row=1, col=2)

    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # MA Process
    st.markdown("""
    <div class="concept-box">
        <h3>4. نموذج المتوسط المتحرك - Moving Average Model (MA)</h3>
        <p><strong>العربية:</strong> سيرورة تعتمد على قيم الأخطاء السابقة</p>
        <p><strong>English:</strong> A process that depends on past error terms</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>نموذج MA(1)</h4>
        <p>Yₜ = εₜ + θεₜ₋₁</p>
        <p>جميع نماذج MA مستقرة دائماً - All MA models are always stationary</p>
        <br>
        <h4>نموذج MA(q) العام</h4>
        <p>Yₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θₑεₜ₋ₑ</p>
    </div>
    """, unsafe_allow_html=True)

    # ARMA Process
    st.markdown("""
    <div class="concept-box">
        <h3>5. نموذج ARMA</h3>
        <p><strong>العربية:</strong> دمج بين AR و MA</p>
        <p><strong>English:</strong> Combination of AR and MA</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>نموذج ARMA(p,q)</h4>
        <p>Yₜ = φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + εₜ + θ₁εₜ₋₁ + ... + θₑεₜ₋ₑ</p>
        <p>الاستقرارية تعتمد على جزء AR - Stationarity depends on AR part</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h3>✨ خصائص السيرورات المستقرة - Stationary Process Properties</h3>
        <ul>
            <li>المتوسط ثابت عبر الزمن - Constant mean over time</li>
            <li>التباين ثابت عبر الزمن - Constant variance over time</li>
            <li>التباين المشترك يعتمد فقط على الفجوة الزمنية - Autocovariance depends only on lag</li>
            <li>يمكن التنبؤ بها بدقة - Can be forecasted accurately</li>
            <li>تعود للمتوسط (Mean Reverting) - Mean reverting behavior</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================= السيرورات غير المستقرة =======================
elif sections[selected_section] == "non_stationary":
    st.header("❌ السيرورات غير المستقرة - Non-Stationary Processes")

    st.markdown("""
    <div class="definition-box">
        <h3>📖 ما هي السيرورة غير المستقرة؟</h3>
        <p><strong>العربية:</strong> هي عملية عشوائية تتغير خصائصها الإحصائية (المتوسط أو التباين أو كليهما) عبر الزمن.</p>
        <p><strong>English:</strong> A stochastic process whose statistical properties (mean, variance, or both) change over time.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔸 أنواع عدم الاستقرارية - Types of Non-Stationarity")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="warning-box">
            <h3>1. عدم استقرارية في المتوسط</h3>
            <h4>Non-Stationarity in Mean</h4>
            <p>المتوسط يتغير مع الزمن</p>
            <p><strong>أمثلة:</strong></p>
            <ul>
                <li>سلاسل مع اتجاه (Trend)</li>
                <li>المشي العشوائي (Random Walk)</li>
                <li>سلاسل مع تغير هيكلي (Structural Break)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
            <h3>2. عدم استقرارية في التباين</h3>
            <h4>Non-Stationarity in Variance</h4>
            <p>التباين يتغير مع الزمن</p>
            <p><strong>أمثلة:</strong></p>
            <ul>
                <li>Heteroskedasticity</li>
                <li>نماذج ARCH/GARCH</li>
                <li>سلاسل متفجرة (Explosive Series)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # مثال 1: الاتجاه
    st.markdown("""
    <div class="concept-box">
        <h3>🔹 النوع الأول: السلاسل مع الاتجاه - Trend</h3>
        <p><strong>الاتجاه الحتمي (Deterministic Trend):</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>صيغة الاتجاه الخطي - Linear Trend Formula</h4>
        <p>Yₜ = α + βt + εₜ</p>
        <p>حيث:</p>
        <p>α = الثابت (intercept)</p>
        <p>β = معامل الاتجاه (trend coefficient)</p>
        <p>t = الزمن (time)</p>
        <p>εₜ = خطأ عشوائي (random error)</p>
    </div>
    """, unsafe_allow_html=True)

    # رسم أنواع الاتجاهات
    t = np.arange(200)
    linear_trend = 0.5 * t + np.random.randn(200) * 5
    quadratic_trend = 0.01 * t ** 2 + np.random.randn(200) * 10
    exponential_trend = np.exp(0.01 * t) + np.random.randn(200) * 10

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=('اتجاه خطي - Linear Trend',
                                        'اتجاه تربيعي - Quadratic Trend',
                                        'اتجاه أسي - Exponential Trend'))

    fig.add_trace(go.Scatter(x=t, y=linear_trend, mode='lines',
                             name='Linear', line=dict(color='#2E86AB')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=quadratic_trend, mode='lines',
                             name='Quadratic', line=dict(color='#A23B72')), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=exponential_trend, mode='lines',
                             name='Exponential', line=dict(color='#EF476F')), row=1, col=3)

    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # مثال 2: المشي العشوائي
    st.markdown("""
    <div class="concept-box">
        <h3>🔹 النوع الثاني: المشي العشوائي - Random Walk</h3>
        <p><strong>الاتجاه العشوائي (Stochastic Trend):</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>صيغة المشي العشوائي - Random Walk Formula</h4>
        <p>Yₜ = Yₜ₋₁ + εₜ</p>
        <p>أو بالصيغة التراكمية:</p>
        <p>Yₜ = Y₀ + Σεᵢ  (من i=1 إلى t)</p>
        <br>
        <p><strong>مع انحراف (Random Walk with Drift):</strong></p>
        <p>Yₜ = δ + Yₜ₋₁ + εₜ</p>
    </div>
    """, unsafe_allow_html=True)

    # رسم أنواع المشي العشوائي
    np.random.seed(456)
    rw_no_drift = np.cumsum(np.random.randn(200))
    rw_with_drift = np.cumsum(np.random.randn(200) + 0.1)
    rw_negative_drift = np.cumsum(np.random.randn(200) - 0.1)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=('بدون انحراف - No Drift',
                                        'انحراف موجب - Positive Drift',
                                        'انحراف سالب - Negative Drift'))

    fig.add_trace(go.Scatter(y=rw_no_drift, mode='lines',
                             name='No Drift', line=dict(color='#06D6A0')), row=1, col=1)
    fig.add_trace(go.Scatter(y=rw_with_drift, mode='lines',
                             name='Positive', line=dict(color='#2E86AB')), row=1, col=2)
    fig.add_trace(go.Scatter(y=rw_negative_drift, mode='lines',
                             name='Negative', line=dict(color='#EF476F')), row=1, col=3)

    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ خصائص المشي العشوائي - Random Walk Properties</h3>
        <ul>
            <li>المتوسط: E(Yₜ) = Y₀ (أو Y₀ + δt مع الانحراف)</li>
            <li>التباين: Var(Yₜ) = tσ² (يزداد مع الزمن!)</li>
            <li>غير مستقر لأن التباين يزداد مع الزمن</li>
            <li>الفرق الأول مستقر: ΔYₜ = εₜ</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # مثال 3: التباين المتغير
    st.markdown("""
    <div class="concept-box">
        <h3>🔹 النوع الثالث: التباين المتغير - Changing Variance</h3>
        <p><strong>Heteroskedasticity:</strong> التباين يتغير عبر الزمن</p>
    </div>
    """, unsafe_allow_html=True)

    # رسم أمثلة التباين المتغير
    increasing_var = np.random.randn(200) * (1 + np.arange(200) / 50)
    arch_like = np.zeros(200)
    h = np.zeros(200)
    h[0] = 1
    for i in range(1, 200):
        h[i] = 0.1 + 0.85 * h[i - 1] + 0.05 * arch_like[i - 1] ** 2
        arch_like[i] = np.random.randn() * np.sqrt(h[i])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('تباين متزايد - Increasing Variance',
                                        'نموذج ARCH - ARCH-like'))

    fig.add_trace(go.Scatter(y=increasing_var, mode='lines',
                             name='Increasing', line=dict(color='#FFB703')), row=1, col=1)
    fig.add_trace(go.Scatter(y=arch_like, mode='lines',
                             name='ARCH', line=dict(color='#A23B72')), row=1, col=2)

    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # التكامل
    st.markdown("""
    <div class="concept-box">
        <h3>🔹 مفهوم التكامل - Integration</h3>
        <p><strong>السلسلة المتكاملة من الدرجة d: I(d)</strong></p>
        <p>Integrated of order d</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>تعريف I(d)</h4>
        <p>السلسلة Yₜ متكاملة من الدرجة d إذا كان:</p>
        <p>ΔᵈYₜ ~ I(0)  (مستقرة)</p>
        <br>
        <p><strong>أمثلة:</strong></p>
        <p>• I(0): سلسلة مستقرة (Stationary)</p>
        <p>• I(1): تحتاج فرق واحد لتصبح مستقرة (المشي العشوائي)</p>
        <p>• I(2): تحتاج فرقين لتصبح مستقرة</p>
    </div>
    """, unsafe_allow_html=True)

    # مشاكل عدم الاستقرارية
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ مشاكل عدم الاستقرارية - Problems of Non-Stationarity</h3>
        <ol>
            <li><strong>الانحدار الزائف (Spurious Regression):</strong> علاقات كاذبة بين المتغيرات</li>
            <li><strong>التنبؤ غير الموثوق:</strong> التنبؤات تصبح أقل دقة مع الزمن</li>
            <li><strong>الاختبارات الإحصائية غير صحيحة:</strong> اختبارات t و F غير صالحة</li>
            <li><strong>فترات الثقة واسعة جداً:</strong> عدم يقين كبير في التقديرات</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ======================= اختبارات الاستقرارية =======================
elif sections[selected_section] == "tests":
    st.header("🔍 اختبارات الاستقرارية - Stationarity Tests")

    st.markdown("""
    <div class="definition-box">
        <h3>📖 لماذا نحتاج اختبارات الاستقرارية؟</h3>
        <p><strong>العربية:</strong> للتحقق بشكل إحصائي من استقرارية السلسلة الزمنية قبل بناء النماذج.</p>
        <p><strong>English:</strong> To statistically verify the stationarity of a time series before building models.</p>
    </div>
    """, unsafe_allow_html=True)

    # الطرق البصرية
    st.subheader("👁️ الطرق البصرية - Visual Methods")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="concept-box">
            <h3>1. الرسم البياني الزمني</h3>
            <h4>Time Series Plot</h4>
            <p><strong>ما نبحث عنه:</strong></p>
            <ul>
                <li>ثبات المتوسط</li>
                <li>ثبات التباين</li>
                <li>عدم وجود اتجاه واضح</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-box">
            <h3>2. دالة الارتباط الذاتي</h3>
            <h4>ACF - Autocorrelation Function</h4>
            <p><strong>للسلسلة المستقرة:</strong></p>
            <ul>
                <li>تتلاشى بسرعة</li>
                <li>تقترب من الصفر</li>
            </ul>
            <p><strong>لغير المستقرة:</strong></p>
            <ul>
                <li>تتلاشى ببطء شديد</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # مثال ACF
    np.random.seed(789)
    stationary_series = np.random.randn(200)
    non_stationary_series = np.cumsum(np.random.randn(200))


    def calculate_acf(series, nlags=30):
        acf_values = []
        for lag in range(nlags + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                c0 = np.var(series)
                c_lag = np.correlate(series[:-lag] - np.mean(series),
                                     series[lag:] - np.mean(series), mode='valid')[0] / len(series)
                acf_values.append(c_lag / c0)
        return acf_values


    acf_stat = calculate_acf(stationary_series)
    acf_nonstat = calculate_acf(non_stationary_series)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('سلسلة مستقرة - Stationary Series',
                                        'ACF للمستقرة',
                                        'سلسلة غير مستقرة - Non-Stationary Series',
                                        'ACF لغير المستقرة'),
                        vertical_spacing=0.12)

    fig.add_trace(go.Scatter(y=stationary_series, mode='lines',
                             line=dict(color='#06D6A0')), row=1, col=1)
    fig.add_trace(go.Bar(y=acf_stat, marker_color='#06D6A0'), row=1, col=2)
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=-0.2, line_dash="dash", line_color="red", row=1, col=2)

    fig.add_trace(go.Scatter(y=non_stationary_series, mode='lines',
                             line=dict(color='#EF476F')), row=2, col=1)
    fig.add_trace(go.Bar(y=acf_nonstat, marker_color='#EF476F'), row=2, col=2)
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-0.2, line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # الاختبارات الإحصائية
    st.subheader("📊 الاختبارات الإحصائية - Statistical Tests")

    # 1. اختبار ADF
    st.markdown("""
    <div class="concept-box">
        <h3>1. اختبار ديكي-فولر المعزز</h3>
        <h4>Augmented Dickey-Fuller (ADF) Test</h4>
        <p><strong>الغرض:</strong> اختبار وجود جذر الوحدة (Unit Root)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>معادلة اختبار ADF</h4>
        <p>ΔYₜ = α + βt + γYₜ₋₁ + Σδᵢ ΔYₜ₋ᵢ + εₜ</p>
        <br>
        <h4>الفرضيات - Hypotheses</h4>
        <p>H₀: γ = 0  (يوجد جذر وحدة، السلسلة غير مستقرة)</p>
        <p>H₁: γ < 0  (لا يوجد جذر وحدة، السلسلة مستقرة)</p>
        <br>
        <h4>القرار - Decision</h4>
        <p>إذا كانت p-value < 0.05 → نرفض H₀ → السلسلة مستقرة</p>
        <p>إذا كانت p-value ≥ 0.05 → نقبل H₀ → السلسلة غير مستقرة</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h3>📝 تفسير نتائج ADF</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #2E86AB; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd;">ADF Statistic</th>
                <th style="padding: 10px; border: 1px solid #ddd;">p-value</th>
                <th style="padding: 10px; border: 1px solid #ddd;">النتيجة - Result</th>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">-4.5</td>
                <td style="padding: 10px; border: 1px solid #ddd;">0.0001</td>
                <td style="padding: 10px; border: 1px solid #ddd; color: green;">✅ مستقرة - Stationary</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 10px; border: 1px solid #ddd;">-2.5</td>
                <td style="padding: 10px; border: 1px solid #ddd;">0.12</td>
                <td style="padding: 10px; border: 1px solid #ddd; color: red;">❌ غير مستقرة - Non-Stationary</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">-1.8</td>
                <td style="padding: 10px; border: 1px solid #ddd;">0.38</td>
                <td style="padding: 10px; border: 1px solid #ddd; color: red;">❌ غير مستقرة - Non-Stationary</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 2. اختبار KPSS
    st.markdown("""
    <div class="concept-box">
        <h3>2. اختبار كواياتكوفسكي-فيليبس-شميت-شين</h3>
        <h4>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</h4>
        <p><strong>الغرض:</strong> اختبار الاستقرارية (عكس ADF)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>الفرضيات - Hypotheses</h4>
        <p>H₀: السلسلة مستقرة (Stationary)</p>
        <p>H₁: السلسلة غير مستقرة (Non-Stationary)</p>
        <br>
        <h4>القرار - Decision</h4>
        <p>إذا كانت p-value > 0.05 → نقبل H₀ → السلسلة مستقرة</p>
        <p>إذا كانت p-value ≤ 0.05 → نرفض H₀ → السلسلة غير مستقرة</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. اختبار PP
    st.markdown("""
    <div class="concept-box">
        <h3>3. اختبار فيليبس-بيرون</h3>
        <h4>Phillips-Perron (PP) Test</h4>
        <p><strong>الميزة:</strong> أقوى من ADF في حالة وجود ارتباط ذاتي</p>
        <p>الفرضيات مشابهة لـ ADF</p>
    </div>
    """, unsafe_allow_html=True)

    # جدول مقارنة الاختبارات
    st.markdown("""
    <div class="example-box">
        <h3>⚖️ مقارنة الاختبارات - Comparison of Tests</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #2E86AB; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd;">الاختبار - Test</th>
                <th style="padding: 10px; border: 1px solid #ddd;">H₀</th>
                <th style="padding: 10px; border: 1px solid #ddd;">الاستخدام - Use</th>
                <th style="padding: 10px; border: 1px solid #ddd;">الميزة - Advantage</th>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>ADF</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">غير مستقرة</td>
                <td style="padding: 10px; border: 1px solid #ddd;">الأكثر شيوعاً</td>
                <td style="padding: 10px; border: 1px solid #ddd;">يعالج الارتباط الذاتي</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>KPSS</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">مستقرة</td>
                <td style="padding: 10px; border: 1px solid #ddd;">مكمل لـ ADF</td>
                <td style="padding: 10px; border: 1px solid #ddd;">تأكيد مزدوج</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>PP</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">غير مستقرة</td>
                <td style="padding: 10px; border: 1px solid #ddd;">بديل لـ ADF</td>
                <td style="padding: 10px; border: 1px solid #ddd;">قوي مع التباين غير المتجانس</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # استراتيجية الاختبار
    st.markdown("""
    <div class="warning-box">
        <h3>🎯 استراتيجية الاختبار الموصى بها</h3>
        <h4>Recommended Testing Strategy</h4>
        <ol>
            <li><strong>الخطوة 1:</strong> ابدأ بالفحص البصري (رسم السلسلة و ACF)</li>
            <li><strong>الخطوة 2:</strong> طبق اختبار ADF</li>
            <li><strong>الخطوة 3:</strong> طبق اختبار KPSS للتأكيد</li>
            <li><strong>الخطوة 4:</strong> إذا كانت النتائج متناقضة، طبق PP</li>
        </ol>
        <br>
        <h4>جدول القرارات - Decision Table</h4>
        <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
            <tr style="background-color: #2E86AB; color: white;">
                <th style="padding: 8px; border: 1px solid #ddd;">ADF</th>
                <th style="padding: 8px; border: 1px solid #ddd;">KPSS</th>
                <th style="padding: 8px; border: 1px solid #ddd;">القرار - Decision</th>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">مستقرة</td>
                <td style="padding: 8px; border: 1px solid #ddd;">مستقرة</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #d4edda;">✅ مستقرة بوضوح</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">غير مستقرة</td>
                <td style="padding: 8px; border: 1px solid #ddd;">غير مستقرة</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #f8d7da;">❌ غير مستقرة بوضوح</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">مستقرة</td>
                <td style="padding: 8px; border: 1px solid #ddd;">غير مستقرة</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #fff3cd;">⚠️ مستقرة حول اتجاه</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">غير مستقرة</td>
                <td style="padding: 8px; border: 1px solid #ddd;">مستقرة</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #fff3cd;">⚠️ حالة نادرة - مزيد من الاختبارات</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ======================= تحويل السلاسل =======================
elif sections[selected_section] == "transformation":
    st.header("🔄 تحويل السلاسل غير المستقرة - Transforming Non-Stationary Series")

    st.markdown("""
    <div class="definition-box">
        <h3>📖 لماذا نحول السلاسل؟</h3>
        <p><strong>العربية:</strong> لتحويل السلسلة غير المستقرة إلى مستقرة حتى نتمكن من بناء نماذج تنبؤية دقيقة.</p>
        <p><strong>English:</strong> To transform non-stationary series into stationary ones for accurate predictive modeling.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. الفرق - Differencing
    st.subheader("1️⃣ الفرق - Differencing")

    st.markdown("""
    <div class="concept-box">
        <h3>الفرق من الدرجة الأولى - First Differencing</h3>
        <p><strong>الأكثر استخداماً لإزالة الاتجاه</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>صيغة الفرق الأول - First Difference Formula</h4>
        <p>ΔYₜ = Yₜ - Yₜ₋₁</p>
        <br>
        <p><strong>الفرق من الدرجة الثانية - Second Difference:</strong></p>
        <p>Δ²Yₜ = ΔYₜ - ΔYₜ₋₁ = (Yₜ - Yₜ₋₁) - (Yₜ₋₁ - Yₜ₋₂)</p>
        <br>
        <p><strong>متى نستخدم كل منهما؟</strong></p>
        <p>• الفرق الأول: للسلاسل I(1)</p>
        <p>• الفرق الثاني: للسلاسل I(2)</p>
    </div>
    """, unsafe_allow_html=True)

    # مثال على الفرق
    np.random.seed(111)
    original = np.cumsum(np.random.randn(200)) + 0.1 * np.arange(200)
    first_diff = np.diff(original)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('السلسلة الأصلية - Original Series',
                                        'بعد الفرق الأول - After First Differencing'))

    fig.add_trace(go.Scatter(y=original, mode='lines',
                             line=dict(color='#EF476F', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=first_diff, mode='lines',
                             line=dict(color='#06D6A0', width=2)), row=1, col=2)
    fig.add_hline(y=np.mean(first_diff), line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 2. الفرق الموسمي
    st.markdown("""
    <div class="concept-box">
        <h3>الفرق الموسمي - Seasonal Differencing</h3>
        <p><strong>لإزالة الموسمية</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>صيغة الفرق الموسمي - Seasonal Difference Formula</h4>
        <p>ΔₛYₜ = Yₜ - Yₜ₋ₛ</p>
        <p>حيث s هي الدورة الموسمية (where s is the seasonal period)</p>
        <br>
        <p><strong>أمثلة:</strong></p>
        <p>• بيانات شهرية: s = 12</p>
        <p>• بيانات ربع سنوية: s = 4</p>
        <p>• بيانات يومية أسبوعية: s = 7</p>
    </div>
    """, unsafe_allow_html=True)

    # مثال موسمي
    t = np.arange(120)
    seasonal = 50 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 2
    seasonal_diff = seasonal[12:] - seasonal[:-12]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('سلسلة موسمية - Seasonal Series',
                                        'بعد الفرق الموسمي (s=12) - After Seasonal Differencing'))

    fig.add_trace(go.Scatter(x=t, y=seasonal, mode='lines',
                             line=dict(color='#A23B72', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[12:], y=seasonal_diff, mode='lines',
                             line=dict(color='#06D6A0', width=2)), row=1, col=2)
    fig.add_hline(y=np.mean(seasonal_diff), line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 3. التحويل اللوغاريتمي
    st.subheader("2️⃣ التحويل اللوغاريتمي - Logarithmic Transformation")

    st.markdown("""
    <div class="concept-box">
        <h3>التحويل اللوغاريتمي</h3>
        <p><strong>لتثبيت التباين المتزايد</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>صيغة التحويل اللوغاريتمي</h4>
        <p>Yₜ* = log(Yₜ)  أو  ln(Yₜ)</p>
        <br>
        <p><strong>متى نستخدمه؟</strong></p>
        <p>• عندما يزداد التباين مع المستوى</p>
        <p>• في السلاسل الأسية</p>
        <p>• البيانات المالية والاقتصادية</p>
    </div>
    """, unsafe_allow_html=True)

    # مثال لوغاريتمي
    exp_series = np.exp(0.05 * np.arange(100)) * (1 + 0.2 * np.random.randn(100))
    log_series = np.log(exp_series)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('سلسلة أسية - Exponential Series',
                                        'بعد التحويل اللوغاريتمي - After Log Transformation'))

    fig.add_trace(go.Scatter(y=exp_series, mode='lines',
                             line=dict(color='#FFB703', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=log_series, mode='lines',
                             line=dict(color='#2E86AB', width=2)), row=1, col=2)

    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 4. تحويل Box-Cox
    st.subheader("3️⃣ تحويل بوكس-كوكس - Box-Cox Transformation")

    st.markdown("""
    <div class="concept-box">
        <h3>تحويل Box-Cox</h3>
        <p><strong>تحويل أعم من اللوغاريتمي</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h4>صيغة Box-Cox</h4>
        <p>Yₜ(λ) = {</p>
        <p style="margin-right: 40px;">(Yₜ^λ - 1) / λ,  if λ ≠ 0</p>
        <p style="margin-right: 40px;">ln(Yₜ),  if λ = 0</p>
        <p>}</p>
        <br>
        <p><strong>قيم λ الشائعة:</strong></p>
        <p>• λ = 1: لا تحويل</p>
        <p>• λ = 0.5: جذر تربيعي</p>
        <p>• λ = 0: لوغاريتمي</p>
        <p>• λ = -1: معكوس</p>
    </div>
    """, unsafe_allow_html=True)

    # 5. إزالة الاتجاه
    st.subheader("4️⃣ إزالة الاتجاه - Detrending")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="concept-box">
            <h3>الطريقة 1: الانحدار الخطي</h3>
            <h4>Linear Regression</h4>
            <p>نقدر معادلة الاتجاه ثم نطرحها:</p>
            <p>Ŷₜ = α + βt</p>
            <p>Residuals = Yₜ - Ŷₜ</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-box">
            <h3>الطريقة 2: المتوسط المتحرك</h3>
            <h4>Moving Average</h4>
            <p>نطرح المتوسط المتحرك:</p>
            <p>MAₜ = (Yₜ₋ₖ + ... + Yₜ + ... + Yₜ₊ₖ) / (2k+1)</p>
            <p>Detrended = Yₜ - MAₜ</p>
        </div>
        """, unsafe_allow_html=True)

    # جدول ملخص التحويلات
    st.markdown("""
    <div class="example-box">
        <h3>📋 ملخص طرق التحويل - Summary of Transformation Methods</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #2E86AB; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd;">الطريقة - Method</th>
                <th style="padding: 10px; border: 1px solid #ddd;">الاستخدام - Use Case</th>
                <th style="padding: 10px; border: 1px solid #ddd;">المعادلة - Formula</th>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>الفرق الأول</strong><br>First Difference</td>
                <td style="padding: 10px; border: 1px solid #ddd;">اتجاه خطي</td>
                <td style="padding: 10px; border: 1px solid #ddd;">ΔYₜ = Yₜ - Yₜ₋₁</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>الفرق الموسمي</strong><br>Seasonal Diff</td>
                <td style="padding: 10px; border: 1px solid #ddd;">موسمية</td>
                <td style="padding: 10px; border: 1px solid #ddd;">ΔₛYₜ = Yₜ - Yₜ₋ₛ</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>اللوغاريتمي</strong><br>Log Transform</td>
                <td style="padding: 10px; border: 1px solid #ddd;">تباين متزايد</td>
                <td style="padding: 10px; border: 1px solid #ddd;">log(Yₜ)</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Box-Cox</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">تثبيت تباين عام</td>
                <td style="padding: 10px; border: 1px solid #ddd;">(Yₜ^λ - 1) / λ</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>إزالة الاتجاه</strong><br>Detrending</td>
                <td style="padding: 10px; border: 1px solid #ddd;">اتجاه حتمي</td>
                <td style="padding: 10px; border: 1px solid #ddd;">Yₜ - Trend</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ نصائح مهمة - Important Tips</h3>
        <ol>
            <li><strong>ابدأ بالبسيط:</strong> جرب الفرق الأول أولاً</li>
            <li><strong>لا تفرط في التحويل:</strong> تجنب الفرق أكثر من مرتين</li>
            <li><strong>اختبر بعد كل تحويل:</strong> استخدم اختبارات الاستقرارية</li>
            <li><strong>احفظ التحويلات:</strong> لعكسها عند التنبؤ</li>
            <li><strong>تحويل + فرق:</strong> يمكن الجمع (مثل log ثم difference)</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ======================= الملخص =======================
elif sections[selected_section] == "summary":
    st.header("📝 ملخص المحاضرة - Lecture Summary")

    st.markdown("""
    <div class="main-header">
        <h2>🎓 الأفكار الرئيسية - Key Concepts</h2>
    </div>
    """, unsafe_allow_html=True)

    # النقاط الرئيسية
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="concept-box">
            <h3>📊 السلاسل الزمنية</h3>
            <ul>
                <li>بيانات مرتبة زمنياً</li>
                <li>لها مكونات: اتجاه، موسمية، دورية، عشوائية</li>
                <li>تستخدم للتنبؤ وفهم الأنماط</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-box">
            <h3>✅ السيرورات المستقرة</h3>
            <ul>
                <li>المتوسط ثابت</li>
                <li>التباين ثابت</li>
                <li>التباين المشترك يعتمد على الفجوة فقط</li>
                <li>أمثلة: WN, AR(p), MA(q), ARMA</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-box">
            <h3>🔍 الاختبارات</h3>
            <ul>
                <li>ADF: H₀ غير مستقرة</li>
                <li>KPSS: H₀ مستقرة</li>
                <li>PP: بديل قوي لـ ADF</li>
                <li>استخدم أكثر من اختبار للتأكيد</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-box">
            <h3>⚖️ الاستقرارية</h3>
            <ul>
                <li>ضعيفة: شروط على المتوسط والتباين</li>
                <li>قوية: التوزيع كاملاً ثابت</li>
                <li>ضرورية لمعظم النماذج</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-box">
            <h3>❌ السيرورات غير المستقرة</h3>
            <ul>
                <li>المتوسط أو التباين يتغير</li>
                <li>أمثلة: اتجاه، Random Walk، I(d)</li>
                <li>تسبب مشاكل في النمذجة</li>
                <li>تحتاج تحويل للاستقرارية</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-box">
            <h3>🔄 التحويلات</h3>
            <ul>
                <li>الفرق: لإزالة الاتجاه</li>
                <li>الفرق الموسمي: لإزالة الموسمية</li>
                <li>اللوغاريتمي: لتثبيت التباين</li>
                <li>Box-Cox: تحويل عام</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # خارطة طريق
    st.markdown("""
    <div class="example-box">
        <h3>🗺️ خارطة طريق التحليل - Analysis Roadmap</h3>
        <div style="background: white; padding: 20px; border-radius: 10px; margin-top: 15px;">
            <ol style="font-size: 16px; line-height: 2;">
                <li><strong>رسم السلسلة</strong> - Plot the series</li>
                <li><strong>رسم ACF و PACF</strong> - Plot ACF and PACF</li>
                <li><strong>اختبار الاستقرارية</strong> (ADF, KPSS) - Test stationarity</li>
                <li><strong>إذا غير مستقرة:</strong> طبق التحويلات - If non-stationary: apply transformations</li>
                <li><strong>أعد الاختبار</strong> - Re-test</li>
                <li><strong>بناء النموذج</strong> - Build model</li>
                <li><strong>التحقق من البواقي</strong> - Check residuals</li>
                <li><strong>التنبؤ</strong> - Forecast</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # المعادلات المهمة
    st.markdown("""
    <div class="formula-box">
        <h3>📐 المعادلات الأساسية - Key Formulas</h3>
        <div style="text-align: right; direction: rtl;">
            <p><strong>1. شروط الاستقرارية الضعيفة:</strong></p>
            <p>E(Yₜ) = μ,  Var(Yₜ) = σ²,  Cov(Yₜ, Yₜ₊ₖ) = γₖ</p>
            <br>
            <p><strong>2. الضوضاء البيضاء:</strong></p>
            <p>εₜ ~ WN(0, σ²),  E(εₜ) = 0,  Cov(εₜ, εₛ) = 0 for t≠s</p>
            <br>
            <p><strong>3. AR(1):</strong></p>
            <p>Yₜ = φYₜ₋₁ + εₜ,  شرط الاستقرارية: |φ| < 1</p>
            <br>
            <p><strong>4. المشي العشوائي:</strong></p>
            <p>Yₜ = Yₜ₋₁ + εₜ  (I(1)),  ΔYₜ = εₜ  (I(0))</p>
            <br>
            <p><strong>5. الفرق الأول:</strong></p>
            <p>ΔYₜ = Yₜ - Yₜ₋₁</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # الرسم التوضيحي النهائي
    st.subheader("📊 الرسم التوضيحي الشامل - Comprehensive Illustration")

    # إنشاء أمثلة متنوعة
    np.random.seed(2024)
    t = np.arange(150)

    stationary_ex = np.random.randn(150)
    trend_ex = 0.3 * t + np.random.randn(150) * 3
    random_walk_ex = np.cumsum(np.random.randn(150))
    seasonal_ex = 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(150) * 2

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('مستقرة (WN) - Stationary',
                        'غير مستقرة (اتجاه) - Non-stationary (Trend)',
                        'غير مستقرة (RW) - Non-stationary (RW)',
                        'مستقرة (موسمية) - Stationary (Seasonal)'),
        vertical_spacing=0.12
    )

    fig.add_trace(go.Scatter(y=stationary_ex, mode='lines',
                             line=dict(color='#06D6A0', width=2)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Scatter(y=trend_ex, mode='lines',
                             line=dict(color='#EF476F', width=2)), row=1, col=2)

    fig.add_trace(go.Scatter(y=random_walk_ex, mode='lines',
                             line=dict(color='#FFB703', width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(y=seasonal_ex, mode='lines',
                             line=dict(color='#2E86AB', width=2)), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(height=600, showlegend=False,
                      title_text="أمثلة على أنواع مختلفة من السلاسل - Examples of Different Series Types")
    st.plotly_chart(fig, use_container_width=True)

    # الخاتمة
    st.markdown("""
    <div class="main-header">
        <h2>🎯 النقاط الأساسية للتذكر - Key Takeaways</h2>
        <ul style="text-align: right; font-size: 18px; line-height: 2;">
            <li>الاستقرارية أساسية لنمذجة السلاسل الزمنية</li>
            <li>استخدم اختبارات متعددة للتحقق من الاستقرارية</li>
            <li>الفرق هو الطريقة الأكثر شيوعاً للتحويل</li>
            <li>فهم الفرق بين الاتجاه الحتمي والعشوائي</li>
            <li>السلاسل I(1) شائعة جداً في البيانات الواقعية</li>
            <li>تحقق دائماً من البواقي بعد النمذجة</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="definition-box" style="text-align: center; margin-top: 30px;">
        <h2>🙏 شكراً لمتابعتكم</h2>
        <h3>Thank You for Your Attention</h3>
        <p style="font-size: 18px; margin-top: 20px;">
        هذه المحاضرة غطت المفاهيم الأساسية للسلاسل الزمنية والاستقرارية<br>
        This lecture covered fundamental concepts of time series and stationarity
        </p>
    </div>
    """, unsafe_allow_html=True)

# معلومات إضافية في الشريط الجانبي
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 مراجع إضافية
**Additional References**

- Box, G. E., & Jenkins, G. M. (1976)
- Hamilton, J. D. (1994)
- Enders, W. (2014)
- Hyndman, R. J., & Athanasopoulos, G. (2018)
""")

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **نصيحة:**
استخدم الأسهم ← → للتنقل بين الأقسام
""")