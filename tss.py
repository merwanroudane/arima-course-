import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

# إعداد الصفحة
st.set_page_config(page_title="السلاسل الزمنية - المحاضرة الأولى", layout="wide", initial_sidebar_state="expanded")

# العنوان الرئيسي
st.title("📊 السلاسل الزمنية - المحاضرة الأولى")
st.markdown("### Time Series - First Lecture")
st.markdown("---")

# الشريط الجانبي
with st.sidebar:
    st.header("🎯 المحتويات - Contents")
    section = st.radio(
        "اختر القسم - Select Section:",
        ["التعريف - Definition",
         "المصطلحات الأساسية - Basic Terms",
         "أنواع السلاسل الزمنية - Types",
         "الخصائص - Properties",
         "الاستقرارية - Stationarity",
         "السيرورات - Processes",
         "اللاحطية - Determinism",
         "التغيرات الهيكلية - Structural Changes"]
    )

    st.markdown("---")
    st.info("💡 استخدم الأزرار للتنقل بين الأقسام المختلفة")

# القسم 1: التعريف
if section == "التعريف - Definition":
    st.header("📖 تعريف السلاسل الزمنية")
    st.markdown("### Definition of Time Series")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;'>
        <h4 style='color: #1976D2;'>التعريف بالعربية</h4>
        <p style='font-size: 16px; line-height: 1.8;'>
        السلسلة الزمنية هي مجموعة من القيم أو الملاحظات المرتبة زمنياً والمسجلة على فترات منتظمة أو غير منتظمة.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color: #f3e5f5; padding: 20px; border-radius: 10px; border-left: 5px solid #9C27B0;'>
        <h4 style='color: #7B1FA2;'>English Definition</h4>
        <p style='font-size: 16px; line-height: 1.8;'>
        A time series is a sequence of observations or values ordered in time and recorded at regular or irregular intervals.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # الصيغة الرياضية
    st.subheader("الصيغة الرياضية - Mathematical Notation")
    st.latex(r"Y_t = \{y_1, y_2, y_3, ..., y_t, ..., y_T\}")
    st.latex(r"\text{حيث (where): } t = 1, 2, 3, ..., T")

    # رسم توضيحي
    t = np.arange(0, 100)
    y = 10 + 2 * np.sin(0.1 * t) + np.random.normal(0, 0.5, 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines+markers',
                             line=dict(color='#2196F3', width=2),
                             marker=dict(size=6, color='#FF5722'),
                             name='السلسلة الزمنية'))
    fig.update_layout(
        title='مثال على سلسلة زمنية - Example of Time Series',
        xaxis_title='الزمن (t) - Time',
        yaxis_title='القيمة (Y) - Value',
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

# القسم 2: المصطلحات الأساسية
elif section == "المصطلحات الأساسية - Basic Terms":
    st.header("📚 المصطلحات الأساسية")
    st.markdown("### Basic Terms in Time Series")

    # جدول المصطلحات
    terms_data = {
        'المصطلح بالعربية': [
            'الملاحظة', 'الفترة الزمنية', 'التردد', 'الاتجاه العام',
            'الموسمية', 'الدورية', 'العشوائية', 'المستوى'
        ],
        'English Term': [
            'Observation', 'Time Period', 'Frequency', 'Trend',
            'Seasonality', 'Cyclical', 'Random/Irregular', 'Level'
        ],
        'الرمز - Symbol': [
            r'$Y_t$', r'$t$', r'$f$', r'$T_t$',
            r'$S_t$', r'$C_t$', r'$\varepsilon_t$', r'$\mu$'
        ],
        'الشرح - Explanation': [
            'قيمة واحدة في السلسلة - Single value in series',
            'نقطة زمنية محددة - Specific time point',
            'عدد الملاحظات في الوحدة الزمنية - Number of observations per time unit',
            'الاتجاه طويل المدى - Long-term direction',
            'نمط متكرر خلال فترة ثابتة - Repeating pattern over fixed period',
            'تذبذبات طويلة المدى - Long-term fluctuations',
            'تقلبات غير منتظمة - Irregular fluctuations',
            'المتوسط الثابت - Constant mean'
        ]
    }

    df_terms = pd.DataFrame(terms_data)
    st.dataframe(df_terms, use_container_width=True, height=350)

    st.markdown("---")

    # التحلل الكلاسيكي
    st.subheader("التحلل الكلاسيكي - Classical Decomposition")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### النموذج الجمعي - Additive Model")
        st.latex(r"Y_t = T_t + S_t + C_t + \varepsilon_t")
        st.info("يُستخدم عندما تكون السعة الموسمية ثابتة")

    with col2:
        st.markdown("#### النموذج الضربي - Multiplicative Model")
        st.latex(r"Y_t = T_t \times S_t \times C_t \times \varepsilon_t")
        st.info("يُستخدم عندما تكون السعة الموسمية متغيرة")

    # رسم توضيحي للمكونات
    t = np.arange(0, 200)
    trend = 0.05 * t + 10
    seasonal = 3 * np.sin(2 * np.pi * t / 20)
    cyclical = 2 * np.sin(2 * np.pi * t / 80)
    random = np.random.normal(0, 0.5, 200)
    y_combined = trend + seasonal + cyclical + random

    fig = make_subplots(rows=5, cols=1,
                        subplot_titles=('السلسلة الكاملة - Complete Series',
                                        'الاتجاه - Trend',
                                        'الموسمية - Seasonality',
                                        'الدورية - Cyclical',
                                        'العشوائية - Random'))

    fig.add_trace(go.Scatter(x=t, y=y_combined, mode='lines', line=dict(color='#2196F3'), name='Y_t'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=trend, mode='lines', line=dict(color='#4CAF50'), name='T_t'), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=seasonal, mode='lines', line=dict(color='#FF9800'), name='S_t'), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=cyclical, mode='lines', line=dict(color='#9C27B0'), name='C_t'), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=random, mode='lines', line=dict(color='#F44336'), name='ε_t'), row=5, col=1)

    fig.update_layout(height=1000, showlegend=False, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# القسم 3: أنواع السلاسل الزمنية
elif section == "أنواع السلاسل الزمنية - Types":
    st.header("🔢 أنواع السلاسل الزمنية")
    st.markdown("### Types of Time Series")

    tab1, tab2, tab3 = st.tabs(["حسب الطبيعة - By Nature",
                                "حسب البيانات - By Data",
                                "حسب السلوك - By Behavior"])

    with tab1:
        st.subheader("حسب طبيعة البيانات - By Nature of Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #2E7D32;'>1. سلاسل متصلة - Continuous Series</h4>
            <p>يمكن أن تأخذ أي قيمة في نطاق معين</p>
            <p><b>أمثلة:</b> درجة الحرارة، الضغط الجوي، سعر الصرف</p>
            </div>
            """, unsafe_allow_html=True)

            # رسم سلسلة متصلة
            t1 = np.linspace(0, 10, 1000)
            y1 = 20 + 5 * np.sin(t1) + np.random.normal(0, 0.3, 1000)
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=t1, y=y1, mode='lines', line=dict(color='#4CAF50')))
            fig1.update_layout(title='سلسلة متصلة - Continuous', height=300, template='plotly_white')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("""
            <div style='background-color: #fff3e0; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #E65100;'>2. سلاسل منفصلة - Discrete Series</h4>
            <p>تأخذ قيماً محددة ومنفصلة</p>
            <p><b>أمثلة:</b> عدد الزبائن، المبيعات اليومية، عدد الحوادث</p>
            </div>
            """, unsafe_allow_html=True)

            # رسم سلسلة منفصلة
            t2 = np.arange(0, 50)
            y2 = np.random.poisson(10, 50)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=t2, y=y2, marker_color='#FF9800'))
            fig2.update_layout(title='سلسلة منفصلة - Discrete', height=300, template='plotly_white')
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("حسب نوع البيانات - By Data Type")

        types_data = {
            'النوع - Type': [
                'أحادية المتغير - Univariate',
                'متعددة المتغيرات - Multivariate',
                'متجهة - Vector',
                'لوحية - Panel'
            ],
            'الوصف - Description': [
                'متغير واحد عبر الزمن - One variable over time',
                'عدة متغيرات عبر الزمن - Multiple variables over time',
                'مجموعة من السلاسل المترابطة - Set of related series',
                'بيانات مقطعية وزمنية - Cross-sectional and temporal data'
            ],
            'الرمز الرياضي - Mathematical Notation': [
                r'$Y_t$',
                r'$\mathbf{Y}_t = [Y_{1t}, Y_{2t}, ..., Y_{kt}]$',
                r'$\mathbf{Y}_t \in \mathbb{R}^k$',
                r'$Y_{it}$ where $i=1,...,N$ and $t=1,...,T$'
            ],
            'مثال - Example': [
                'سعر سهم واحد - One stock price',
                'سعر، حجم التداول، المؤشر - Price, volume, index',
                'أسعار محفظة أسهم - Portfolio of stocks',
                'بيانات عدة شركات عبر الزمن - Multiple companies over time'
            ]
        }

        df_types = pd.DataFrame(types_data)
        st.dataframe(df_types, use_container_width=True, height=250)

        # رسم توضيحي
        t = np.arange(0, 100)
        y1 = 100 + np.cumsum(np.random.normal(0, 2, 100))
        y2 = 50 + np.cumsum(np.random.normal(0, 1.5, 100))
        y3 = 75 + np.cumsum(np.random.normal(0, 1, 100))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y1, name='المتغير 1 - Var 1', line=dict(color='#2196F3')))
        fig.add_trace(go.Scatter(x=t, y=y2, name='المتغير 2 - Var 2', line=dict(color='#4CAF50')))
        fig.add_trace(go.Scatter(x=t, y=y3, name='المتغير 3 - Var 3', line=dict(color='#FF9800')))
        fig.update_layout(title='سلسلة متعددة المتغيرات - Multivariate Series',
                          height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("حسب السلوك الزمني - By Temporal Behavior")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### ذات اتجاه - Trending")
            t = np.arange(0, 100)
            y_trend = 10 + 0.5 * t + np.random.normal(0, 2, 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y_trend, mode='lines', line=dict(color='#2196F3')))
            fig.update_layout(height=250, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### موسمية - Seasonal")
            y_seasonal = 10 + 5 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.5, 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y_seasonal, mode='lines', line=dict(color='#4CAF50')))
            fig.update_layout(height=250, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("#### عشوائية - Random Walk")
            y_random = np.cumsum(np.random.normal(0, 1, 100))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y_random, mode='lines', line=dict(color='#FF9800')))
            fig.update_layout(height=250, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

# القسم 4: الخصائص
elif section == "الخصائص - Properties":
    st.header("⚙️ خصائص السلاسل الزمنية")
    st.markdown("### Properties of Time Series")

    st.markdown("---")

    # 1. المتوسط
    st.subheader("1️⃣ المتوسط - Mean")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### الصيغة الرياضية - Mathematical Formula")
        st.latex(r"\mu_t = E[Y_t] = \int_{-\infty}^{\infty} y \cdot f(y,t) \, dy")
        st.markdown("**للعينة - Sample Mean:**")
        st.latex(r"\bar{Y} = \frac{1}{T} \sum_{t=1}^{T} Y_t")

    with col2:
        st.info("""
        **الوصف - Description:**
        - القيمة المتوقعة للسلسلة عند الزمن t
        - Expected value of series at time t
        - يمثل مركز توزيع البيانات
        - Represents the center of data distribution
        """)

    # رسم توضيحي
    t = np.arange(0, 100)
    y_varying = 10 + 0.1 * t + np.random.normal(0, 2, 100)
    y_constant = 10 + np.random.normal(0, 2, 100)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('متوسط متغير - Varying Mean',
                                        'متوسط ثابت - Constant Mean'))

    fig.add_trace(go.Scatter(x=t, y=y_varying, mode='lines', line=dict(color='#2196F3'),
                             name='البيانات'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=10 + 0.1 * t, mode='lines', line=dict(color='#F44336', dash='dash'),
                             name='المتوسط'), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=y_constant, mode='lines', line=dict(color='#4CAF50'),
                             name='البيانات'), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=[10] * 100, mode='lines', line=dict(color='#F44336', dash='dash'),
                             name='المتوسط'), row=1, col=2)

    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2. التباين
    st.subheader("2️⃣ التباين - Variance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### الصيغة الرياضية - Mathematical Formula")
        st.latex(r"\sigma_t^2 = Var(Y_t) = E[(Y_t - \mu_t)^2]")
        st.markdown("**للعينة - Sample Variance:**")
        st.latex(r"s^2 = \frac{1}{T-1} \sum_{t=1}^{T} (Y_t - \bar{Y})^2")

    with col2:
        st.info("""
        **الوصف - Description:**
        - مقياس تشتت البيانات حول المتوسط
        - Measure of dispersion around mean
        - يقيس التقلب في السلسلة
        - Measures volatility in series
        """)

    # رسم توضيحي
    y_low_var = 10 + np.random.normal(0, 1, 100)
    y_high_var = 10 + np.random.normal(0, 5, 100)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('تباين منخفض - Low Variance (σ²=1)',
                                        'تباين مرتفع - High Variance (σ²=25)'))

    fig.add_trace(go.Scatter(x=t, y=y_low_var, mode='lines', line=dict(color='#4CAF50')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=y_high_var, mode='lines', line=dict(color='#F44336')), row=1, col=2)

    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. التباين المشترك
    st.subheader("3️⃣ التباين المشترك - Covariance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### الصيغة الرياضية - Mathematical Formula")
        st.latex(r"\gamma(t, s) = Cov(Y_t, Y_s) = E[(Y_t - \mu_t)(Y_s - \mu_s)]")
        st.markdown("**دالة الارتباط الذاتي - Autocovariance Function:**")
        st.latex(r"\gamma(k) = Cov(Y_t, Y_{t-k})")

    with col2:
        st.info("""
        **الوصف - Description:**
        - يقيس العلاقة الخطية بين قيمتين
        - Measures linear relationship between two values
        - k هو التأخر الزمني (lag)
        - k is the time lag
        """)

    st.markdown("---")

    # 4. معامل الارتباط
    st.subheader("4️⃣ معامل الارتباط الذاتي - Autocorrelation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### الصيغة الرياضية - Mathematical Formula")
        st.latex(r"\rho(k) = \frac{\gamma(k)}{\gamma(0)} = \frac{Cov(Y_t, Y_{t-k})}{\sqrt{Var(Y_t)Var(Y_{t-k})}}")
        st.latex(r"-1 \leq \rho(k) \leq 1")

    with col2:
        st.info("""
        **الوصف - Description:**
        - التباين المشترك المعياري
        - Standardized covariance
        - يأخذ قيم بين -1 و 1
        - Takes values between -1 and 1
        """)

    # رسم ACF
    np.random.seed(42)
    data = np.cumsum(np.random.normal(0, 1, 200))

    lags = range(0, 20)
    acf_values = [np.corrcoef(data[:-i if i > 0 else None], data[i:])[0, 1] if i > 0
                  else 1.0 for i in lags]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(lags), y=acf_values, marker_color='#2196F3'))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.add_hline(y=1.96 / np.sqrt(len(data)), line_dash="dash", line_color="red")
    fig.add_hline(y=-1.96 / np.sqrt(len(data)), line_dash="dash", line_color="red")
    fig.update_layout(title='دالة الارتباط الذاتي - Autocorrelation Function (ACF)',
                      xaxis_title='التأخر الزمني - Lag (k)',
                      yaxis_title='ρ(k)',
                      height=400,
                      template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# القسم 5: الاستقرارية
elif section == "الاستقرارية - Stationarity":
    st.header("📈 الاستقرارية")
    st.markdown("### Stationarity")

    st.markdown("""
    <div style='background-color: #fff3e0; padding: 20px; border-radius: 10px; border-left: 5px solid #FF9800;'>
    <h4 style='color: #E65100;'>⚠️ أهمية الاستقرارية - Importance of Stationarity</h4>
    <p>معظم النماذج الإحصائية تفترض أن السلسلة الزمنية مستقرة</p>
    <p>Most statistical models assume that the time series is stationary</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # أنواع الاستقرارية
    tab1, tab2, tab3 = st.tabs(["الاستقرارية القوية - Strict Stationarity",
                                "الاستقرارية الضعيفة - Weak Stationarity",
                                "عدم الاستقرارية - Non-Stationarity"])

    with tab1:
        st.subheader("الاستقرارية القوية (الصارمة)")
        st.markdown("### Strict (Strong) Stationarity")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### التعريف - Definition")
            st.markdown("""
            السلسلة الزمنية تكون مستقرة بشكل صارم إذا كان التوزيع الاحتمالي المشترك 
            لا يتغير بتغير الزمن
            """)
            st.markdown("""
            A time series is strictly stationary if the joint probability 
            distribution does not change when shifted in time
            """)

        with col2:
            st.markdown("#### الصيغة الرياضية - Mathematical Formula")
            st.latex(r"F(y_1, y_2, ..., y_k) = F(y_{1+h}, y_{2+h}, ..., y_{k+h})")
            st.latex(r"\forall h, k \in \mathbb{Z}")

        st.info("""
        **الخصائص - Properties:**
        - جميع العزوم تكون ثابتة - All moments are constant
        - التوزيع الكامل لا يتغير - Complete distribution unchanged
        - شرط قوي جداً - Very strong condition
        """)

    with tab2:
        st.subheader("الاستقرارية الضعيفة (من الدرجة الثانية)")
        st.markdown("### Weak (Second-Order) Stationarity")

        st.markdown("#### الشروط - Conditions")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px;'>
            <h5>1. المتوسط ثابت</h5>
            <h5>Constant Mean</h5>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"E[Y_t] = \mu \quad \forall t")

        with col2:
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px;'>
            <h5>2. التباين ثابت</h5>
            <h5>Constant Variance</h5>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"Var(Y_t) = \sigma^2 \quad \forall t")

        with col3:
            st.markdown("""
            <div style='background-color: #fff3e0; padding: 15px; border-radius: 10px;'>
            <h5>3. التباين المشترك يعتمد على k فقط</h5>
            <h5>Covariance depends only on k</h5>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"Cov(Y_t, Y_{t-k}) = \gamma(k)")

        st.markdown("---")

        # رسم سلسلة مستقرة
        np.random.seed(42)
        t = np.arange(0, 200)
        stationary = np.random.normal(10, 2, 200)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=stationary, mode='lines', line=dict(color='#4CAF50')))
        fig.add_hline(y=10, line_dash="dash", line_color="red",
                      annotation_text="μ = 10")
        fig.add_hrect(y0=10 - 2 * 2, y1=10 + 2 * 2, fillcolor="red", opacity=0.1,
                      annotation_text="±2σ", annotation_position="right")
        fig.update_layout(title='سلسلة مستقرة ضعيفاً - Weakly Stationary Series',
                          xaxis_title='الزمن - Time',
                          yaxis_title='القيمة - Value',
                          height=400,
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("عدم الاستقرارية")
        st.markdown("### Non-Stationarity")

        st.markdown("#### أنواع عدم الاستقرارية - Types of Non-Stationarity")

        # 1. اتجاه في المتوسط
        st.markdown("**1. اتجاه في المتوسط - Trend in Mean**")
        t = np.arange(0, 100)
        y_trend = 10 + 0.2 * t + np.random.normal(0, 2, 100)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=t, y=y_trend, mode='lines', line=dict(color='#F44336')))
        fig1.add_trace(go.Scatter(x=t, y=10 + 0.2 * t, mode='lines',
                                  line=dict(color='blue', dash='dash'),
                                  name='Trend'))
        fig1.update_layout(height=300, showlegend=False, template='plotly_white',
                           title='المتوسط غير ثابت - Non-constant Mean')
        st.plotly_chart(fig1, use_container_width=True)

        # 2. تباين متغير
        st.markdown("**2. تباين متغير - Changing Variance (Heteroscedasticity)**")
        variance = 1 + 0.05 * t
        y_hetero = 10 + np.random.normal(0, np.sqrt(variance), 100)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t, y=y_hetero, mode='lines', line=dict(color='#9C27B0')))
        fig2.update_layout(height=300, template='plotly_white',
                           title='التباين غير ثابت - Non-constant Variance')
        st.plotly_chart(fig2, use_container_width=True)

        # 3. موسمية
        st.markdown("**3. الموسمية - Seasonality**")
        y_seasonal = 10 + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 100)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t, y=y_seasonal, mode='lines', line=dict(color='#FF9800')))
        fig3.update_layout(height=300, template='plotly_white',
                           title='نمط موسمي - Seasonal Pattern')
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        st.markdown("#### طرق التحويل للاستقرارية - Methods to Achieve Stationarity")

        methods_data = {
            'الطريقة - Method': [
                'الفروق - Differencing',
                'التحويل اللوغاريتمي - Log Transformation',
                'إزالة الاتجاه - Detrending',
                'الفروق الموسمية - Seasonal Differencing'
            ],
            'الصيغة - Formula': [
                r'$\Delta Y_t = Y_t - Y_{t-1}$',
                r'$\log(Y_t)$',
                r'$Y_t - T_t$',
                r'$\Delta_s Y_t = Y_t - Y_{t-s}$'
            ],
            'الاستخدام - Use Case': [
                'إزالة الاتجاه - Remove trend',
                'استقرار التباين - Stabilize variance',
                'إزالة الاتجاه الخطي - Remove linear trend',
                'إزالة الموسمية - Remove seasonality'
            ]
        }

        df_methods = pd.DataFrame(methods_data)
        st.dataframe(df_methods, use_container_width=True)

# القسم 6: السيرورات
elif section == "السيرورات - Processes":
    st.header("🔄 السيرورات العشوائية")
    st.markdown("### Stochastic Processes")

    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px;'>
    <h4>التعريف - Definition</h4>
    <p>السيرورة العشوائية هي مجموعة من المتغيرات العشوائية المفهرسة بالزمن</p>
    <p>A stochastic process is a collection of random variables indexed by time</p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\{Y_t : t \in T\}")

    st.markdown("---")

    # أنواع السيرورات
    tab1, tab2, tab3, tab4 = st.tabs([
        "الضوضاء البيضاء - White Noise",
        "المسير العشوائي - Random Walk",
        "السيرورات الذاتية الانحدار - AR",
        "السيرورات المتوسطة المتحركة - MA"
    ])

    with tab1:
        st.subheader("الضوضاء البيضاء")
        st.markdown("### White Noise Process")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### الخصائص - Properties")
            st.latex(r"E[\varepsilon_t] = 0")
            st.latex(r"Var(\varepsilon_t) = \sigma^2")
            st.latex(r"Cov(\varepsilon_t, \varepsilon_s) = 0 \quad \forall t \neq s")

            st.info("""
            **الصفات - Characteristics:**
            - متوسط صفر - Zero mean
            - تباين ثابت - Constant variance
            - لا ارتباط ذاتي - No autocorrelation
            - عشوائية بحتة - Pure randomness
            """)

        with col2:
            st.markdown("#### الترميز - Notation")
            st.latex(r"\varepsilon_t \sim WN(0, \sigma^2)")
            st.latex(r"\varepsilon_t \sim iid(0, \sigma^2)")

            if st.checkbox("عرض توزيع غاوسي - Show Gaussian"):
                st.latex(r"\varepsilon_t \sim N(0, \sigma^2)")

        # رسم الضوضاء البيضاء
        np.random.seed(42)
        t = np.arange(0, 200)
        white_noise = np.random.normal(0, 1, 200)

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('السلسلة الزمنية - Time Series',
                                            'دالة الارتباط الذاتي - ACF'))

        fig.add_trace(go.Scatter(x=t, y=white_noise, mode='lines',
                                 line=dict(color='#2196F3')), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # ACF
        lags = range(0, 21)
        acf = [1.0] + [0.0] * 20
        fig.add_trace(go.Bar(x=list(lags), y=acf, marker_color='#4CAF50'), row=2, col=1)
        fig.add_hline(y=1.96 / np.sqrt(200), line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-1.96 / np.sqrt(200), line_dash="dash", line_color="red", row=2, col=1)

        fig.update_layout(height=600, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("المسير العشوائي")
        st.markdown("### Random Walk")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### النموذج - Model")
            st.latex(r"Y_t = Y_{t-1} + \varepsilon_t")
            st.latex(r"Y_t = Y_0 + \sum_{i=1}^{t} \varepsilon_i")

            st.markdown("**مع انحراف - With Drift:**")
            st.latex(r"Y_t = \delta + Y_{t-1} + \varepsilon_t")

        with col2:
            st.markdown("#### الخصائص - Properties")
            st.latex(r"E[Y_t] = Y_0 + t\delta")
            st.latex(r"Var(Y_t) = t\sigma^2")

            st.warning("""
            **⚠️ غير مستقر - Non-stationary**
            - التباين يزداد مع الزمن
            - Variance increases with time
            """)

        # رسم المسير العشوائي
        drift = st.slider("الانحراف - Drift (δ)", -0.5, 0.5, 0.0, 0.1)

        np.random.seed(42)
        innovations = np.random.normal(0, 1, 200)
        rw = np.cumsum(innovations) + drift * np.arange(200)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=rw, mode='lines', line=dict(color='#F44336')))
        if drift != 0:
            fig.add_trace(go.Scatter(x=t, y=drift * np.arange(200), mode='lines',
                                     line=dict(color='blue', dash='dash'),
                                     name='Drift'))
        fig.update_layout(title=f'مسير عشوائي - Random Walk (δ={drift})',
                          height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("السيرورة الذاتية الانحدار")
        st.markdown("### Autoregressive Process - AR(p)")

        st.markdown("#### النموذج العام - General Model")
        st.latex(r"Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \varepsilon_t")
        st.latex(r"\varepsilon_t \sim WN(0, \sigma^2)")

        st.markdown("---")

        # AR(1)
        st.markdown("##### AR(1) - First Order")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**النموذج - Model:**")
            st.latex(r"Y_t = c + \phi Y_{t-1} + \varepsilon_t")

            st.markdown("**شرط الاستقرارية - Stationarity Condition:**")
            st.latex(r"|\phi| < 1")

        with col2:
            st.markdown("**الخصائص - Properties:**")
            st.latex(r"E[Y_t] = \frac{c}{1-\phi}")
            st.latex(r"Var(Y_t) = \frac{\sigma^2}{1-\phi^2}")
            st.latex(r"\rho(k) = \phi^k")

        # رسم AR(1)
        phi = st.slider("معامل AR - φ", -0.9, 0.9, 0.7, 0.1)

        np.random.seed(42)
        ar1 = [0]
        for i in range(1, 200):
            ar1.append(phi * ar1[-1] + np.random.normal(0, 1))
        ar1 = np.array(ar1)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'AR(1): φ={phi}', 'ACF'))

        fig.add_trace(go.Scatter(x=t, y=ar1, mode='lines',
                                 line=dict(color='#9C27B0')), row=1, col=1)

        # ACF نظري
        lags_ar = range(0, 21)
        acf_ar = [phi ** k for k in lags_ar]
        fig.add_trace(go.Bar(x=list(lags_ar), y=acf_ar,
                             marker_color='#FF9800'), row=1, col=2)

        fig.update_layout(height=400, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        if abs(phi) >= 1:
            st.error("⚠️ السيرورة غير مستقرة - Process is non-stationary!")

    with tab4:
        st.subheader("السيرورة المتوسطة المتحركة")
        st.markdown("### Moving Average Process - MA(q)")

        st.markdown("#### النموذج العام - General Model")
        st.latex(
            r"Y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + ... + \theta_q \varepsilon_{t-q}")
        st.latex(r"\varepsilon_t \sim WN(0, \sigma^2)")

        st.markdown("---")

        # MA(1)
        st.markdown("##### MA(1) - First Order")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**النموذج - Model:**")
            st.latex(r"Y_t = \mu + \varepsilon_t + \theta \varepsilon_{t-1}")

            st.info("**دائماً مستقر - Always stationary**")

        with col2:
            st.markdown("**الخصائص - Properties:**")
            st.latex(r"E[Y_t] = \mu")
            st.latex(r"Var(Y_t) = \sigma^2(1 + \theta^2)")
            st.latex(r"\rho(1) = \frac{\theta}{1+\theta^2}")
            st.latex(r"\rho(k) = 0 \quad \forall k > 1")

        # رسم MA(1)
        theta = st.slider("معامل MA - θ", -0.9, 0.9, 0.5, 0.1)

        np.random.seed(42)
        eps = np.random.normal(0, 1, 201)
        ma1 = np.array([eps[i] + theta * eps[i - 1] for i in range(1, 201)])

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'MA(1): θ={theta}', 'ACF'))

        fig.add_trace(go.Scatter(x=t, y=ma1, mode='lines',
                                 line=dict(color='#00BCD4')), row=1, col=1)

        # ACF نظري
        acf_ma = [1.0, theta / (1 + theta ** 2)] + [0.0] * 19
        fig.add_trace(go.Bar(x=list(range(21)), y=acf_ma,
                             marker_color='#4CAF50'), row=1, col=2)
        fig.add_hline(y=1.96 / np.sqrt(200), line_dash="dash",
                      line_color="red", row=1, col=2)
        fig.add_hline(y=-1.96 / np.sqrt(200), line_dash="dash",
                      line_color="red", row=1, col=2)

        fig.update_layout(height=400, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# القسم 7: اللاحطية
elif section == "اللاحطية - Determinism":
    st.header("🎲 اللاحطية والعشوائية")
    st.markdown("### Determinism and Stochasticity")

    st.markdown("""
    <div style='background-color: #f3e5f5; padding: 20px; border-radius: 10px;'>
    <h4>المفهوم الأساسي - Basic Concept</h4>
    <p>السلاسل الزمنية يمكن تصنيفها بناءً على درجة القابلية للتنبؤ</p>
    <p>Time series can be classified based on their predictability</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "حطية كاملة - Deterministic",
        "عشوائية كاملة - Stochastic",
        "مختلطة - Mixed"
    ])

    with tab1:
        st.subheader("السلاسل الحطية (القطعية)")
        st.markdown("### Deterministic Series")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px;'>
            <h5>التعريف - Definition</h5>
            <p>سلسلة زمنية يمكن التنبؤ بقيمها المستقبلية بدقة كاملة</p>
            <p>A time series whose future values can be predicted with complete accuracy</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**الخصائص - Characteristics:**")
            st.markdown("""
            - لا يوجد مكون عشوائي
            - No random component
            - قابلة للتنبؤ التام
            - Completely predictable
            - تتبع قانوناً رياضياً محدداً
            - Follows a specific mathematical law
            """)

        with col2:
            st.markdown("**أمثلة - Examples:**")

            example_type = st.selectbox(
                "اختر مثالاً - Select example:",
                ["خطي - Linear", "تربيعي - Quadratic",
                 "جيبي - Sinusoidal", "أسي - Exponential"]
            )

        t = np.arange(0, 100)

        if example_type == "خطي - Linear":
            y = 2 + 0.5 * t
            formula = r"Y_t = 2 + 0.5t"
        elif example_type == "تربيعي - Quadratic":
            y = 1 + 0.1 * t + 0.01 * t ** 2
            formula = r"Y_t = 1 + 0.1t + 0.01t^2"
        elif example_type == "جيبي - Sinusoidal":
            y = 10 + 5 * np.sin(2 * np.pi * t / 20)
            formula = r"Y_t = 10 + 5\sin(2\pi t/20)"
        else:  # أسي
            y = 2 * np.exp(0.02 * t)
            formula = r"Y_t = 2e^{0.02t}"

        st.latex(formula)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode='lines+markers',
                                 line=dict(color='#4CAF50', width=2),
                                 marker=dict(size=4)))
        fig.update_layout(title=f'سلسلة حطية - Deterministic: {example_type}',
                          xaxis_title='t',
                          yaxis_title='Y_t',
                          height=400,
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ قابل للتنبؤ بنسبة 100% - 100% Predictable")

    with tab2:
        st.subheader("السلاسل العشوائية (الاحتمالية)")
        st.markdown("### Stochastic Series")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style='background-color: #ffebee; padding: 15px; border-radius: 10px;'>
            <h5>التعريف - Definition</h5>
            <p>سلسلة زمنية تحتوي على مكون عشوائي لا يمكن التنبؤ به</p>
            <p>A time series containing a random component that cannot be predicted</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**الخصائص - Characteristics:**")
            st.markdown("""
            - تحتوي على مكون عشوائي
            - Contains random component
            - التنبؤ احتمالي فقط
            - Only probabilistic prediction
            - تتبع توزيعاً احتمالياً
            - Follows probability distribution
            """)

        with col2:
            st.markdown("**أنواع - Types:**")

            stoch_type = st.selectbox(
                "اختر النوع - Select type:",
                ["ضوضاء بيضاء - White Noise",
                 "مسير عشوائي - Random Walk",
                 "AR(1)",
                 "MA(1)"]
            )

        np.random.seed(42)
        t = np.arange(0, 200)

        if stoch_type == "ضوضاء بيضاء - White Noise":
            y = np.random.normal(0, 1, 200)
            formula = r"\varepsilon_t \sim N(0, 1)"
        elif stoch_type == "مسير عشوائي - Random Walk":
            y = np.cumsum(np.random.normal(0, 1, 200))
            formula = r"Y_t = Y_{t-1} + \varepsilon_t"
        elif stoch_type == "AR(1)":
            y = [0]
            for i in range(199):
                y.append(0.7 * y[-1] + np.random.normal(0, 1))
            y = np.array(y)
            formula = r"Y_t = 0.7Y_{t-1} + \varepsilon_t"
        else:  # MA(1)
            eps = np.random.normal(0, 1, 201)
            y = np.array([eps[i] + 0.5 * eps[i - 1] for i in range(1, 201)])
            formula = r"Y_t = \varepsilon_t + 0.5\varepsilon_{t-1}"

        st.latex(formula)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode='lines',
                                 line=dict(color='#F44336', width=1.5)))
        fig.update_layout(title=f'سلسلة عشوائية - Stochastic: {stoch_type}',
                          xaxis_title='t',
                          yaxis_title='Y_t',
                          height=400,
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        st.warning("⚠️ غير قابل للتنبؤ الدقيق - Cannot be precisely predicted")

    with tab3:
        st.subheader("السلاسل المختلطة")
        st.markdown("### Mixed (Deterministic + Stochastic)")

        st.markdown("""
        <div style='background-color: #e1f5fe; padding: 20px; border-radius: 10px;'>
        <h5>الصيغة العامة - General Form</h5>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r"Y_t = D_t + S_t")
        st.latex(r"D_t: \text{المكون الحطي - Deterministic Component}")
        st.latex(r"S_t: \text{المكون العشوائي - Stochastic Component}")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            trend_strength = st.slider("قوة الاتجاه - Trend", 0.0, 2.0, 0.5, 0.1)
        with col2:
            seasonal_strength = st.slider("قوة الموسمية - Seasonality", 0.0, 10.0, 3.0, 0.5)
        with col3:
            noise_strength = st.slider("قوة العشوائية - Noise", 0.0, 5.0, 1.0, 0.1)

        t = np.arange(0, 200)
        deterministic = 10 + trend_strength * t + seasonal_strength * np.sin(2 * np.pi * t / 20)
        stochastic = np.random.normal(0, noise_strength, 200)
        mixed = deterministic + stochastic

        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=('المكون الحطي - Deterministic',
                                            'المكون العشوائي - Stochastic',
                                            'السلسلة المختلطة - Mixed'))

        fig.add_trace(go.Scatter(x=t, y=deterministic, mode='lines',
                                 line=dict(color='#4CAF50')), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=stochastic, mode='lines',
                                 line=dict(color='#F44336')), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=mixed, mode='lines',
                                 line=dict(color='#2196F3')), row=3, col=1)

        fig.update_layout(height=800, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        st.latex(r"Y_t = (10 + " + f"{trend_strength:.1f}" + r"t + " +
                 f"{seasonal_strength:.1f}" + r"\sin(2\pi t/20)) + \varepsilon_t")

        # نسبة المساهمة
        var_det = np.var(deterministic)
        var_stoch = np.var(stochastic)
        total_var = var_det + var_stoch

        col1, col2 = st.columns(2)
        with col1:
            st.metric("نسبة المكون الحطي - Deterministic %",
                      f"{100 * var_det / total_var:.1f}%")
        with col2:
            st.metric("نسبة المكون العشوائي - Stochastic %",
                      f"{100 * var_stoch / total_var:.1f}%")

# القسم 8: التغيرات الهيكلية
else:  # التغيرات الهيكلية - Structural Changes
    st.header("🔧 التغيرات الهيكلية")
    st.markdown("### Structural Changes (Breaks)")

    st.markdown("""
    <div style='background-color: #fff9c4; padding: 20px; border-radius: 10px; border-left: 5px solid #FBC02D;'>
    <h4 style='color: #F57F17;'>⚡ التعريف - Definition</h4>
    <p>التغير الهيكلي هو تغير مفاجئ أو تدريجي في خصائص السلسلة الزمنية</p>
    <p>Structural change is a sudden or gradual change in the properties of a time series</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "أنواع التغيرات - Types",
        "تغير في المستوى - Level Shift",
        "تغير في الاتجاه - Trend Change",
        "تغير في التباين - Variance Change"
    ])

    with tab1:
        st.subheader("أنواع التغيرات الهيكلية")
        st.markdown("### Types of Structural Changes")

        types_breaks = {
            'النوع - Type': [
                'تغير في المستوى - Level Shift',
                'تغير في الاتجاه - Trend Change',
                'تغير في التباين - Variance Change',
                'تغير في الموسمية - Seasonal Change',
                'تغير في المعاملات - Parameter Change'
            ],
            'الوصف - Description': [
                'قفزة مفاجئة في المتوسط - Sudden jump in mean',
                'تغير في معدل النمو - Change in growth rate',
                'تغير في التقلب - Change in volatility',
                'تغير في النمط الموسمي - Change in seasonal pattern',
                'تغير في معاملات النموذج - Change in model parameters'
            ],
            'الرمز - Symbol': [
                r'$\mu_1 \rightarrow \mu_2$',
                r'$\beta_1 \rightarrow \beta_2$',
                r'$\sigma_1^2 \rightarrow \sigma_2^2$',
                r'$S_1 \rightarrow S_2$',
                r'$\theta_1 \rightarrow \theta_2$'
            ],
            'مثال - Example': [
                'تغير في السياسة الاقتصادية',
                'تسارع أو تباطؤ النمو',
                'أزمة مالية',
                'تغير في نمط الاستهلاك',
                'تغير في آلية السوق'
            ]
        }

        df_breaks = pd.DataFrame(types_breaks)
        st.dataframe(df_breaks, use_container_width=True, height=250)

        st.markdown("---")

        st.markdown("#### التصنيف حسب طبيعة التغير - Classification by Nature")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px;'>
            <h5>1. تغير مفاجئ - Abrupt Change</h5>
            <p>يحدث في نقطة زمنية محددة</p>
            <p>Occurs at a specific time point</p>
            </div>
            """, unsafe_allow_html=True)

            st.latex(
                r"Y_t = \begin{cases} \mu_1 + \varepsilon_t & t < \tau \\ \mu_2 + \varepsilon_t & t \geq \tau \end{cases}")

        with col2:
            st.markdown("""
            <div style='background-color: #fff3e0; padding: 15px; border-radius: 10px;'>
            <h5>2. تغير تدريجي - Gradual Change</h5>
            <p>يحدث على فترة زمنية</p>
            <p>Occurs over a period of time</p>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"Y_t = \mu_1 + (\mu_2 - \mu_1)F(t, \tau, \gamma) + \varepsilon_t")

    with tab2:
        st.subheader("تغير في المستوى")
        st.markdown("### Level Shift")

        st.markdown("#### النموذج الرياضي - Mathematical Model")

        col1, col2 = st.columns(2)

        with col1:
            st.latex(r"Y_t = \mu + \delta \cdot I(t \geq \tau) + \varepsilon_t")
            st.markdown("حيث (where):")
            st.latex(r"I(t \geq \tau) = \begin{cases} 0 & t < \tau \\ 1 & t \geq \tau \end{cases}")

        with col2:
            st.markdown("""
            **المعاملات - Parameters:**
            - μ: المستوى الأصلي - Original level
            - δ: حجم التغير - Size of shift
            - τ: نقطة التغير - Break point
            - ε: الخطأ العشوائي - Random error
            """)

        st.markdown("---")

        # تطبيق تفاعلي
        st.markdown("#### تطبيق تفاعلي - Interactive Application")

        col1, col2, col3 = st.columns(3)

        with col1:
            mu = st.slider("المستوى الأصلي - μ", 0.0, 20.0, 10.0, 1.0)
        with col2:
            delta = st.slider("حجم التغير - δ", -10.0, 10.0, 5.0, 0.5)
        with col3:
            tau = st.slider("نقطة التغير - τ", 20, 180, 100, 10)

        t = np.arange(0, 200)
        np.random.seed(42)
        level_shift = mu + delta * (t >= tau) + np.random.normal(0, 1, 200)

        fig = go.Figure()

        # قبل التغير
        fig.add_trace(go.Scatter(x=t[t < tau], y=level_shift[t < tau],
                                 mode='lines', line=dict(color='#2196F3', width=2),
                                 name=f'قبل: μ={mu}'))

        # بعد التغير
        fig.add_trace(go.Scatter(x=t[t >= tau], y=level_shift[t >= tau],
                                 mode='lines', line=dict(color='#F44336', width=2),
                                 name=f'بعد: μ={mu + delta}'))

        # خط التغير
        fig.add_vline(x=tau, line_dash="dash", line_color="green",
                      annotation_text=f"τ = {tau}")

        # خطوط المتوسط
        fig.add_hline(y=mu, line_dash="dot", line_color="#2196F3",
                      annotation_text=f"μ₁ = {mu}")
        fig.add_hline(y=mu + delta, line_dash="dot", line_color="#F44336",
                      annotation_text=f"μ₂ = {mu + delta}")

        fig.update_layout(title='تغير في المستوى - Level Shift',
                          xaxis_title='t',
                          yaxis_title='Y_t',
                          height=500,
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        st.latex(f"Y_t = {mu} + {delta} \cdot I(t \geq {tau}) + \varepsilon_t")

    with tab3:
        st.subheader("تغير في الاتجاه")
        st.markdown("### Trend Change")

        st.markdown("#### النموذج الرياضي - Mathematical Model")

        col1, col2 = st.columns(2)

        with col1:
            st.latex(r"Y_t = \alpha + \beta_1 t + (\beta_2 - \beta_1)(t - \tau) \cdot I(t \geq \tau) + \varepsilon_t")

        with col2:
            st.markdown("""
            **المعاملات - Parameters:**
            - α: المقطع - Intercept
            - β₁: الميل الأصلي - Original slope
            - β₂: الميل الجديد - New slope
            - τ: نقطة التغير - Break point
            """)

        st.markdown("---")

        # تطبيق تفاعلي
        st.markdown("#### تطبيق تفاعلي - Interactive Application")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            alpha = st.slider("المقطع - α", 0.0, 20.0, 10.0, 1.0)
        with col2:
            beta1 = st.slider("الميل الأول - β₁", -0.5, 0.5, 0.1, 0.05)
        with col3:
            beta2 = st.slider("الميل الثاني - β₂", -0.5, 0.5, -0.05, 0.05)
        with col4:
            tau_trend = st.slider("نقطة التغير - τ", 20, 180, 100, 10, key='tau_trend')

        t = np.arange(0, 200)
        np.random.seed(42)
        trend_change = (alpha + beta1 * t +
                        (beta2 - beta1) * (t - tau_trend) * (t >= tau_trend) +
                        np.random.normal(0, 2, 200))

        # خطوط الاتجاه
        trend1 = alpha + beta1 * t
        trend2_part1 = alpha + beta1 * t
        trend2_part2 = alpha + beta1 * tau_trend + beta2 * (t - tau_trend)

        fig = go.Figure()

        # البيانات
        fig.add_trace(go.Scatter(x=t, y=trend_change, mode='lines',
                                 line=dict(color='#9E9E9E', width=1),
                                 name='البيانات'))

        # الاتجاه الأول
        fig.add_trace(go.Scatter(x=t[t < tau_trend], y=trend1[t < tau_trend],
                                 mode='lines', line=dict(color='#2196F3', width=3, dash='dash'),
                                 name=f'اتجاه 1: β={beta1}'))

        # الاتجاه الثاني
        fig.add_trace(go.Scatter(x=t[t >= tau_trend], y=trend2_part2[t >= tau_trend],
                                 mode='lines', line=dict(color='#F44336', width=3, dash='dash'),
                                 name=f'اتجاه 2: β={beta2}'))

        # خط التغير
        fig.add_vline(x=tau_trend, line_dash="dash", line_color="green",
                      annotation_text=f"τ = {tau_trend}")

        fig.update_layout(title='تغير في الاتجاه - Trend Change',
                          xaxis_title='t',
                          yaxis_title='Y_t',
                          height=500,
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        # عرض التفسير
        if beta1 > 0 and beta2 < 0:
            st.info("📊 من نمو إيجابي إلى تراجع - From positive growth to decline")
        elif beta1 < 0 and beta2 > 0:
            st.info("📊 من تراجع إلى نمو إيجابي - From decline to positive growth")
        elif abs(beta2) > abs(beta1):
            st.info("📊 تسارع في الاتجاه - Acceleration of trend")
        else:
            st.info("📊 تباطؤ في الاتجاه - Deceleration of trend")

    with tab4:
        st.subheader("تغير في التباين")
        st.markdown("### Variance Change (Heteroscedasticity)")

        st.markdown("#### النموذج الرياضي - Mathematical Model")

        col1, col2 = st.columns(2)

        with col1:
            st.latex(r"Y_t = \mu + \varepsilon_t")
            st.latex(
                r"\varepsilon_t \sim \begin{cases} N(0, \sigma_1^2) & t < \tau \\ N(0, \sigma_2^2) & t \geq \tau \end{cases}")

        with col2:
            st.markdown("""
            **المعاملات - Parameters:**
            - σ₁²: التباين الأصلي - Original variance
            - σ₂²: التباين الجديد - New variance
            - τ: نقطة التغير - Break point
            """)

        st.markdown("---")

        # تطبيق تفاعلي
        st.markdown("#### تطبيق تفاعلي - Interactive Application")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            mu_var = st.slider("المتوسط - μ", 0.0, 20.0, 10.0, 1.0, key='mu_var')
        with col2:
            sigma1 = st.slider("الانحراف المعياري 1 - σ₁", 0.5, 5.0, 1.0, 0.5)
        with col3:
            sigma2 = st.slider("الانحراف المعياري 2 - σ₂", 0.5, 5.0, 3.0, 0.5)
        with col4:
            tau_var = st.slider("نقطة التغير - τ", 20, 180, 100, 10, key='tau_var')

        t = np.arange(0, 200)
        np.random.seed(42)

        variance_change = np.zeros(200)
        variance_change[t < tau_var] = mu_var + np.random.normal(0, sigma1, sum(t < tau_var))
        variance_change[t >= tau_var] = mu_var + np.random.normal(0, sigma2, sum(t >= tau_var))

        fig = go.Figure()

        # البيانات قبل التغير
        fig.add_trace(go.Scatter(x=t[t < tau_var], y=variance_change[t < tau_var],
                                 mode='lines', line=dict(color='#2196F3', width=1.5),
                                 name=f'σ₁ = {sigma1}'))

        # البيانات بعد التغير
        fig.add_trace(go.Scatter(x=t[t >= tau_var], y=variance_change[t >= tau_var],
                                 mode='lines', line=dict(color='#F44336', width=1.5),
                                 name=f'σ₂ = {sigma2}'))

        # المتوسط
        fig.add_hline(y=mu_var, line_dash="dash", line_color="black",
                      annotation_text=f"μ = {mu_var}")

        # نطاقات الثقة
        fig.add_hrect(y0=mu_var - 2 * sigma1, y1=mu_var + 2 * sigma1,
                      fillcolor="blue", opacity=0.1,
                      annotation_text="±2σ₁", annotation_position="left")
        fig.add_hrect(y0=mu_var - 2 * sigma2, y1=mu_var + 2 * sigma2,
                      fillcolor="red", opacity=0.1,
                      annotation_text="±2σ₂", annotation_position="right")

        # خط التغير
        fig.add_vline(x=tau_var, line_dash="dash", line_color="green",
                      annotation_text=f"τ = {tau_var}")

        fig.update_layout(title='تغير في التباين - Variance Change',
                          xaxis_title='t',
                          yaxis_title='Y_t',
                          height=500,
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        # مقارنة
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("التباين الأول - σ₁²", f"{sigma1 ** 2:.2f}")
        with col2:
            st.metric("التباين الثاني - σ₂²", f"{sigma2 ** 2:.2f}")
        with col3:
            change_pct = ((sigma2 ** 2 - sigma1 ** 2) / sigma1 ** 2) * 100
            st.metric("التغير % - Change %", f"{change_pct:+.1f}%")

        if sigma2 > sigma1:
            st.warning("⚠️ زيادة في التقلب (التباين) - Increase in volatility")
        else:
            st.success("✅ انخفاض في التقلب (التباين) - Decrease in volatility")

# تذييل الصفحة
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p>📚 تطبيق تعليمي تفاعلي للسلاسل الزمنية</p>
<p>Interactive Educational Application for Time Series</p>
<p style='font-size: 12px; margin-top: 10px;'>
جميع الحقوق محفوظة © 2025
</p>
</div>
""", unsafe_allow_html=True)