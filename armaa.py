import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="ARMA & ARIMA Models Tutorial", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0d47a1;
        border-left: 5px solid #1f77b4;
        padding-left: 1rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .formula-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Select Topic:", [
    "🏠 Introduction",
    "📊 Stationarity Concepts",
    "🔄 Integration & Differencing",
    "📈 ARMA Models",
    "🎯 ARIMA Models",
    "🧪 Interactive Simulation",
    "📝 Summary & Comparison"
])

# ============================================================================
# PAGE 1: INTRODUCTION
# ============================================================================
if page == "🏠 Introduction":
    st.markdown('<div class="main-header">Time Series Analysis: ARMA & ARIMA Models</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Welcome to Time Series Modeling! 👋

    This interactive tutorial will guide you through the fundamental concepts of time series analysis, 
    focusing on ARMA and ARIMA models. No prior knowledge required!
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 What You Will Learn:

        1. **Stationarity**: Understanding why it matters
        2. **Trends**: Deterministic vs Stochastic
        3. **Integration**: Order and meaning
        4. **ARMA Models**: Autoregressive and Moving Average
        5. **ARIMA Models**: Combining everything together
        6. **Practical Applications**: How to use these models
        """)

    with col2:
        st.markdown("""
        ### 🛠️ What This App Offers:

        - **Interactive Visualizations**: See concepts in action
        - **Mathematical Formulations**: Understand the theory
        - **Simulations**: Generate and analyze time series
        - **Step-by-Step Explanations**: From basics to advanced
        - **Real-Time Testing**: Experiment with parameters
        """)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📖 How to Use This Tutorial:

    Navigate through the topics using the **sidebar menu** on the left. Each section builds upon previous concepts:

    - Start with **Stationarity Concepts** to understand the foundation
    - Move to **Integration & Differencing** to learn transformations
    - Explore **ARMA Models** to understand the basic building blocks
    - Study **ARIMA Models** to see how everything combines
    - Use **Interactive Simulation** to experiment
    - Review **Summary & Comparison** to consolidate your learning
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Ready to begin? Select a topic from the sidebar!")

# ============================================================================
# PAGE 2: STATIONARITY CONCEPTS
# ============================================================================
elif page == "📊 Stationarity Concepts":
    st.markdown('<div class="main-header">Understanding Stationarity</div>', unsafe_allow_html=True)

    st.markdown("""
    ## What is Stationarity? 🤔

    Stationarity is a fundamental concept in time series analysis. A time series is **stationary** if its 
    statistical properties (mean, variance, autocorrelation) do not change over time.
    """)

    # Mathematical Definition
    st.markdown('<div class="section-header">Mathematical Definition</div>', unsafe_allow_html=True)

    st.markdown("A time series $\{Y_t\}$ is **strictly stationary** if:")
    st.latex(r"P(Y_{t_1}, Y_{t_2}, ..., Y_{t_n}) = P(Y_{t_1+h}, Y_{t_2+h}, ..., Y_{t_n+h})")
    st.markdown("for all $t_1, t_2, ..., t_n$ and any shift $h$.")

    st.markdown("A time series is **weakly (covariance) stationary** if:")
    st.latex(
        r"\begin{align*} E[Y_t] &= \mu \quad \text{(constant mean)} \\ Var(Y_t) &= \sigma^2 \quad \text{(constant variance)} \\ Cov(Y_t, Y_{t-k}) &= \gamma_k \quad \text{(depends only on lag } k\text{)} \end{align*}")

    # Why Stationarity Matters
    st.markdown('<div class="section-header">Why Stationarity Matters</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Why do we need stationarity?**

    1. **Predictability**: Stationary series have consistent patterns that can be modeled
    2. **Mathematical Tractability**: Many theorems and tools assume stationarity
    3. **Model Validity**: ARMA models require stationary data
    4. **Reliable Inference**: Statistical properties remain constant, allowing valid conclusions
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Types of Non-Stationarity: Trends
    st.markdown('<div class="section-header">Types of Non-Stationarity: Trends</div>', unsafe_allow_html=True)

    st.markdown("### 1. Deterministic Trend")

    st.markdown("A **deterministic trend** is a predictable, systematic pattern over time:")
    st.latex(r"Y_t = \alpha + \beta t + \varepsilon_t")

    st.markdown("""
    where:
    - $\\alpha$ = intercept
    - $\\beta$ = trend coefficient (slope)
    - $t$ = time
    - $\\varepsilon_t$ = random error (stationary)
    """)

    st.markdown("**Key Characteristics:**")
    st.markdown("""
    - Trend is a **function of time**
    - Can be **removed by de-trending** (subtracting the trend)
    - After de-trending, series becomes stationary
    - Deterministic and predictable
    """)

    # Simulation: Deterministic Trend
    st.markdown("#### Simulation: Deterministic Trend")

    col1, col2 = st.columns([1, 2])
    with col1:
        det_alpha = st.slider("Intercept (α)", 0.0, 10.0, 5.0, 0.5, key="det_alpha")
        det_beta = st.slider("Trend (β)", -0.5, 0.5, 0.1, 0.05, key="det_beta")
        det_noise = st.slider("Noise level (σ)", 0.1, 5.0, 1.0, 0.1, key="det_noise")

    np.random.seed(42)
    n = 200
    t = np.arange(n)
    deterministic_trend = det_alpha + det_beta * t + np.random.normal(0, det_noise, n)
    detrended = deterministic_trend - (det_alpha + det_beta * t)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Original Series (with Deterministic Trend)",
                                                        "After De-trending (Stationary)"))

    fig.add_trace(go.Scatter(y=deterministic_trend, mode='lines', name='Original',
                             line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=det_alpha + det_beta * t, mode='lines', name='Trend Line',
                             line=dict(color='#ff7f0e', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(y=detrended, mode='lines', name='Detrended',
                             line=dict(color='#2ca02c', width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(height=600, showlegend=True, title_text="Deterministic Trend Example")
    st.plotly_chart(fig, use_container_width=True)

    # Stochastic Trend
    st.markdown("---")
    st.markdown("### 2. Stochastic Trend (Random Walk)")

    st.markdown(
        "A **stochastic trend** is a random, unpredictable trend. The most common example is a **random walk**:")
    st.latex(r"Y_t = Y_{t-1} + \varepsilon_t")

    st.markdown("where $\\varepsilon_t \\sim N(0, \\sigma^2)$ is white noise.")

    st.markdown("This can also be written as:")
    st.latex(r"Y_t = Y_0 + \sum_{i=1}^{t} \varepsilon_i")

    st.markdown("**Key Characteristics:**")
    st.markdown("""
    - Trend is **random and unpredictable**
    - Variance **increases over time**: $Var(Y_t) = t\\sigma^2$
    - **Cannot be removed by de-trending** (no deterministic pattern)
    - Must be removed by **differencing**
    - Also called **unit root process** or **integrated of order 1**
    """)

    # Random Walk with Drift
    st.markdown("#### Random Walk with Drift")
    st.latex(r"Y_t = \delta + Y_{t-1} + \varepsilon_t")
    st.markdown("where $\\delta$ is the drift parameter (constant upward or downward tendency).")

    # Simulation: Stochastic Trend
    st.markdown("#### Simulation: Stochastic Trend (Random Walk)")

    col1, col2 = st.columns([1, 2])
    with col1:
        rw_drift = st.slider("Drift (δ)", -0.5, 0.5, 0.1, 0.05, key="rw_drift")
        rw_noise = st.slider("Innovation variance (σ²)", 0.1, 5.0, 1.0, 0.1, key="rw_noise")
        rw_seed = st.slider("Random seed", 1, 100, 42, 1, key="rw_seed")

    np.random.seed(rw_seed)
    n = 200
    random_walk = np.zeros(n)
    random_walk[0] = 0
    for i in range(1, n):
        random_walk[i] = rw_drift + random_walk[i - 1] + np.random.normal(0, rw_noise)

    # First difference
    diff_rw = np.diff(random_walk)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Random Walk (Non-Stationary)",
                                                        "First Difference (Stationary)"))

    fig.add_trace(go.Scatter(y=random_walk, mode='lines', name='Random Walk',
                             line=dict(color='#d62728', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=diff_rw, mode='lines', name='Differenced',
                             line=dict(color='#9467bd', width=2)), row=2, col=1)
    fig.add_hline(y=rw_drift, line_dash="dash", line_color="gray", row=2, col=1,
                  annotation_text=f"Mean ≈ {rw_drift:.2f}")

    fig.update_layout(height=600, showlegend=True, title_text="Stochastic Trend Example")
    st.plotly_chart(fig, use_container_width=True)

    # Comparison Table
    st.markdown('<div class="section-header">Deterministic vs Stochastic Trends: Comparison</div>',
                unsafe_allow_html=True)

    comparison_df = pd.DataFrame({
        'Characteristic': ['Nature', 'Predictability', 'Variance over time', 'Removal method',
                           'Mathematical form', 'Effect of shocks', 'Model type'],
        'Deterministic Trend': ['Function of time', 'Predictable', 'Constant', 'De-trending',
                                'α + βt + εₜ', 'Temporary', 'Trend-stationary (TS)'],
        'Stochastic Trend': ['Random accumulation', 'Unpredictable', 'Increasing (tσ²)', 'Differencing',
                             'Yₜ₋₁ + εₜ', 'Permanent', 'Difference-stationary (DS)']
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Testing for Stationarity
    st.markdown('<div class="section-header">Testing for Stationarity: ADF Test</div>', unsafe_allow_html=True)

    st.markdown("""
    The **Augmented Dickey-Fuller (ADF) test** is used to test for a unit root (stochastic trend).

    **Hypotheses:**
    """)
    st.latex(
        r"\begin{align*} H_0 &: \text{Series has a unit root (non-stationary)} \\ H_1 &: \text{Series is stationary} \end{align*}")

    st.markdown("""
    **Interpretation:**
    - If **p-value < 0.05**: Reject H₀ → Series is stationary
    - If **p-value ≥ 0.05**: Fail to reject H₀ → Series is non-stationary
    """)

    # ADF Test on simulations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ADF Test: Deterministic Trend**")
        adf_det = adfuller(deterministic_trend)
        st.metric("ADF Statistic", f"{adf_det[0]:.4f}")
        st.metric("p-value", f"{adf_det[1]:.4f}")
        if adf_det[1] < 0.05:
            st.success("✅ Series is stationary (after accounting for trend)")
        else:
            st.warning("⚠️ Series is non-stationary")

    with col2:
        st.markdown("**ADF Test: Random Walk**")
        adf_rw = adfuller(random_walk)
        st.metric("ADF Statistic", f"{adf_rw[0]:.4f}")
        st.metric("p-value", f"{adf_rw[1]:.4f}")
        if adf_rw[1] < 0.05:
            st.success("✅ Series is stationary")
        else:
            st.warning("⚠️ Series is non-stationary (has unit root)")

# ============================================================================
# PAGE 3: INTEGRATION & DIFFERENCING
# ============================================================================
elif page == "🔄 Integration & Differencing":
    st.markdown('<div class="main-header">Integration and Differencing</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Understanding Integration 📐

    Integration refers to how many times a series must be **differenced** to become stationary.
    This concept is crucial for ARIMA models.
    """)

    # Order of Integration
    st.markdown('<div class="section-header">Order of Integration</div>', unsafe_allow_html=True)

    st.markdown("A time series $Y_t$ is **integrated of order d**, denoted $I(d)$, if:")
    st.latex(r"\nabla^d Y_t \text{ is stationary}")

    st.markdown("where $\\nabla$ is the **difference operator**:")
    st.latex(r"\nabla Y_t = Y_t - Y_{t-1} = (1-L)Y_t")

    st.markdown("and $L$ is the **lag operator**: $LY_t = Y_{t-1}$")

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Common Integration Orders:**

    - **I(0)**: Series is already stationary (no differencing needed)
    - **I(1)**: Series becomes stationary after first differencing (e.g., random walk)
    - **I(2)**: Series needs second differencing to be stationary (rare in practice)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # First Difference
    st.markdown('<div class="section-header">First Differencing</div>', unsafe_allow_html=True)

    st.markdown("**First difference** removes linear trends and converts I(1) to I(0):")
    st.latex(r"\nabla Y_t = Y_t - Y_{t-1}")

    st.markdown("**Example: Random Walk**")
    st.latex(
        r"\begin{align*} Y_t &= Y_{t-1} + \varepsilon_t \quad \text{(I(1) - non-stationary)} \\ \nabla Y_t &= Y_t - Y_{t-1} = \varepsilon_t \quad \text{(I(0) - stationary)} \end{align*}")

    # Second Difference
    st.markdown('<div class="section-header">Second Differencing</div>', unsafe_allow_html=True)

    st.markdown("**Second difference** is the difference of the first difference:")
    st.latex(r"\nabla^2 Y_t = \nabla(\nabla Y_t) = (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2})")
    st.latex(r"\nabla^2 Y_t = Y_t - 2Y_{t-1} + Y_{t-2}")

    st.markdown("This removes **quadratic trends** and converts I(2) to I(0).")

    # Interactive Simulation
    st.markdown('<div class="section-header">Interactive Differencing Simulation</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Generate Time Series**")
        series_type = st.selectbox("Select series type:",
                                   ["I(0) - White Noise",
                                    "I(1) - Random Walk",
                                    "I(1) - Random Walk with Drift",
                                    "I(2) - Double Integration"])
        n_obs = st.slider("Number of observations:", 100, 500, 200, 50)
        sigma = st.slider("Innovation standard deviation:", 0.5, 3.0, 1.0, 0.1)
        sim_seed = st.slider("Random seed:", 1, 100, 42, 1, key="int_seed")

    np.random.seed(sim_seed)
    innovations = np.random.normal(0, sigma, n_obs)

    if series_type == "I(0) - White Noise":
        series = innovations
        true_order = 0
    elif series_type == "I(1) - Random Walk":
        series = np.cumsum(innovations)
        true_order = 1
    elif series_type == "I(1) - Random Walk with Drift":
        series = np.cumsum(innovations + 0.1)
        true_order = 1
    else:  # I(2)
        series = np.cumsum(np.cumsum(innovations))
        true_order = 2

    # Calculate differences
    diff1 = np.diff(series)
    diff2 = np.diff(diff1)

    # Create subplots
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=(f"Original Series: {series_type}",
                                        "First Difference",
                                        "Second Difference"),
                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=series, mode='lines', name='Original',
                             line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=diff1, mode='lines', name='1st Diff',
                             line=dict(color='#ff7f0e', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(y=diff2, mode='lines', name='2nd Diff',
                             line=dict(color='#2ca02c', width=2)), row=3, col=1)

    fig.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Statistical Tests
    st.markdown("### Statistical Tests for Integration Order")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Series**")
        adf_orig = adfuller(series, autolag='AIC')
        st.metric("ADF Statistic", f"{adf_orig[0]:.4f}")
        st.metric("p-value", f"{adf_orig[1]:.4f}")
        if adf_orig[1] < 0.05:
            st.success("✅ Stationary")
        else:
            st.error("❌ Non-stationary")

    with col2:
        st.markdown("**First Difference**")
        adf_diff1 = adfuller(diff1, autolag='AIC')
        st.metric("ADF Statistic", f"{adf_diff1[0]:.4f}")
        st.metric("p-value", f"{adf_diff1[1]:.4f}")
        if adf_diff1[1] < 0.05:
            st.success("✅ Stationary")
        else:
            st.error("❌ Non-stationary")

    with col3:
        st.markdown("**Second Difference**")
        adf_diff2 = adfuller(diff2, autolag='AIC')
        st.metric("ADF Statistic", f"{adf_diff2[0]:.4f}")
        st.metric("p-value", f"{adf_diff2[1]:.4f}")
        if adf_diff2[1] < 0.05:
            st.success("✅ Stationary")
        else:
            st.error("❌ Non-stationary")

    # Determine integration order
    st.markdown("---")
    if adf_orig[1] < 0.05:
        estimated_order = 0
    elif adf_diff1[1] < 0.05:
        estimated_order = 1
    else:
        estimated_order = 2

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **Integration Order Determination:**

    - **True integration order**: I({true_order})
    - **Estimated integration order** (via ADF test): I({estimated_order})

    The series requires **{estimated_order} differencing step(s)** to achieve stationarity.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Over-differencing Warning
    st.markdown('<div class="section-header">Warning: Over-differencing</div>', unsafe_allow_html=True)

    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    **Be careful not to over-difference!**

    - Differencing more than necessary can introduce **artificial patterns**
    - It can create **negative autocorrelation** in the series
    - This leads to **poor forecasting performance**
    - Always check stationarity after each differencing step

    **Rule of thumb**: Most economic/financial time series are I(0) or I(1). I(2) series are rare.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Seasonal Differencing
    st.markdown('<div class="section-header">Seasonal Differencing</div>', unsafe_allow_html=True)

    st.markdown("For seasonal data, we use **seasonal differencing**:")
    st.latex(r"\nabla_s Y_t = Y_t - Y_{t-s}")

    st.markdown("where $s$ is the seasonal period (e.g., 12 for monthly data with yearly seasonality).")

    st.markdown("Combined with regular differencing:")
    st.latex(r"\nabla \nabla_s Y_t = (Y_t - Y_{t-s}) - (Y_{t-1} - Y_{t-s-1})")

# ============================================================================
# PAGE 4: ARMA MODELS
# ============================================================================
elif page == "📈 ARMA Models":
    st.markdown('<div class="main-header">ARMA Models: Autoregressive Moving Average</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Introduction to ARMA Models 🎯

    ARMA models combine two components:
    - **AR (Autoregressive)**: Uses past values of the series
    - **MA (Moving Average)**: Uses past forecast errors

    ARMA models work **only on stationary series** (I(0)).
    """)

    # AR Model
    st.markdown('<div class="section-header">1. Autoregressive (AR) Model</div>', unsafe_allow_html=True)

    st.markdown("### AR(p) Model Definition")

    st.markdown("An **AR(p)** model uses $p$ lagged values of the series to predict the current value:")
    st.latex(r"Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \varepsilon_t")

    st.markdown("where:")
    st.markdown("""
    - $Y_t$ = current value
    - $c$ = constant term
    - $\\phi_1, \\phi_2, ..., \\phi_p$ = autoregressive coefficients
    - $p$ = order of the AR model (number of lags)
    - $\\varepsilon_t \\sim N(0, \\sigma^2)$ = white noise error term
    """)

    st.markdown("### AR(1) Model - Detailed Example")

    st.latex(r"Y_t = c + \phi_1 Y_{t-1} + \varepsilon_t")

    st.markdown("**Characteristics of AR(1):**")
    st.markdown("""
    - **Stationarity condition**: $|\\phi_1| < 1$
    - **Mean**: $E[Y_t] = \\frac{c}{1-\\phi_1}$
    - **Variance**: $Var(Y_t) = \\frac{\\sigma^2}{1-\\phi_1^2}$
    - **Autocorrelation function (ACF)**: $\\rho_k = \\phi_1^k$ (exponential decay)
    - **Partial autocorrelation (PACF)**: Cuts off after lag 1
    """)

    # AR(1) Simulation
    st.markdown("#### AR(1) Simulation")

    col1, col2 = st.columns([1, 2])

    with col1:
        ar1_c = st.slider("Constant (c)", -5.0, 5.0, 0.0, 0.5, key="ar1_c")
        ar1_phi = st.slider("AR coefficient (φ₁)", -0.95, 0.95, 0.7, 0.05, key="ar1_phi")
        ar1_sigma = st.slider("Error std (σ)", 0.1, 3.0, 1.0, 0.1, key="ar1_sigma")
        ar1_n = st.slider("Sample size", 100, 500, 200, 50, key="ar1_n")
        ar1_seed = st.slider("Random seed", 1, 100, 42, 1, key="ar1_seed")

    # Generate AR(1) data
    np.random.seed(ar1_seed)
    ar1_series = np.zeros(ar1_n)
    ar1_series[0] = ar1_c / (1 - ar1_phi) if abs(ar1_phi) < 1 else 0

    for t in range(1, ar1_n):
        ar1_series[t] = ar1_c + ar1_phi * ar1_series[t - 1] + np.random.normal(0, ar1_sigma)

    # Create plots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("AR(1) Time Series", "ACF", "PACF", "Distribution"),
                        specs=[[{"colspan": 2}, None], [{}, {}]])

    # Time series
    fig.add_trace(go.Scatter(y=ar1_series, mode='lines', name='AR(1)',
                             line=dict(color='#1f77b4', width=2)), row=1, col=1)

    # ACF
    acf_vals = acf(ar1_series, nlags=20)
    fig.add_trace(go.Bar(x=list(range(21)), y=acf_vals, name='ACF',
                         marker_color='#ff7f0e'), row=2, col=1)
    fig.add_hline(y=1.96 / np.sqrt(ar1_n), line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-1.96 / np.sqrt(ar1_n), line_dash="dash", line_color="red", row=2, col=1)

    # PACF
    pacf_vals = pacf(ar1_series, nlags=20)
    fig.add_trace(go.Bar(x=list(range(21)), y=pacf_vals, name='PACF',
                         marker_color='#2ca02c'), row=2, col=2)
    fig.add_hline(y=1.96 / np.sqrt(ar1_n), line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-1.96 / np.sqrt(ar1_n), line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **Observed Properties:**
    - Series mean: {np.mean(ar1_series):.3f} (Theoretical: {ar1_c / (1 - ar1_phi) if abs(ar1_phi) < 1 else 'N/A':.3f})
    - Series variance: {np.var(ar1_series):.3f} (Theoretical: {ar1_sigma ** 2 / (1 - ar1_phi ** 2) if abs(ar1_phi) < 1 else 'N/A':.3f})
    - **ACF**: Exponential decay (geometric decay at rate φ₁)
    - **PACF**: Cuts off after lag 1 (identifies AR order)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # MA Model
    st.markdown('<div class="section-header">2. Moving Average (MA) Model</div>', unsafe_allow_html=True)

    st.markdown("### MA(q) Model Definition")

    st.markdown("An **MA(q)** model uses $q$ lagged forecast errors:")
    st.latex(
        r"Y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + ... + \theta_q \varepsilon_{t-q}")

    st.markdown("where:")
    st.markdown("""
    - $Y_t$ = current value
    - $\\mu$ = mean of the series
    - $\\theta_1, \\theta_2, ..., \\theta_q$ = moving average coefficients
    - $q$ = order of the MA model
    - $\\varepsilon_t \\sim N(0, \\sigma^2)$ = white noise innovations
    """)

    st.markdown("### MA(1) Model - Detailed Example")

    st.latex(r"Y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1}")

    st.markdown("**Characteristics of MA(1):**")
    st.markdown("""
    - **Always stationary** (regardless of θ₁ value)
    - **Invertibility condition**: $|\\theta_1| < 1$ (for unique representation)
    - **Mean**: $E[Y_t] = \\mu$
    - **Variance**: $Var(Y_t) = \\sigma^2(1 + \\theta_1^2)$
    - **ACF**: Cuts off after lag 1
    - **PACF**: Exponential decay
    """)

    # MA(1) Simulation
    st.markdown("#### MA(1) Simulation")

    col1, col2 = st.columns([1, 2])

    with col1:
        ma1_mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.5, key="ma1_mu")
        ma1_theta = st.slider("MA coefficient (θ₁)", -0.95, 0.95, 0.6, 0.05, key="ma1_theta")
        ma1_sigma = st.slider("Innovation std (σ)", 0.1, 3.0, 1.0, 0.1, key="ma1_sigma")
        ma1_n = st.slider("Sample size", 100, 500, 200, 50, key="ma1_n")
        ma1_seed = st.slider("Random seed", 1, 100, 42, 1, key="ma1_seed")

    # Generate MA(1) data
    np.random.seed(ma1_seed)
    ma1_errors = np.random.normal(0, ma1_sigma, ma1_n + 1)
    ma1_series = ma1_mu + ma1_errors[1:] + ma1_theta * ma1_errors[:-1]

    # Create plots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("MA(1) Time Series", "ACF", "PACF", "Distribution"),
                        specs=[[{"colspan": 2}, None], [{}, {}]])

    # Time series
    fig.add_trace(go.Scatter(y=ma1_series, mode='lines', name='MA(1)',
                             line=dict(color='#d62728', width=2)), row=1, col=1)

    # ACF
    acf_vals_ma = acf(ma1_series, nlags=20)
    fig.add_trace(go.Bar(x=list(range(21)), y=acf_vals_ma, name='ACF',
                         marker_color='#ff7f0e'), row=2, col=1)
    fig.add_hline(y=1.96 / np.sqrt(ma1_n), line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-1.96 / np.sqrt(ma1_n), line_dash="dash", line_color="red", row=2, col=1)

    # PACF
    pacf_vals_ma = pacf(ma1_series, nlags=20)
    fig.add_trace(go.Bar(x=list(range(21)), y=pacf_vals_ma, name='PACF',
                         marker_color='#2ca02c'), row=2, col=2)
    fig.add_hline(y=1.96 / np.sqrt(ma1_n), line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-1.96 / np.sqrt(ma1_n), line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **Observed Properties:**
    - Series mean: {np.mean(ma1_series):.3f} (Theoretical: {ma1_mu:.3f})
    - Series variance: {np.var(ma1_series):.3f} (Theoretical: {ma1_sigma ** 2 * (1 + ma1_theta ** 2):.3f})
    - **ACF**: Cuts off after lag 1 (identifies MA order)
    - **PACF**: Exponential decay
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # ARMA Model
    st.markdown('<div class="section-header">3. ARMA(p,q) Model</div>', unsafe_allow_html=True)

    st.markdown("### ARMA(p,q) Model Definition")

    st.markdown("An **ARMA(p,q)** model combines AR and MA components:")
    st.latex(
        r"Y_t = c + \phi_1 Y_{t-1} + ... + \phi_p Y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + ... + \theta_q \varepsilon_{t-q}")

    st.markdown("**Compact notation using lag operators:**")
    st.latex(r"\phi(L) Y_t = c + \theta(L) \varepsilon_t")

    st.markdown("where:")
    st.latex(
        r"\begin{align*} \phi(L) &= 1 - \phi_1 L - \phi_2 L^2 - ... - \phi_p L^p \\ \theta(L) &= 1 + \theta_1 L + \theta_2 L^2 + ... + \theta_q L^q \end{align*}")

    st.markdown("### ARMA(1,1) Model Example")

    st.latex(r"Y_t = c + \phi_1 Y_{t-1} + \varepsilon_t + \theta_1 \varepsilon_{t-1}")

    st.markdown("**Characteristics:**")
    st.markdown("""
    - Combines both AR and MA effects
    - More flexible than pure AR or MA models
    - Can model more complex autocorrelation patterns
    - **ACF and PACF both decay** (no clear cutoff)
    """)

    # Model Identification Table
    st.markdown('<div class="section-header">Model Identification Guide</div>', unsafe_allow_html=True)

    identification_df = pd.DataFrame({
        'Model': ['AR(p)', 'MA(q)', 'ARMA(p,q)'],
        'ACF Pattern': ['Exponential decay or damped sine wave', 'Cuts off after lag q',
                        'Exponential decay (no cutoff)'],
        'PACF Pattern': ['Cuts off after lag p', 'Exponential decay or damped sine wave',
                         'Exponential decay (no cutoff)'],
        'Use Case': ['When recent values influence future', 'When recent shocks influence future',
                     'Complex dependencies']
    })

    st.dataframe(identification_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    **Important Notes:**

    1. **Stationarity**: All ARMA models require the series to be stationary (I(0))
    2. **Parsimony**: Choose the simplest model that fits well (lowest p and q)
    3. **Information Criteria**: Use AIC or BIC to compare models
    4. **Residual Diagnostics**: Check that residuals are white noise
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 5: ARIMA MODELS
# ============================================================================
elif page == "🎯 ARIMA Models":
    st.markdown('<div class="main-header">ARIMA Models: AutoRegressive Integrated Moving Average</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ## What is ARIMA? 🎯

    **ARIMA** extends ARMA models to handle **non-stationary** time series by incorporating **differencing**.

    ARIMA combines three components:
    - **AR (p)**: Autoregressive order
    - **I (d)**: Integration order (number of differences)
    - **MA (q)**: Moving average order
    """)

    # ARIMA Definition
    st.markdown('<div class="section-header">ARIMA(p,d,q) Model Definition</div>', unsafe_allow_html=True)

    st.markdown("An **ARIMA(p,d,q)** model is defined as:")
    st.latex(r"\phi(L)(1-L)^d Y_t = c + \theta(L)\varepsilon_t")

    st.markdown("Equivalently:")
    st.latex(r"\phi(L) \nabla^d Y_t = c + \theta(L)\varepsilon_t")

    st.markdown("where:")
    st.markdown("""
    - $\\phi(L) = 1 - \\phi_1 L - ... - \\phi_p L^p$ (AR polynomial)
    - $(1-L)^d = \\nabla^d$ (differencing operator applied d times)
    - $\\theta(L) = 1 + \\theta_1 L + ... + \\theta_q L^q$ (MA polynomial)
    - $Y_t$ is the original (possibly non-stationary) series
    - $\\nabla^d Y_t$ is the differenced (stationary) series
    """)

    st.markdown("### Step-by-Step Process:")

    st.markdown("""
    1. **Difference** the series d times: $W_t = \\nabla^d Y_t$
    2. **Fit ARMA(p,q)** to the differenced series $W_t$
    3. **Forecast** $W_t$ using the ARMA model
    4. **Reverse differencing** to get forecasts for $Y_t$
    """)

    # Common ARIMA Models
    st.markdown('<div class="section-header">Common ARIMA Models</div>', unsafe_allow_html=True)

    st.markdown("### 1. ARIMA(0,1,0) - Random Walk")

    st.latex(r"\nabla Y_t = Y_t - Y_{t-1} = c + \varepsilon_t")
    st.latex(r"Y_t = c + Y_{t-1} + \varepsilon_t")

    st.markdown("- Simplest non-stationary model")
    st.markdown("- Best forecast is the last observed value plus drift")
    st.markdown("- Used when first difference is white noise")

    st.markdown("### 2. ARIMA(1,1,0) - Differenced AR(1)")

    st.latex(r"\nabla Y_t = c + \phi_1 \nabla Y_{t-1} + \varepsilon_t")

    st.markdown("- First difference follows AR(1)")
    st.markdown("- Captures momentum in changes")

    st.markdown("### 3. ARIMA(0,1,1) - Exponential Smoothing")

    st.latex(r"\nabla Y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1}")

    st.markdown("- Equivalent to exponential smoothing")
    st.markdown("- Good for series with no clear trend pattern in differences")

    # Interactive ARIMA Simulation
    st.markdown('<div class="section-header">Interactive ARIMA Simulation</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**ARIMA Parameters**")
        arima_p = st.selectbox("AR order (p)", [0, 1, 2], index=1, key="arima_p")
        arima_d = st.selectbox("Integration order (d)", [0, 1, 2], index=1, key="arima_d")
        arima_q = st.selectbox("MA order (q)", [0, 1, 2], index=1, key="arima_q")

        st.markdown("**Coefficients**")
        if arima_p > 0:
            ar_coefs = []
            for i in range(arima_p):
                ar_coefs.append(st.slider(f"φ_{i + 1}", -0.95, 0.95, 0.5, 0.05, key=f"ar_{i}"))

        if arima_q > 0:
            ma_coefs = []
            for i in range(arima_q):
                ma_coefs.append(st.slider(f"θ_{i + 1}", -0.95, 0.95, 0.3, 0.05, key=f"ma_{i}"))

        arima_const = st.slider("Constant/Drift", -1.0, 1.0, 0.1, 0.1, key="arima_const")
        arima_sigma = st.slider("Innovation std (σ)", 0.1, 3.0, 1.0, 0.1, key="arima_sigma")
        arima_n = st.slider("Sample size", 100, 500, 200, 50, key="arima_n")
        arima_seed = st.slider("Random seed", 1, 100, 42, 1, key="arima_seed")

    # Generate ARIMA data
    np.random.seed(arima_seed)

    # Start with innovations
    innovations = np.random.normal(0, arima_sigma, arima_n + 100)

    # Generate ARMA part (on differenced series)
    arma_series = np.zeros(arima_n + 100)

    for t in range(max(arima_p, arima_q), arima_n + 100):
        value = arima_const

        # AR part
        if arima_p > 0:
            for i in range(arima_p):
                value += ar_coefs[i] * arma_series[t - i - 1]

        # MA part
        value += innovations[t]
        if arima_q > 0:
            for i in range(arima_q):
                value += ma_coefs[i] * innovations[t - i - 1]

        arma_series[t] = value

    # Apply integration (reverse differencing)
    integrated_series = arma_series.copy()
    for _ in range(arima_d):
        integrated_series = np.cumsum(integrated_series)

    # Take last arima_n points
    final_series = integrated_series[-arima_n:]

    # Create visualization
    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=(f"ARIMA({arima_p},{arima_d},{arima_q}) Series",
                                        "ACF of Original",
                                        f"Differenced Series (d={arima_d})" if arima_d > 0 else "Original (already stationary)",
                                        "ACF of Differenced",
                                        "PACF of Differenced",
                                        "Residual Distribution"),
                        specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
                        vertical_spacing=0.1)

    # Original series
    fig.add_trace(go.Scatter(y=final_series, mode='lines', name=f'ARIMA({arima_p},{arima_d},{arima_q})',
                             line=dict(color='#1f77b4', width=2)), row=1, col=1)

    # ACF of original
    acf_orig = acf(final_series, nlags=20)
    fig.add_trace(go.Bar(x=list(range(21)), y=acf_orig, name='ACF Original',
                         marker_color='#ff7f0e', showlegend=False), row=1, col=2)
    fig.add_hline(y=1.96 / np.sqrt(arima_n), line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=-1.96 / np.sqrt(arima_n), line_dash="dash", line_color="red", row=1, col=2)

    # Differenced series
    if arima_d > 0:
        diff_series = final_series.copy()
        for _ in range(arima_d):
            diff_series = np.diff(diff_series)
    else:
        diff_series = final_series

    fig.add_trace(go.Scatter(y=diff_series, mode='lines', name='Differenced',
                             line=dict(color='#2ca02c', width=2), showlegend=False), row=2, col=1)

    # ACF of differenced
    acf_diff = acf(diff_series, nlags=20)
    fig.add_trace(go.Bar(x=list(range(21)), y=acf_diff, name='ACF Diff',
                         marker_color='#d62728', showlegend=False), row=2, col=2)
    fig.add_hline(y=1.96 / np.sqrt(len(diff_series)), line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-1.96 / np.sqrt(len(diff_series)), line_dash="dash", line_color="red", row=2, col=2)

    # PACF of differenced
    pacf_diff = pacf(diff_series, nlags=20)
    fig.add_trace(go.Bar(x=list(range(21)), y=pacf_diff, name='PACF Diff',
                         marker_color='#9467bd', showlegend=False), row=3, col=1)
    fig.add_hline(y=1.96 / np.sqrt(len(diff_series)), line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-1.96 / np.sqrt(len(diff_series)), line_dash="dash", line_color="red", row=3, col=1)

    # Histogram of residuals (innovations)
    fig.add_trace(go.Histogram(x=innovations[-arima_n:], name='Innovations',
                               marker_color='#8c564b', showlegend=False), row=3, col=2)

    fig.update_layout(height=1000, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Statistical tests
    st.markdown("### Stationarity Tests")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Series**")
        adf_orig = adfuller(final_series)
        st.metric("ADF Statistic", f"{adf_orig[0]:.4f}")
        st.metric("p-value", f"{adf_orig[1]:.4f}")
        if adf_orig[1] < 0.05:
            st.success("✅ Stationary")
        else:
            st.error("❌ Non-stationary")

    with col2:
        st.markdown(f"**After {arima_d} Difference(s)**")
        if arima_d > 0:
            adf_diff = adfuller(diff_series)
            st.metric("ADF Statistic", f"{adf_diff[0]:.4f}")
            st.metric("p-value", f"{adf_diff[1]:.4f}")
            if adf_diff[1] < 0.05:
                st.success("✅ Stationary")
            else:
                st.error("❌ Non-stationary")
        else:
            st.info("No differencing applied (d=0)")

    # Model Summary
    st.markdown('<div class="section-header">Model Specification Summary</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **ARIMA({arima_p},{arima_d},{arima_q}) Model:**

    **Mathematical Form:**
    """)

    # Build equation string
    if arima_d > 0:
        st.latex(r"\nabla^{" + str(arima_d) + r"} Y_t = " +
                 (f"{arima_const:.3f}" if arima_const != 0 else "") +
                 ("".join([f" + {ar_coefs[i]:.3f}\\nabla^{arima_d} Y_{{t-{i + 1}}}" for i in
                           range(arima_p)]) if arima_p > 0 else "") +
                 r" + \varepsilon_t" +
                 ("".join([f" + {ma_coefs[i]:.3f}\\varepsilon_{{t-{i + 1}}}" for i in
                           range(arima_q)]) if arima_q > 0 else ""))
    else:
        st.latex(r"Y_t = " +
                 (f"{arima_const:.3f}" if arima_const != 0 else "") +
                 ("".join([f" + {ar_coefs[i]:.3f}Y_{{t-{i + 1}}}" for i in range(arima_p)]) if arima_p > 0 else "") +
                 r" + \varepsilon_t" +
                 ("".join([f" + {ma_coefs[i]:.3f}\\varepsilon_{{t-{i + 1}}}" for i in
                           range(arima_q)]) if arima_q > 0 else ""))

    st.markdown(f"""
    **Parameters:**
    - AR coefficients (p={arima_p}): {[f"{c:.3f}" for c in ar_coefs] if arima_p > 0 else 'None'}
    - Integration order (d={arima_d})
    - MA coefficients (q={arima_q}): {[f"{c:.3f}" for c in ma_coefs] if arima_q > 0 else 'None'}
    - Constant/Drift: {arima_const:.3f}
    - Innovation variance: {arima_sigma ** 2:.3f}
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Box-Jenkins Methodology
    st.markdown('<div class="section-header">Box-Jenkins Methodology for ARIMA</div>', unsafe_allow_html=True)

    st.markdown("""
    The **Box-Jenkins approach** is a systematic method for building ARIMA models:

    ### Step 1: Identification
    - **Check stationarity** using ADF test and visual inspection
    - **Determine d**: Number of differences needed for stationarity
    - **Examine ACF and PACF** of differenced series to identify p and q

    ### Step 2: Estimation
    - **Estimate model parameters** using maximum likelihood estimation
    - **Compare models** using information criteria (AIC, BIC)
    - Choose the most parsimonious model

    ### Step 3: Diagnostic Checking
    - **Residual analysis**: Check if residuals are white noise
    - **Ljung-Box test**: Test for autocorrelation in residuals
    - **Normality tests**: Check residual distribution

    ### Step 4: Forecasting
    - Use the validated model for prediction
    - Generate confidence intervals
    - Monitor forecast performance
    """)

    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    **Model Selection Tips:**

    1. **Start simple**: Begin with low orders (p, d, q ≤ 2)
    2. **Use information criteria**: Lower AIC/BIC is better
    3. **Check residuals**: Should be white noise (no patterns)
    4. **Avoid overfitting**: More parameters ≠ better model
    5. **Validate**: Test on out-of-sample data
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 6: INTERACTIVE SIMULATION
# ============================================================================
elif page == "🧪 Interactive Simulation":
    st.markdown('<div class="main-header">Interactive ARIMA Simulation & Forecasting</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Build Your Own ARIMA Model 🔧

    Use this interactive tool to:
    1. Generate synthetic time series data
    2. Fit ARIMA models
    3. Evaluate model performance
    4. Generate forecasts with confidence intervals
    """)

    # Sidebar for data generation
    st.sidebar.markdown("### Data Generation")
    data_source = st.sidebar.radio("Data source:", ["Generate Synthetic", "Upload CSV"])

    if data_source == "Generate Synthetic":
        gen_model = st.sidebar.selectbox("True model:", [
            "ARIMA(1,1,0)", "ARIMA(0,1,1)", "ARIMA(1,1,1)",
            "ARIMA(2,1,0)", "ARIMA(0,1,2)", "Random Walk"
        ])

        n_obs = st.sidebar.slider("Number of observations:", 100, 500, 200)
        noise_level = st.sidebar.slider("Noise level:", 0.1, 3.0, 1.0, 0.1)
        data_seed = st.sidebar.slider("Random seed:", 1, 100, 42)

        # Generate data based on model
        np.random.seed(data_seed)
        innovations = np.random.normal(0, noise_level, n_obs + 50)

        if gen_model == "ARIMA(1,1,0)":
            # AR(1) on first difference
            diff_series = np.zeros(n_obs + 50)
            phi1 = 0.6
            for t in range(1, n_obs + 50):
                diff_series[t] = 0.1 + phi1 * diff_series[t - 1] + innovations[t]
            data = np.cumsum(diff_series[-n_obs:])
            true_params = (1, 1, 0)

        elif gen_model == "ARIMA(0,1,1)":
            # MA(1) on first difference
            diff_series = np.zeros(n_obs + 50)
            theta1 = 0.5
            for t in range(1, n_obs + 50):
                diff_series[t] = 0.1 + innovations[t] + theta1 * innovations[t - 1]
            data = np.cumsum(diff_series[-n_obs:])
            true_params = (0, 1, 1)

        elif gen_model == "ARIMA(1,1,1)":
            # ARMA(1,1) on first difference
            diff_series = np.zeros(n_obs + 50)
            phi1, theta1 = 0.5, 0.3
            for t in range(1, n_obs + 50):
                diff_series[t] = 0.1 + phi1 * diff_series[t - 1] + innovations[t] + theta1 * innovations[t - 1]
            data = np.cumsum(diff_series[-n_obs:])
            true_params = (1, 1, 1)

        elif gen_model == "ARIMA(2,1,0)":
            # AR(2) on first difference
            diff_series = np.zeros(n_obs + 50)
            phi1, phi2 = 0.5, -0.3
            for t in range(2, n_obs + 50):
                diff_series[t] = 0.1 + phi1 * diff_series[t - 1] + phi2 * diff_series[t - 2] + innovations[t]
            data = np.cumsum(diff_series[-n_obs:])
            true_params = (2, 1, 0)

        elif gen_model == "ARIMA(0,1,2)":
            # MA(2) on first difference
            diff_series = np.zeros(n_obs + 50)
            theta1, theta2 = 0.5, 0.3
            for t in range(2, n_obs + 50):
                diff_series[t] = 0.1 + innovations[t] + theta1 * innovations[t - 1] + theta2 * innovations[t - 2]
            data = np.cumsum(diff_series[-n_obs:])
            true_params = (0, 1, 2)

        else:  # Random Walk
            data = np.cumsum(0.1 + innovations[-n_obs:])
            true_params = (0, 1, 0)

        ts_data = pd.Series(data, index=pd.date_range(start='2020-01-01', periods=len(data), freq='D'))

    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.write("Columns:", df.columns.tolist())
            date_col = st.sidebar.selectbox("Select date column:", df.columns)
            value_col = st.sidebar.selectbox("Select value column:", df.columns)

            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            ts_data = df[value_col]
            true_params = None
        else:
            st.info("Please upload a CSV file with time series data")
            st.stop()

    # Main analysis section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Time Series Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines',
                                 name='Original Series', line=dict(color='#1f77b4', width=2)))
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Summary Statistics")
        st.metric("Observations", len(ts_data))
        st.metric("Mean", f"{ts_data.mean():.3f}")
        st.metric("Std Dev", f"{ts_data.std():.3f}")
        st.metric("Min", f"{ts_data.min():.3f}")
        st.metric("Max", f"{ts_data.max():.3f}")

        # Stationarity test
        adf_result = adfuller(ts_data)
        st.markdown("**ADF Test**")
        st.metric("p-value", f"{adf_result[1]:.4f}")
        if adf_result[1] < 0.05:
            st.success("Stationary")
        else:
            st.error("Non-stationary")

    # ACF and PACF plots
    st.markdown("### ACF and PACF Analysis")

    col1, col2 = st.columns(2)

    with col1:
        acf_vals = acf(ts_data, nlags=30)
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=list(range(31)), y=acf_vals, marker_color='#ff7f0e'))
        fig_acf.add_hline(y=1.96 / np.sqrt(len(ts_data)), line_dash="dash", line_color="red")
        fig_acf.add_hline(y=-1.96 / np.sqrt(len(ts_data)), line_dash="dash", line_color="red")
        fig_acf.update_layout(title="ACF", height=300, xaxis_title="Lag", yaxis_title="Correlation")
        st.plotly_chart(fig_acf, use_container_width=True)

    with col2:
        pacf_vals = pacf(ts_data, nlags=30)
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(x=list(range(31)), y=pacf_vals, marker_color='#2ca02c'))
        fig_pacf.add_hline(y=1.96 / np.sqrt(len(ts_data)), line_dash="dash", line_color="red")
        fig_pacf.add_hline(y=-1.96 / np.sqrt(len(ts_data)), line_dash="dash", line_color="red")
        fig_pacf.update_layout(title="PACF", height=300, xaxis_title="Lag", yaxis_title="Correlation")
        st.plotly_chart(fig_pacf, use_container_width=True)

    # Model fitting section
    st.markdown("---")
    st.markdown("### Model Fitting")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fit_p = st.selectbox("AR order (p):", [0, 1, 2, 3], index=1)
    with col2:
        fit_d = st.selectbox("Differencing (d):", [0, 1, 2], index=1)
    with col3:
        fit_q = st.selectbox("MA order (q):", [0, 1, 2, 3], index=1)
    with col4:
        n_forecast = st.slider("Forecast periods:", 10, 100, 30)

    if st.button("Fit Model & Forecast", type="primary"):
        with st.spinner("Fitting ARIMA model..."):
            try:
                # Fit model
                model = ARIMA(ts_data, order=(fit_p, fit_d, fit_q))
                fitted_model = model.fit()

                # Generate forecasts
                forecast_result = fitted_model.forecast(steps=n_forecast)
                forecast_index = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1),
                                               periods=n_forecast, freq='D')

                # Get confidence intervals
                forecast_df = fitted_model.get_forecast(steps=n_forecast)
                forecast_ci = forecast_df.conf_int()

                # Display results
                st.success(f"✅ Model ARIMA({fit_p},{fit_d},{fit_q}) fitted successfully!")

                # Model summary
                st.markdown("### Model Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AIC", f"{fitted_model.aic:.2f}")
                with col2:
                    st.metric("BIC", f"{fitted_model.bic:.2f}")
                with col3:
                    st.metric("Log-Likelihood", f"{fitted_model.llf:.2f}")

                # Coefficients
                st.markdown("**Estimated Coefficients:**")
                coef_df = pd.DataFrame({
                    'Parameter': fitted_model.params.index,
                    'Coefficient': fitted_model.params.values,
                    'Std Error': fitted_model.bse.values,
                    'p-value': fitted_model.pvalues.values
                })
                st.dataframe(coef_df, use_container_width=True, hide_index=True)

                # Forecast plot
                st.markdown("### Forecast Visualization")

                fig = go.Figure()

                # Historical data
                fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values,
                                         mode='lines', name='Historical',
                                         line=dict(color='#1f77b4', width=2)))

                # Forecasts
                fig.add_trace(go.Scatter(x=forecast_index, y=forecast_result,
                                         mode='lines', name='Forecast',
                                         line=dict(color='#ff7f0e', width=2, dash='dash')))

                # Confidence intervals
                fig.add_trace(go.Scatter(x=forecast_index, y=forecast_ci.iloc[:, 0],
                                         mode='lines', name='Lower CI',
                                         line=dict(color='rgba(255,127,14,0.3)', width=1),
                                         showlegend=False))
                fig.add_trace(go.Scatter(x=forecast_index, y=forecast_ci.iloc[:, 1],
                                         mode='lines', name='Upper CI',
                                         line=dict(color='rgba(255,127,14,0.3)', width=1),
                                         fill='tonexty', fillcolor='rgba(255,127,14,0.2)',
                                         showlegend=True))

                fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Value",
                                  title=f"ARIMA({fit_p},{fit_d},{fit_q}) Forecast")
                st.plotly_chart(fig, use_container_width=True)

                # Residual diagnostics
                st.markdown("### Residual Diagnostics")

                residuals = fitted_model.resid

                fig_resid = make_subplots(rows=2, cols=2,
                                          subplot_titles=("Residuals Over Time", "Residual ACF",
                                                          "Residual Distribution", "Q-Q Plot"))

                # Residuals plot
                fig_resid.add_trace(go.Scatter(y=residuals, mode='lines',
                                               line=dict(color='#d62728', width=1)), row=1, col=1)
                fig_resid.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

                # Residual ACF
                resid_acf = acf(residuals, nlags=20)
                fig_resid.add_trace(go.Bar(x=list(range(21)), y=resid_acf,
                                           marker_color='#9467bd'), row=1, col=2)
                fig_resid.add_hline(y=1.96 / np.sqrt(len(residuals)), line_dash="dash",
                                    line_color="red", row=1, col=2)
                fig_resid.add_hline(y=-1.96 / np.sqrt(len(residuals)), line_dash="dash",
                                    line_color="red", row=1, col=2)

                # Histogram
                fig_resid.add_trace(go.Histogram(x=residuals, marker_color='#8c564b',
                                                 nbinsx=30), row=2, col=1)

                # Q-Q plot
                from scipy import stats

                (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
                fig_resid.add_trace(go.Scatter(x=osm, y=osr, mode='markers',
                                               marker=dict(color='#e377c2', size=4)), row=2, col=2)
                fig_resid.add_trace(go.Scatter(x=osm, y=slope * osm + intercept,
                                               mode='lines', line=dict(color='red', dash='dash')),
                                    row=2, col=2)

                fig_resid.update_layout(height=700, showlegend=False)
                st.plotly_chart(fig_resid, use_container_width=True)

                # Residual statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{residuals.mean():.4f}")
                with col2:
                    st.metric("Std Dev", f"{residuals.std():.4f}")
                with col3:
                    st.metric("Skewness", f"{stats.skew(residuals):.4f}")
                with col4:
                    st.metric("Kurtosis", f"{stats.kurtosis(residuals):.4f}")

                # Ljung-Box test
                from statsmodels.stats.diagnostic import acorr_ljungbox

                lb_test = acorr_ljungbox(residuals, lags=20, return_df=True)

                st.markdown("**Ljung-Box Test for Residual Autocorrelation:**")
                st.markdown("(Tests if residuals are white noise)")

                if (lb_test['lb_pvalue'] > 0.05).all():
                    st.success("✅ Residuals appear to be white noise (no autocorrelation)")
                else:
                    st.warning("⚠️ Some autocorrelation detected in residuals")

                with st.expander("View Ljung-Box Test Results"):
                    st.dataframe(lb_test, use_container_width=True)

            except Exception as e:
                st.error(f"Error fitting model: {str(e)}")
                st.info("Try different model parameters or check your data")

# ============================================================================
# PAGE 7: SUMMARY & COMPARISON
# ============================================================================
elif page == "📝 Summary & Comparison":
    st.markdown('<div class="main-header">Summary & Model Comparison</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Complete Overview of Time Series Models 📚

    This section provides a comprehensive comparison of all models covered in this tutorial.
    """)

    # Model Comparison Table
    st.markdown('<div class="section-header">Model Comparison Table</div>', unsafe_allow_html=True)

    comparison_data = {
        'Model': ['AR(p)', 'MA(q)', 'ARMA(p,q)', 'ARIMA(p,d,q)'],
        'Full Name': ['Autoregressive', 'Moving Average', 'Autoregressive Moving Average',
                      'Autoregressive Integrated Moving Average'],
        'Equation': ['Yₜ = c + φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + εₜ',
                     'Yₜ = μ + εₜ + θ₁εₜ₋₁ + ... + θ_qεₜ₋q',
                     'Yₜ = c + φ₁Yₜ₋₁ + ... + εₜ + θ₁εₜ₋₁ + ...',
                     '∇ᵈYₜ follows ARMA(p,q)'],
        'Data Requirement': ['Stationary (I(0))', 'Stationary (I(0))', 'Stationary (I(0))',
                             'Can handle non-stationary'],
        'ACF Pattern': ['Exponential decay', 'Cuts off after lag q', 'Gradual decay',
                        'Depends on differenced series'],
        'PACF Pattern': ['Cuts off after lag p', 'Exponential decay', 'Gradual decay',
                         'Depends on differenced series'],
        'Parameters': ['p, c, σ²', 'q, μ, σ²', 'p, q, c, σ²', 'p, d, q, c, σ²'],
        'Use Case': ['Momentum effects', 'Shock effects', 'Complex patterns',
                     'Non-stationary series']
    }

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Stationarity vs Non-Stationarity
    st.markdown('<div class="section-header">Stationarity: Key Concepts</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Stationary Series (I(0))")
        st.markdown("""
        **Characteristics:**
        - Constant mean over time
        - Constant variance
        - Autocorrelation depends only on lag
        - No trend or seasonality

        **Examples:**
        - White noise
        - AR(p) with |φ| < 1
        - MA(q) models
        - Returns on financial assets

        **Models to use:**
        - AR, MA, ARMA
        """)

    with col2:
        st.markdown("### Non-Stationary Series (I(d), d>0)")
        st.markdown("""
        **Characteristics:**
        - Changing mean/variance over time
        - Trends present
        - Variance increases with time
        - Autocorrelation doesn't decay

        **Examples:**
        - Random walk
        - Stock prices
        - GDP growth
        - Temperature records

        **Models to use:**
        - ARIMA (after differencing)
        """)

    # Trend Types Comparison
    st.markdown('<div class="section-header">Trend Types: Quick Reference</div>', unsafe_allow_html=True)

    trend_data = {
        'Aspect': ['Definition', 'Removal Method', 'Effect of Shocks', 'Variance',
                   'Forecasting', 'Example'],
        'Deterministic Trend': [
            'Time-based function (α + βt)',
            'De-trending (subtract trend)',
            'Temporary (mean-reverting)',
            'Constant over time',
            'Predictable pattern',
            'Linear growth in population'
        ],
        'Stochastic Trend': [
            'Random accumulation (∑εᵢ)',
            'Differencing',
            'Permanent (no reversion)',
            'Increases with time (tσ²)',
            'Uncertainty grows',
            'Stock prices (random walk)'
        ]
    }

    trend_df = pd.DataFrame(trend_data)
    st.dataframe(trend_df, use_container_width=True, hide_index=True)

    # Model Selection Guide
    st.markdown('<div class="section-header">Model Selection: Step-by-Step Guide</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1️⃣ Check Stationarity

    **Visual Inspection:**
    - Plot the series
    - Look for trends, changing variance, or level shifts

    **Statistical Test:**
    - Use **ADF test** (Augmented Dickey-Fuller)
    - If p-value < 0.05 → Stationary
    - If p-value ≥ 0.05 → Non-stationary

    ### 2️⃣ Determine Integration Order (d)

    - If stationary: **d = 0** (use ARMA)
    - If not stationary:
      - Apply first difference
      - Test again with ADF
      - If now stationary: **d = 1**
      - If still not stationary: apply second difference, **d = 2**

    ### 3️⃣ Identify AR order (p) and MA order (q)

    **Using ACF and PACF:**

    | Model | ACF | PACF |
    |-------|-----|------|
    | AR(p) | Gradual decay | Cuts off after lag p |
    | MA(q) | Cuts off after lag q | Gradual decay |
    | ARMA(p,q) | Gradual decay | Gradual decay |

    **Alternatively, use Information Criteria:**
    - Fit multiple models with different (p,q)
    - Choose model with **lowest AIC or BIC**

    ### 4️⃣ Estimate and Validate

    - Fit the selected ARIMA(p,d,q) model
    - Check residuals are white noise
    - Verify coefficients are significant
    - Test on out-of-sample data

    ### 5️⃣ Forecast

    - Generate point forecasts
    - Compute confidence intervals
    - Monitor and update as new data arrives
    """)

    # Common Pitfalls
    st.markdown('<div class="section-header">Common Pitfalls & Solutions</div>', unsafe_allow_html=True)

    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    ### ⚠️ Pitfall 1: Over-differencing

    **Problem:** Differencing a stationary series creates artificial patterns

    **Solution:** 
    - Always test stationarity after each difference
    - Most series need at most d=1 or d=2

    ### ⚠️ Pitfall 2: Overfitting

    **Problem:** Using too many parameters (high p and q)

    **Solution:**
    - Prefer simpler models (parsimony principle)
    - Use information criteria (AIC/BIC)
    - Validate on holdout data

    ### ⚠️ Pitfall 3: Ignoring Residual Diagnostics

    **Problem:** Not checking if residuals are white noise

    **Solution:**
    - Always plot residuals
    - Run Ljung-Box test
    - Check ACF of residuals

    ### ⚠️ Pitfall 4: Structural Breaks

    **Problem:** Model assumes stable relationships, but data has regime changes

    **Solution:**
    - Split data at break points
    - Use rolling window estimation
    - Consider advanced models (e.g., GARCH, regime-switching)

    ### ⚠️ Pitfall 5: Confusing Correlation with Causation

    **Problem:** ARIMA models don't establish causality

    **Solution:**
    - Use for forecasting, not causal inference
    - Consider VAR models for multivariate relationships
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Key Formulas Summary
    st.markdown('<div class="section-header">Key Formulas: Quick Reference</div>', unsafe_allow_html=True)

    st.markdown("### Lag Operator")
    st.latex(r"L Y_t = Y_{t-1}, \quad L^k Y_t = Y_{t-k}")

    st.markdown("### Difference Operator")
    st.latex(r"\nabla Y_t = (1-L)Y_t = Y_t - Y_{t-1}")
    st.latex(r"\nabla^2 Y_t = (1-L)^2 Y_t = Y_t - 2Y_{t-1} + Y_{t-2}")

    st.markdown("### AR(p) Process")
    st.latex(r"\phi(L) Y_t = c + \varepsilon_t")
    st.latex(r"\phi(L) = 1 - \phi_1 L - \phi_2 L^2 - ... - \phi_p L^p")

    st.markdown("### MA(q) Process")
    st.latex(r"Y_t = \mu + \theta(L) \varepsilon_t")
    st.latex(r"\theta(L) = 1 + \theta_1 L + \theta_2 L^2 + ... + \theta_q L^q")

    st.markdown("### ARMA(p,q) Process")
    st.latex(r"\phi(L) Y_t = c + \theta(L) \varepsilon_t")

    st.markdown("### ARIMA(p,d,q) Process")
    st.latex(r"\phi(L) (1-L)^d Y_t = c + \theta(L) \varepsilon_t")

    # Best Practices
    st.markdown('<div class="section-header">Best Practices Summary</div>', unsafe_allow_html=True)

    st.markdown("""
    ### ✅ Do's

    1. **Always visualize** your data first
    2. **Test for stationarity** before modeling
    3. **Start simple** (low-order models)
    4. **Check residuals** thoroughly
    5. **Use multiple evaluation metrics** (AIC, BIC, RMSE, MAE)
    6. **Validate forecasts** on out-of-sample data
    7. **Update models** as new data arrives
    8. **Document assumptions** and limitations

    ### ❌ Don'ts

    1. **Don't skip** stationarity testing
    2. **Don't overfit** with too many parameters
    3. **Don't ignore** seasonal patterns
    4. **Don't extrapolate** too far into the future
    5. **Don't assume** ARIMA is always appropriate
    6. **Don't forget** confidence intervals
    7. **Don't neglect** economic/domain knowledge
    8. **Don't use** ARIMA for causal inference
    """)

    # Further Learning
    st.markdown('<div class="section-header">Further Learning Resources</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 📚 Recommended Topics to Explore Next:

    1. **Seasonal ARIMA (SARIMA)**
       - Handling seasonal patterns
       - SARIMA(p,d,q)(P,D,Q)ₛ notation

    2. **ARCH/GARCH Models**
       - Modeling volatility clustering
       - Financial time series

    3. **Vector Autoregression (VAR)**
       - Multivariate time series
       - Dynamic relationships

    4. **State Space Models**
       - Kalman filtering
       - Structural time series

    5. **Machine Learning for Time Series**
       - LSTM networks
       - Prophet
       - Deep learning approaches

    ### 📖 Classic References:

    - Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015). *Time Series Analysis: Forecasting and Control*
    - Hamilton, J.D. (1994). *Time Series Analysis*
    - Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*
    """)

    # Final Summary
    st.markdown("---")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎓 Congratulations!

    You've completed this comprehensive tutorial on ARMA and ARIMA models. You now understand:

    - The difference between stationary and non-stationary series
    - Deterministic vs stochastic trends
    - Integration and differencing concepts
    - AR, MA, and ARMA model structures
    - How ARIMA extends ARMA for non-stationary data
    - Model identification, estimation, and validation
    - Practical forecasting with confidence intervals

    **Remember:** Practice is key! Use the Interactive Simulation page to experiment with different scenarios.

    Happy forecasting! 📈
    """)
    st.markdown('</div>', unsafe_allow_html=True)