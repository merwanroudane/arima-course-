import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Box-Jenkins Models Tutorial", layout="wide", page_icon="📊")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📊 Complete Guide to Box-Jenkins (ARIMA) Models")
st.markdown("### A Comprehensive Tutorial for Absolute Beginners")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section:",
    ["Introduction", "Time Series Components", "Stationarity",
     "AR Model", "MA Model", "ARMA Model", "ARIMA Model",
     "Identification Process", "Complete Example", "Model Diagnostics"]
)

# ============================================================================
# SECTION 1: INTRODUCTION
# ============================================================================
if section == "Introduction":
    st.header("1. What are Box-Jenkins Models?")

    st.markdown("""
    **Box-Jenkins models** (named after statisticians George Box and Gwilym Jenkins) are a family of 
    statistical models used for analyzing and forecasting **time series data**.

    ### What is Time Series Data?
    Time series data is a sequence of observations recorded at regular time intervals. Examples include:
    - Daily stock prices
    - Monthly temperature readings
    - Yearly GDP growth
    - Hourly website traffic
    """)

    st.subheader("The Box-Jenkins Family")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Main Models:**
        1. **AR (AutoRegressive)**: Uses past values
        2. **MA (Moving Average)**: Uses past errors
        3. **ARMA**: Combines AR and MA
        4. **ARIMA**: ARMA for non-stationary data
        """)

    with col2:
        st.markdown("""
        **Key Characteristics:**
        - Linear models
        - Assume stationarity (or can be made stationary)
        - Use past observations to predict future
        - Based on autocorrelation patterns
        """)

    st.subheader("The Box-Jenkins Methodology: 4 Steps")

    st.markdown("""
    The Box-Jenkins approach follows a systematic 4-step cycle:
    """)

    # Create a flowchart
    fig = go.Figure()

    # Define positions for the boxes
    steps = [
        {"name": "1. IDENTIFICATION", "x": 0, "y": 3, "color": "#FF6B6B"},
        {"name": "2. ESTIMATION", "x": 2, "y": 3, "color": "#4ECDC4"},
        {"name": "3. DIAGNOSTIC CHECKING", "x": 2, "y": 1, "color": "#45B7D1"},
        {"name": "4. FORECASTING", "x": 0, "y": 1, "color": "#96CEB4"}
    ]

    for step in steps:
        fig.add_trace(go.Scatter(
            x=[step["x"]], y=[step["y"]],
            mode='markers+text',
            marker=dict(size=80, color=step["color"]),
            text=[step["name"]],
            textposition="middle center",
            textfont=dict(size=10, color="white", family="Arial Black"),
            showlegend=False,
            hoverinfo='text',
            hovertext=step["name"]
        ))

    # Add arrows
    arrows = [
        {"x0": 0.3, "y0": 3, "x1": 1.7, "y1": 3},  # 1 to 2
        {"x0": 2, "y0": 2.7, "x1": 2, "y1": 1.3},  # 2 to 3
        {"x0": 1.7, "y0": 1, "x1": 0.3, "y1": 1},  # 3 to 4
        {"x0": 0, "y0": 2.7, "x1": 0, "y1": 1.3},  # 1 to 4
        {"x0": 0.3, "y0": 1.2, "x1": 1.7, "y1": 2.8}  # 4 to 2 (iteration)
    ]

    for arrow in arrows:
        fig.add_annotation(
            x=arrow["x1"], y=arrow["y1"],
            ax=arrow["x0"], ay=arrow["y0"],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='gray'
        )

    fig.update_layout(
        title="Box-Jenkins Methodology Cycle",
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Detailed Steps:**

    1. **Identification**: Determine the order of AR (p), MA (q), and differencing (d)
        - Check stationarity
        - Analyze ACF and PACF plots
        - Choose preliminary model orders

    2. **Estimation**: Estimate model parameters using statistical methods
        - Maximum Likelihood Estimation (MLE)
        - Least Squares methods

    3. **Diagnostic Checking**: Verify if the model is adequate
        - Check residuals (should be white noise)
        - Perform statistical tests
        - If inadequate, return to Step 1

    4. **Forecasting**: Use the validated model to make predictions
    """)

# ============================================================================
# SECTION 2: TIME SERIES COMPONENTS
# ============================================================================
elif section == "Time Series Components":
    st.header("2. Understanding Time Series Components")

    st.markdown("""
    Before working with Box-Jenkins models, you must understand that any time series can be 
    decomposed into several components:
    """)

    st.latex(r"Y_t = T_t + S_t + C_t + I_t")

    st.markdown("""
    Where:
    - $Y_t$ = Observed value at time $t$
    - $T_t$ = Trend component
    - $S_t$ = Seasonal component
    - $C_t$ = Cyclical component
    - $I_t$ = Irregular (random) component
    """)

    # Interactive example
    st.subheader("Interactive Decomposition Example")

    col1, col2, col3 = st.columns(3)
    with col1:
        trend_strength = st.slider("Trend Strength", 0.0, 5.0, 2.0, 0.1)
    with col2:
        seasonal_strength = st.slider("Seasonal Strength", 0.0, 10.0, 5.0, 0.1)
    with col3:
        noise_strength = st.slider("Noise Strength", 0.0, 5.0, 1.0, 0.1)

    # Generate synthetic time series
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 200)

    trend = trend_strength * t
    seasonal = seasonal_strength * np.sin(t)
    noise = noise_strength * np.random.randn(len(t))
    time_series = trend + seasonal + noise

    # Create subplots
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=('Original Time Series', 'Trend Component', 'Seasonal Component',
                        'Noise Component', 'Sum of Components'),
        vertical_spacing=0.08,
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2]
    )

    # Original series
    fig.add_trace(go.Scatter(y=time_series, mode='lines', name='Original',
                             line=dict(color='#2E86AB', width=2)), row=1, col=1)

    # Trend
    fig.add_trace(go.Scatter(y=trend, mode='lines', name='Trend',
                             line=dict(color='#A23B72', width=2)), row=2, col=1)

    # Seasonal
    fig.add_trace(go.Scatter(y=seasonal, mode='lines', name='Seasonal',
                             line=dict(color='#F18F01', width=2)), row=3, col=1)

    # Noise
    fig.add_trace(go.Scatter(y=noise, mode='lines', name='Noise',
                             line=dict(color='#C73E1D', width=1)), row=4, col=1)

    # Sum
    fig.add_trace(go.Scatter(y=time_series, mode='lines', name='Sum',
                             line=dict(color='#6A994E', width=2)), row=5, col=1)

    fig.update_layout(height=900, showlegend=False, title_text="Time Series Decomposition")
    fig.update_xaxes(title_text="Time", row=5, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Why Decomposition Matters for Box-Jenkins Models

    Box-Jenkins models (ARIMA) work best when:
    1. **No strong trend**: We remove trend through differencing (the 'I' in ARIMA)
    2. **No seasonality**: We use seasonal ARIMA (SARIMA) if seasonal patterns exist
    3. **Focus on irregular component**: AR and MA models capture the patterns in the irregular component
    """)

# ============================================================================
# SECTION 3: STATIONARITY
# ============================================================================
elif section == "Stationarity":
    st.header("3. Stationarity: The Foundation")

    st.markdown("""
    **Stationarity** is the most important concept in Box-Jenkins models. A time series is 
    **stationary** if its statistical properties do not change over time.
    """)

    st.subheader("Mathematical Definition")

    st.markdown("""
    A time series $\\{Y_t\\}$ is **strictly stationary** if:
    """)

    st.latex(r"P(Y_{t_1}, Y_{t_2}, ..., Y_{t_n}) = P(Y_{t_1+h}, Y_{t_2+h}, ..., Y_{t_n+h})")

    st.markdown("""
    For practical purposes, we use **weak stationarity** (or covariance stationarity):

    1. **Constant mean**:
    """)
    st.latex(r"E[Y_t] = \mu \quad \text{(constant for all } t\text{)}")

    st.markdown("2. **Constant variance**:")
    st.latex(r"Var[Y_t] = \sigma^2 \quad \text{(constant for all } t\text{)}")

    st.markdown("3. **Autocovariance depends only on lag**:")
    st.latex(r"Cov(Y_t, Y_{t-k}) = \gamma_k \quad \text{(depends only on } k\text{, not } t\text{)}")

    st.subheader("Visual Comparison")

    # Generate stationary and non-stationary series
    np.random.seed(42)
    n = 200

    # Stationary: White noise around constant mean
    stationary = np.random.randn(n)

    # Non-stationary: Random walk with trend
    non_stationary = np.cumsum(np.random.randn(n)) + 0.05 * np.arange(n)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Stationary Series', 'Non-Stationary Series')
    )

    fig.add_trace(go.Scatter(y=stationary, mode='lines', name='Stationary',
                             line=dict(color='#06D6A0', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(y=non_stationary, mode='lines', name='Non-Stationary',
                             line=dict(color='#EF476F', width=2)), row=1, col=2)

    # Add mean lines
    fig.add_hline(y=np.mean(stationary), line_dash="dash", line_color="green",
                  annotation_text="Mean", row=1, col=1)
    fig.add_hline(y=np.mean(non_stationary), line_dash="dash", line_color="red",
                  annotation_text="Mean (not constant!)", row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Testing for Stationarity")

    st.markdown("""
    ### 1. Augmented Dickey-Fuller (ADF) Test

    **Null Hypothesis ($H_0$)**: The series has a unit root (non-stationary)

    **Alternative Hypothesis ($H_1$)**: The series is stationary

    The test equation:
    """)

    st.latex(r"\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \epsilon_t")

    st.markdown("""
    Where:
    - $\\Delta Y_t = Y_t - Y_{t-1}$ (first difference)
    - $\\gamma$ is the coefficient we test
    - If $\\gamma = 0$, the series has a unit root (non-stationary)

    **Decision Rule**: If p-value < 0.05, reject $H_0$ → series is stationary
    """)

    # Perform ADF test
    adf_stationary = adfuller(stationary)
    adf_non_stationary = adfuller(non_stationary)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Stationary Series Test")
        st.metric("ADF Statistic", f"{adf_stationary[0]:.4f}")
        st.metric("P-value", f"{adf_stationary[1]:.4f}")
        if adf_stationary[1] < 0.05:
            st.success("✅ Series is stationary (p < 0.05)")
        else:
            st.error("❌ Series is non-stationary (p ≥ 0.05)")

    with col2:
        st.markdown("#### Non-Stationary Series Test")
        st.metric("ADF Statistic", f"{adf_non_stationary[0]:.4f}")
        st.metric("P-value", f"{adf_non_stationary[1]:.4f}")
        if adf_non_stationary[1] < 0.05:
            st.success("✅ Series is stationary (p < 0.05)")
        else:
            st.error("❌ Series is non-stationary (p ≥ 0.05)")

    st.markdown("""
    ### 2. KPSS Test

    **Null Hypothesis ($H_0$)**: The series is stationary

    **Alternative Hypothesis ($H_1$)**: The series has a unit root (non-stationary)

    **Note**: KPSS is the opposite of ADF! This is useful for confirmation.

    **Decision Rule**: If p-value < 0.05, reject $H_0$ → series is non-stationary
    """)

    st.subheader("Making Series Stationary: Differencing")

    st.markdown("""
    If a series is non-stationary, we use **differencing** to make it stationary.

    **First Difference**:
    """)
    st.latex(r"\Delta Y_t = Y_t - Y_{t-1}")

    st.markdown("**Second Difference**:")
    st.latex(r"\Delta^2 Y_t = \Delta Y_t - \Delta Y_{t-1} = (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2})")

    # Show differencing effect
    diff_1 = np.diff(non_stationary, n=1)
    diff_2 = np.diff(non_stationary, n=2)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Original Non-Stationary Series',
                        'After First Differencing',
                        'After Second Differencing'),
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(y=non_stationary, mode='lines', name='Original',
                             line=dict(color='#EF476F', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(y=diff_1, mode='lines', name='1st Diff',
                             line=dict(color='#FFD166', width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(y=diff_2, mode='lines', name='2nd Diff',
                             line=dict(color='#06D6A0', width=2)), row=3, col=1)

    fig.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Test differenced series
    adf_diff1 = adfuller(diff_1)

    st.markdown(f"""
    **ADF Test on First Differenced Series**:
    - ADF Statistic: {adf_diff1[0]:.4f}
    - P-value: {adf_diff1[1]:.6f}
    - Result: {'✅ Stationary' if adf_diff1[1] < 0.05 else '❌ Non-stationary'}
    """)

# ============================================================================
# SECTION 4: AR MODEL
# ============================================================================
elif section == "AR Model":
    st.header("4. AutoRegressive (AR) Model")

    st.markdown("""
    The **AutoRegressive (AR) model** predicts future values based on past values. 
    It's called "autoregressive" because it regresses on itself.
    """)

    st.subheader("Mathematical Formulation")

    st.markdown("""
    An **AR(p)** model of order $p$ is defined as:
    """)

    st.latex(r"Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \epsilon_t")

    st.markdown("""
    Or more compactly:
    """)

    st.latex(r"Y_t = c + \sum_{i=1}^{p} \phi_i Y_{t-i} + \epsilon_t")

    st.markdown("""
    Where:
    - $Y_t$ = Value at time $t$
    - $c$ = Constant term (intercept)
    - $\\phi_1, \\phi_2, ..., \\phi_p$ = AR coefficients (parameters to estimate)
    - $p$ = Order of the AR model (number of lags used)
    - $\\epsilon_t$ = White noise error term with mean 0 and variance $\\sigma^2$
    """)

    st.subheader("Characteristics of AR Models")

    st.markdown("""
    **1. Stationarity Condition**:

    For an AR(p) model to be stationary, the roots of the characteristic equation must lie outside the unit circle:
    """)

    st.latex(r"1 - \phi_1 z - \phi_2 z^2 - ... - \phi_p z^p = 0")

    st.markdown("""
    **Simple case for AR(1)**:
    """)
    st.latex(r"|\phi_1| < 1")

    st.markdown("""
    **2. Mean of AR(p)**:
    """)
    st.latex(r"\mu = \frac{c}{1 - \phi_1 - \phi_2 - ... - \phi_p}")

    st.markdown("""
    **3. Autocorrelation Function (ACF)**:
    - Decays exponentially (for AR(1))
    - Decays with damped sine wave pattern (for complex roots)
    - Tails off gradually

    **4. Partial Autocorrelation Function (PACF)**:
    - Cuts off after lag $p$
    - This is the **key identifier** for AR models!
    """)

    st.subheader("Example: AR(1) Model")

    st.markdown("""
    Let's examine the simplest AR model:
    """)

    st.latex(r"Y_t = c + \phi_1 Y_{t-1} + \epsilon_t")

    # Interactive AR(1) simulator
    st.markdown("### Interactive AR(1) Simulator")

    col1, col2, col3 = st.columns(3)
    with col1:
        phi1 = st.slider("φ₁ coefficient", -0.95, 0.95, 0.7, 0.05,
                         help="Must be between -1 and 1 for stationarity")
    with col2:
        c = st.slider("Constant (c)", -5.0, 5.0, 0.0, 0.5)
    with col3:
        sigma = st.slider("Noise σ", 0.1, 3.0, 1.0, 0.1)

    # Check stationarity
    if abs(phi1) < 1:
        st.success(f"✅ Model is stationary (|φ₁| = {abs(phi1):.2f} < 1)")
        theoretical_mean = c / (1 - phi1) if phi1 != 1 else 0
        st.info(f"Theoretical mean: μ = {theoretical_mean:.4f}")
    else:
        st.error(f"❌ Model is NOT stationary (|φ₁| = {abs(phi1):.2f} ≥ 1)")

    # Generate AR(1) process
    np.random.seed(42)
    n = 300
    y = np.zeros(n)
    epsilon = np.random.normal(0, sigma, n)

    for t in range(1, n):
        y[t] = c + phi1 * y[t - 1] + epsilon[t]

    # Plot the series
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode='lines', name=f'AR(1): φ₁={phi1}',
                             line=dict(color='#2E86AB', width=2)))

    if abs(phi1) < 1:
        fig.add_hline(y=theoretical_mean, line_dash="dash", line_color="red",
                      annotation_text=f"Mean = {theoretical_mean:.2f}")

    fig.update_layout(title=f"AR(1) Process: Y_t = {c} + {phi1}·Y_{{t-1}} + ε_t",
                      xaxis_title="Time", yaxis_title="Value", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ACF and PACF
    st.markdown("### ACF and PACF Analysis")

    from statsmodels.tsa.stattools import acf, pacf

    acf_values = acf(y, nlags=20)
    pacf_values = pacf(y, nlags=20)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'PACF'))

    # ACF plot
    fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values,
                         name='ACF', marker_color='#118AB2'), row=1, col=1)

    # PACF plot
    fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values,
                         name='PACF', marker_color='#EF476F'), row=1, col=2)

    # Add confidence intervals (95%)
    conf_interval = 1.96 / np.sqrt(n)
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Observations**:
    - **ACF**: Gradually decays (exponentially for positive φ₁, or oscillating for negative φ₁)
    - **PACF**: Significant spike at lag 1, then cuts off → This indicates **AR(1)**!

    The PACF cutting off after lag 1 is the signature of an AR(1) model.
    """)

    st.subheader("Higher Order AR Models")

    st.markdown("""
    **AR(2) Model**:
    """)
    st.latex(r"Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \epsilon_t")

    st.markdown("""
    **Stationarity conditions for AR(2)**:
    """)
    st.latex(r"\phi_1 + \phi_2 < 1")
    st.latex(r"\phi_2 - \phi_1 < 1")
    st.latex(r"|\phi_2| < 1")

    st.markdown("""
    **General AR(p)**:
    - PACF cuts off after lag $p$
    - ACF tails off (decays gradually)
    - Number of significant PACF spikes = order $p$
    """)

# ============================================================================
# SECTION 5: MA MODEL
# ============================================================================
elif section == "MA Model":
    st.header("5. Moving Average (MA) Model")

    st.markdown("""
    The **Moving Average (MA) model** predicts future values based on past forecast errors 
    (not past values like AR). Despite its name, it's NOT a simple moving average!
    """)

    st.subheader("Mathematical Formulation")

    st.markdown("""
    An **MA(q)** model of order $q$ is defined as:
    """)

    st.latex(
        r"Y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}")

    st.markdown("""
    Or more compactly:
    """)

    st.latex(r"Y_t = \mu + \epsilon_t + \sum_{i=1}^{q} \theta_i \epsilon_{t-i}")

    st.markdown("""
    Where:
    - $Y_t$ = Value at time $t$
    - $\\mu$ = Mean of the series
    - $\\epsilon_t, \\epsilon_{t-1}, ..., \\epsilon_{t-q}$ = White noise error terms
    - $\\theta_1, \\theta_2, ..., \\theta_q$ = MA coefficients (parameters to estimate)
    - $q$ = Order of the MA model (number of lagged errors used)
    - Each $\\epsilon_t \\sim N(0, \\sigma^2)$ independently
    """)

    st.subheader("Characteristics of MA Models")

    st.markdown("""
    **1. Stationarity**:
    - MA models are **always stationary** regardless of parameter values!
    - This is a major difference from AR models

    **2. Invertibility Condition**:

    For an MA(q) model to be invertible (can be expressed as an infinite AR), roots of:
    """)

    st.latex(r"1 + \theta_1 z + \theta_2 z^2 + ... + \theta_q z^q = 0")

    st.markdown("""
    must lie outside the unit circle.

    **Simple case for MA(1)**:
    """)
    st.latex(r"|\theta_1| < 1")

    st.markdown("""
    **3. Mean of MA(q)**:
    """)
    st.latex(r"E[Y_t] = \mu")

    st.markdown("""
    **4. Variance of MA(q)**:
    """)
    st.latex(r"Var(Y_t) = \sigma^2(1 + \theta_1^2 + \theta_2^2 + ... + \theta_q^2)")

    st.markdown("""
    **5. Autocorrelation Function (ACF)**:
    - Cuts off after lag $q$ (this is the **key identifier**!)
    - Zero for lags > $q$

    **6. Partial Autocorrelation Function (PACF)**:
    - Tails off gradually (decays exponentially or with damped oscillation)
    """)

    st.subheader("Example: MA(1) Model")

    st.markdown("""
    Let's examine the simplest MA model:
    """)

    st.latex(r"Y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1}")

    st.markdown("""
    **Theoretical ACF for MA(1)**:
    """)
    st.latex(r"\rho_1 = \frac{\theta_1}{1 + \theta_1^2}")
    st.latex(r"\rho_k = 0 \text{ for } k > 1")

    # Interactive MA(1) simulator
    st.markdown("### Interactive MA(1) Simulator")

    col1, col2, col3 = st.columns(3)
    with col1:
        theta1 = st.slider("θ₁ coefficient", -0.95, 0.95, 0.6, 0.05)
    with col2:
        mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.5)
    with col3:
        sigma_ma = st.slider("Noise σ (MA)", 0.1, 3.0, 1.0, 0.1)

    # Check invertibility
    if abs(theta1) < 1:
        st.success(f"✅ Model is invertible (|θ₁| = {abs(theta1):.2f} < 1)")
    else:
        st.error(f"❌ Model is NOT invertible (|θ₁| = {abs(theta1):.2f} ≥ 1)")

    theoretical_rho1 = theta1 / (1 + theta1 ** 2)
    st.info(f"Theoretical ACF at lag 1: ρ₁ = {theoretical_rho1:.4f}")

    # Generate MA(1) process
    np.random.seed(42)
    n = 300
    epsilon = np.random.normal(0, sigma_ma, n + 1)
    y_ma = np.zeros(n)

    for t in range(n):
        y_ma[t] = mu + epsilon[t] + theta1 * epsilon[t - 1]

    # Plot the series
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_ma, mode='lines', name=f'MA(1): θ₁={theta1}',
                             line=dict(color='#06D6A0', width=2)))
    fig.add_hline(y=mu, line_dash="dash", line_color="red",
                  annotation_text=f"Mean = {mu:.2f}")

    fig.update_layout(title=f"MA(1) Process: Y_t = {mu} + ε_t + {theta1}·ε_{{t-1}}",
                      xaxis_title="Time", yaxis_title="Value", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ACF and PACF
    st.markdown("### ACF and PACF Analysis")

    from statsmodels.tsa.stattools import acf, pacf

    acf_values_ma = acf(y_ma, nlags=20)
    pacf_values_ma = pacf(y_ma, nlags=20)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'PACF'))

    # ACF plot
    fig.add_trace(go.Bar(x=list(range(len(acf_values_ma))), y=acf_values_ma,
                         name='ACF', marker_color='#118AB2'), row=1, col=1)

    # PACF plot
    fig.add_trace(go.Bar(x=list(range(len(pacf_values_ma))), y=pacf_values_ma,
                         name='PACF', marker_color='#EF476F'), row=1, col=2)

    # Add confidence intervals
    conf_interval = 1.96 / np.sqrt(n)
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="gray", row=1, col=2)

    # Mark theoretical ACF at lag 1
    fig.add_trace(go.Scatter(x=[1], y=[theoretical_rho1], mode='markers',
                             marker=dict(size=12, color='red', symbol='x'),
                             name='Theoretical ρ₁'), row=1, col=1)

    fig.update_layout(height=400, showlegend=True)
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Observations**:
    - **ACF**: Significant spike at lag 1, then cuts off (≈0 for lag > 1) → This indicates **MA(1)**!
    - **PACF**: Gradually decays (tails off)
    - Red 'X' shows theoretical ACF value at lag 1

    The ACF cutting off after lag 1 is the signature of an MA(1) model.
    """)

    st.subheader("Comparison: AR vs MA")

    comparison_df = pd.DataFrame({
        'Property': ['Depends on', 'ACF Pattern', 'PACF Pattern', 'Stationarity',
                     'Order Identification', 'Mean Formula'],
        'AR(p)': [
            'Past values (Y_{t-1}, Y_{t-2}, ...)',
            'Tails off gradually',
            'Cuts off after lag p',
            'Conditional (depends on φ)',
            'Count PACF spikes',
            'μ = c/(1-Σφᵢ)'
        ],
        'MA(q)': [
            'Past errors (ε_{t-1}, ε_{t-2}, ...)',
            'Cuts off after lag q',
            'Tails off gradually',
            'Always stationary',
            'Count ACF spikes',
            'μ (direct parameter)'
        ]
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Key Insight**: AR and MA models are "duals" of each other:
    - AR: PACF cuts off, ACF tails off
    - MA: ACF cuts off, PACF tails off

    This duality helps us identify which model to use!
    """)

# ============================================================================
# SECTION 6: ARMA MODEL
# ============================================================================
elif section == "ARMA Model":
    st.header("6. AutoRegressive Moving Average (ARMA) Model")

    st.markdown("""
    The **ARMA model** combines both AR and MA components. It's more flexible and can 
    model a wider range of time series patterns.
    """)

    st.subheader("Mathematical Formulation")

    st.markdown("""
    An **ARMA(p,q)** model combines AR(p) and MA(q):
    """)

    st.latex(
        r"Y_t = c + \phi_1 Y_{t-1} + ... + \phi_p Y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}")

    st.markdown("""
    More compactly:
    """)

    st.latex(r"Y_t = c + \sum_{i=1}^{p} \phi_i Y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}")

    st.markdown("""
    **Using Backshift Operator Notation**:

    Define the backshift operator $B$:
    """)
    st.latex(r"B Y_t = Y_{t-1}, \quad B^k Y_t = Y_{t-k}")

    st.markdown("""
    Then ARMA(p,q) can be written as:
    """)
    st.latex(r"\phi(B) Y_t = \theta(B) \epsilon_t")

    st.markdown("""
    Where:
    """)
    st.latex(r"\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - ... - \phi_p B^p")
    st.latex(r"\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + ... + \theta_q B^q")

    st.subheader("Characteristics of ARMA Models")

    st.markdown("""
    **1. Stationarity**: 
    - Depends on AR part (roots of $\\phi(B) = 0$ must be outside unit circle)

    **2. Invertibility**: 
    - Depends on MA part (roots of $\\theta(B) = 0$ must be outside unit circle)

    **3. Mean**:
    """)
    st.latex(r"E[Y_t] = \frac{c}{1 - \phi_1 - \phi_2 - ... - \phi_p}")

    st.markdown("""
    **4. ACF and PACF**:
    - Both tail off gradually (no clear cutoff)
    - This is the signature of ARMA models!
    - Makes identification more challenging
    """)

    st.subheader("Example: ARMA(1,1) Model")

    st.markdown("""
    The simplest mixed model:
    """)

    st.latex(r"Y_t = c + \phi_1 Y_{t-1} + \epsilon_t + \theta_1 \epsilon_{t-1}")

    st.markdown("""
    **Stationarity condition**: $|\\phi_1| < 1$

    **Invertibility condition**: $|\\theta_1| < 1$
    """)

    # Interactive ARMA(1,1) simulator
    st.markdown("### Interactive ARMA(1,1) Simulator")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        phi1_arma = st.slider("φ₁ (AR)", -0.95, 0.95, 0.6, 0.05)
    with col2:
        theta1_arma = st.slider("θ₁ (MA)", -0.95, 0.95, 0.4, 0.05)
    with col3:
        c_arma = st.slider("Constant", -3.0, 3.0, 0.0, 0.5)
    with col4:
        sigma_arma = st.slider("Noise σ", 0.1, 3.0, 1.0, 0.1)

    # Check conditions
    col1, col2 = st.columns(2)
    with col1:
        if abs(phi1_arma) < 1:
            st.success(f"✅ Stationary (|φ₁| = {abs(phi1_arma):.2f} < 1)")
        else:
            st.error(f"❌ Not stationary")

    with col2:
        if abs(theta1_arma) < 1:
            st.success(f"✅ Invertible (|θ₁| = {abs(theta1_arma):.2f} < 1)")
        else:
            st.error(f"❌ Not invertible")

    # Generate ARMA(1,1) process
    np.random.seed(42)
    n = 300
    y_arma = np.zeros(n)
    epsilon_arma = np.random.normal(0, sigma_arma, n)

    for t in range(1, n):
        y_arma[t] = c_arma + phi1_arma * y_arma[t - 1] + epsilon_arma[t] + theta1_arma * epsilon_arma[t - 1]

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_arma, mode='lines',
                             name=f'ARMA(1,1): φ₁={phi1_arma}, θ₁={theta1_arma}',
                             line=dict(color='#9D4EDD', width=2)))

    if abs(phi1_arma) < 1:
        mean_arma = c_arma / (1 - phi1_arma)
        fig.add_hline(y=mean_arma, line_dash="dash", line_color="red",
                      annotation_text=f"Mean = {mean_arma:.2f}")

    fig.update_layout(
        title=f"ARMA(1,1): Y_t = {c_arma} + {phi1_arma}·Y_{{t-1}} + ε_t + {theta1_arma}·ε_{{t-1}}",
        xaxis_title="Time", yaxis_title="Value", height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # ACF and PACF
    st.markdown("### ACF and PACF Analysis")

    from statsmodels.tsa.stattools import acf, pacf

    acf_arma = acf(y_arma, nlags=20)
    pacf_arma = pacf(y_arma, nlags=20)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'PACF'))

    # ACF
    fig.add_trace(go.Bar(x=list(range(len(acf_arma))), y=acf_arma,
                         name='ACF', marker_color='#118AB2'), row=1, col=1)

    # PACF
    fig.add_trace(go.Bar(x=list(range(len(pacf_arma))), y=pacf_arma,
                         name='PACF', marker_color='#EF476F'), row=1, col=2)

    # Confidence intervals
    conf_interval = 1.96 / np.sqrt(n)
    for col in [1, 2]:
        fig.add_hline(y=conf_interval, line_dash="dash", line_color="gray", row=1, col=col)
        fig.add_hline(y=-conf_interval, line_dash="dash", line_color="gray", row=1, col=col)

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Observations**:
    - **ACF**: Tails off gradually (exponential decay)
    - **PACF**: Tails off gradually (exponential decay)
    - Both tail off → Indicates **ARMA model** (not pure AR or MA)

    When both ACF and PACF tail off, we need an ARMA model!
    """)

    st.subheader("Model Selection Summary")

    # Create summary table
    summary_df = pd.DataFrame({
        'Model': ['AR(p)', 'MA(q)', 'ARMA(p,q)'],
        'ACF Pattern': ['Tails off', 'Cuts off after lag q', 'Tails off'],
        'PACF Pattern': ['Cuts off after lag p', 'Tails off', 'Tails off'],
        'How to Identify Order': [
            'Count significant PACF lags',
            'Count significant ACF lags',
            'Use information criteria (AIC, BIC)'
        ]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Information Criteria for ARMA Order Selection

    When both ACF and PACF tail off, we use information criteria to choose p and q:

    **1. Akaike Information Criterion (AIC)**:
    """)
    st.latex(r"AIC = -2\ln(L) + 2k")

    st.markdown("""
    **2. Bayesian Information Criterion (BIC)**:
    """)
    st.latex(r"BIC = -2\ln(L) + k\ln(n)")

    st.markdown("""
    Where:
    - $L$ = Maximum likelihood
    - $k$ = Number of parameters ($p + q + 1$ for ARMA)
    - $n$ = Sample size

    **Decision Rule**: Choose model with **minimum** AIC or BIC

    BIC penalizes complexity more than AIC, often preferring simpler models.
    """)

# ============================================================================
# SECTION 7: ARIMA MODEL
# ============================================================================
elif section == "ARIMA Model":
    st.header("7. Integrated ARMA (ARIMA) Model")

    st.markdown("""
    **ARIMA** extends ARMA to handle **non-stationary** time series by adding 
    **integration** (differencing). It's the most general Box-Jenkins model.
    """)

    st.subheader("Mathematical Formulation")

    st.markdown("""
    An **ARIMA(p,d,q)** model has three components:
    - **p**: Order of AutoRegression (AR)
    - **d**: Degree of differencing (Integration)
    - **q**: Order of Moving Average (MA)

    **Step 1: Differencing** (d times)
    """)

    st.latex(r"W_t = \Delta^d Y_t")

    st.markdown("""
    Where:
    - $\\Delta Y_t = Y_t - Y_{t-1}$ (first difference)
    - $\\Delta^2 Y_t = \\Delta(\\Delta Y_t)$ (second difference)
    - Generally: $\\Delta^d Y_t$ is the $d$-th difference

    **Step 2: Apply ARMA(p,q) to the differenced series**
    """)

    st.latex(r"\phi(B) W_t = \theta(B) \epsilon_t")

    st.markdown("""
    Or equivalently, in full form:
    """)

    st.latex(r"\phi(B) \Delta^d Y_t = \theta(B) \epsilon_t")

    st.markdown("""
    **Expanded form**:
    """)

    st.latex(r"(1 - \phi_1 B - ... - \phi_p B^p)(1-B)^d Y_t = (1 + \theta_1 B + ... + \theta_q B^q)\epsilon_t")

    st.subheader("Common ARIMA Models")

    st.markdown("""
    **ARIMA(0,1,0)**: Random Walk
    """)
    st.latex(r"Y_t = Y_{t-1} + \epsilon_t")
    st.latex(r"\Delta Y_t = \epsilon_t")

    st.markdown("""
    **ARIMA(0,1,0) with drift**: Random Walk with Drift
    """)
    st.latex(r"Y_t = c + Y_{t-1} + \epsilon_t")

    st.markdown("""
    **ARIMA(1,1,0)**: Differenced AR(1)
    """)
    st.latex(r"\Delta Y_t = \phi_1 \Delta Y_{t-1} + \epsilon_t")

    st.markdown("""
    **ARIMA(0,1,1)**: Exponential Smoothing
    """)
    st.latex(r"\Delta Y_t = \epsilon_t + \theta_1 \epsilon_{t-1}")

    st.subheader("Example: ARIMA(1,1,1) Model")

    st.markdown("""
    This model combines:
    1. First differencing (d=1)
    2. AR(1) component
    3. MA(1) component

    **Mathematical form**:
    """)

    st.latex(r"(1 - \phi_1 B)(1 - B) Y_t = (1 + \theta_1 B)\epsilon_t")

    st.markdown("""
    **Expanded**:
    """)

    st.latex(r"Y_t - Y_{t-1} = \phi_1(Y_{t-1} - Y_{t-2}) + \epsilon_t + \theta_1 \epsilon_{t-1}")

    # Interactive ARIMA simulator
    st.markdown("### Interactive ARIMA(1,1,1) Simulator")

    col1, col2, col3 = st.columns(3)
    with col1:
        phi1_arima = st.slider("φ₁ (AR coef)", -0.95, 0.95, 0.5, 0.05)
    with col2:
        theta1_arima = st.slider("θ₁ (MA coef)", -0.95, 0.95, -0.3, 0.05)
    with col3:
        sigma_arima = st.slider("σ (noise)", 0.1, 3.0, 1.0, 0.1)

    drift = st.slider("Drift (constant after differencing)", -0.5, 0.5, 0.1, 0.05)

    # Generate ARIMA(1,1,1)
    np.random.seed(42)
    n = 300

    # Generate differenced series (ARMA(1,1))
    w = np.zeros(n)
    epsilon = np.random.normal(0, sigma_arima, n)

    for t in range(1, n):
        w[t] = drift + phi1_arima * w[t - 1] + epsilon[t] + theta1_arima * epsilon[t - 1]

    # Integrate (cumulative sum) to get original series
    y_arima = np.cumsum(w)

    # Plot original and differenced series
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Original Series Y_t (Non-Stationary)',
                        'Differenced Series ΔY_t (Stationary)'),
        vertical_spacing=0.15
    )

    fig.add_trace(go.Scatter(y=y_arima, mode='lines', name='Y_t',
                             line=dict(color='#E63946', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(y=w, mode='lines', name='ΔY_t',
                             line=dict(color='#06D6A0', width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Notice**:
    - Top plot: Original series shows **trend** (non-stationary)
    - Bottom plot: After differencing, series is **stationary** (fluctuates around mean)
    """)

    # ACF/PACF of differenced series
    st.markdown("### ACF and PACF of Differenced Series")

    from statsmodels.tsa.stattools import acf, pacf

    acf_diff = acf(w, nlags=20)
    pacf_diff = pacf(w, nlags=20)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('ACF of ΔY_t', 'PACF of ΔY_t'))

    fig.add_trace(go.Bar(x=list(range(len(acf_diff))), y=acf_diff,
                         name='ACF', marker_color='#118AB2'), row=1, col=1)

    fig.add_trace(go.Bar(x=list(range(len(pacf_diff))), y=pacf_diff,
                         name='PACF', marker_color='#EF476F'), row=1, col=2)

    conf_interval = 1.96 / np.sqrt(len(w))
    for col in [1, 2]:
        fig.add_hline(y=conf_interval, line_dash="dash", line_color="gray", row=1, col=col)
        fig.add_hline(y=-conf_interval, line_dash="dash", line_color="gray", row=1, col=col)

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="Correlation")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation**:
    - Both ACF and PACF tail off → ARMA structure after differencing
    - This confirms ARIMA(p,1,q) model is appropriate
    """)

    st.subheader("Choosing d: How Many Times to Difference?")

    st.markdown("""
    **Guidelines for choosing d**:

    1. **d = 0**: Series is already stationary

    2. **d = 1**: Most common for economic/business data
        - Series has a trend
        - First difference removes linear trend

    3. **d = 2**: Rare, but needed when:
        - Series has quadratic trend
        - First difference is still non-stationary

    **Warning**: Over-differencing (d too large) can:
    - Create artificial patterns
    - Increase forecast variance
    - Lead to poor models

    **Rule**: Use **minimum d** that makes series stationary

    **Test procedure**:
    """)

    steps_df = pd.DataFrame({
        'Step': [1, 2, 3, 4],
        'Action': [
            'Test original series with ADF test',
            'If p-value > 0.05, difference once (d=1)',
            'Test differenced series with ADF',
            'If still p-value > 0.05, difference again (d=2)'
        ],
        'Decision': [
            'If p < 0.05, d=0 (stationary)',
            'If p < 0.05 now, stop at d=1',
            'If p < 0.05 now, stop at d=1',
            'Rarely need d > 2'
        ]
    })

    st.dataframe(steps_df, use_container_width=True, hide_index=True)

    st.subheader("ARIMA Model Selection Strategy")

    st.markdown("""
    **Complete strategy for identifying ARIMA(p,d,q)**:

    **Step 1: Determine d (order of differencing)**
    - Plot series and look for trend
    - Use ADF test on original series
    - Difference until stationary (typically d=0, 1, or 2)

    **Step 2: Examine ACF and PACF of differenced series**
    - If PACF cuts off at lag p → try AR(p), so ARIMA(p,d,0)
    - If ACF cuts off at lag q → try MA(q), so ARIMA(0,d,q)
    - If both tail off → try ARMA(p,q), so ARIMA(p,d,q)

    **Step 3: Use information criteria**
    - Fit several candidate models
    - Compare AIC and BIC
    - Choose model with lowest AIC/BIC

    **Step 4: Check residuals**
    - Should resemble white noise
    - If not, try different (p,d,q)
    """)

# ============================================================================
# SECTION 8: IDENTIFICATION PROCESS
# ============================================================================
elif section == "Identification Process":
    st.header("8. Model Identification Process")

    st.markdown("""
    This is the most crucial step in Box-Jenkins methodology. We'll go through 
    each sub-step in detail with mathematical foundations.
    """)

    st.subheader("Step 1: Test for Stationarity")

    st.markdown("""
    ### 1.1 Visual Inspection

    Plot the time series and look for:
    - **Trend**: Increasing or decreasing pattern
    - **Changing variance**: Spread of values changes over time
    - **Level shifts**: Sudden jumps in the mean

    ### 1.2 Augmented Dickey-Fuller (ADF) Test

    **Test equation** (with drift and trend):
    """)

    st.latex(r"\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \epsilon_t")

    st.markdown("""
    **Hypotheses**:
    - $H_0$: $\\gamma = 0$ (unit root exists, non-stationary)
    - $H_1$: $\\gamma < 0$ (no unit root, stationary)

    **Test statistic**:
    """)

    st.latex(r"ADF = \frac{\hat{\gamma}}{SE(\hat{\gamma})}")

    st.markdown("""
    **Critical values** (at 5% significance):
    - No constant, no trend: -1.95
    - Constant, no trend: -2.86
    - Constant and trend: -3.41

    **Decision**: 
    - If ADF < critical value (or p-value < 0.05) → Reject $H_0$ → Stationary
    - If ADF ≥ critical value (or p-value ≥ 0.05) → Fail to reject $H_0$ → Non-stationary

    ### 1.3 KPSS Test (Confirmation)

    **Test statistic**:
    """)

    st.latex(r"KPSS = \frac{1}{T^2} \sum_{t=1}^{T} \frac{S_t^2}{\hat{\sigma}^2}")

    st.markdown("""
    Where $S_t = \\sum_{i=1}^{t} e_i$ (cumulative sum of residuals)

    **Hypotheses** (opposite of ADF!):
    - $H_0$: Series is stationary
    - $H_1$: Series has unit root

    **Decision**:
    - If p-value > 0.05 → Stationary
    - If p-value < 0.05 → Non-stationary

    **Recommendation**: Use both ADF and KPSS for robust conclusion.
    """)

    # Interactive stationarity testing
    st.markdown("### Interactive Stationarity Testing")

    series_type = st.radio(
        "Choose series type:",
        ["Stationary (White Noise)", "Non-Stationary (Random Walk)",
         "Non-Stationary (Trend)", "Non-Stationary (Trend + Random Walk)"]
    )

    np.random.seed(42)
    n = 300

    if series_type == "Stationary (White Noise)":
        test_series = np.random.randn(n)
        description = "Pure white noise (stationary)"
    elif series_type == "Non-Stationary (Random Walk)":
        test_series = np.cumsum(np.random.randn(n))
        description = "Random walk (non-stationary, d=1 needed)"
    elif series_type == "Non-Stationary (Trend)":
        test_series = 0.05 * np.arange(n) + np.random.randn(n)
        description = "Linear trend (non-stationary, d=1 needed)"
    else:
        test_series = np.cumsum(np.random.randn(n)) + 0.05 * np.arange(n)
        description = "Trend + Random walk (non-stationary, d=1 or d=2 needed)"

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=test_series, mode='lines',
                             line=dict(color='#2E86AB', width=2)))
    fig.update_layout(title=f"Series: {description}",
                      xaxis_title="Time", yaxis_title="Value", height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Tests
    adf_result = adfuller(test_series, regression='ct')
    kpss_result = kpss(test_series, regression='ct')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ADF Test Results")
        st.metric("Test Statistic", f"{adf_result[0]:.4f}")
        st.metric("P-value", f"{adf_result[1]:.4f}")
        st.metric("Critical Value (5%)", f"{adf_result[4]['5%']:.4f}")

        if adf_result[1] < 0.05:
            st.success("✅ Reject H₀: Series is STATIONARY")
        else:
            st.error("❌ Fail to reject H₀: Series is NON-STATIONARY")

    with col2:
        st.markdown("#### KPSS Test Results")
        st.metric("Test Statistic", f"{kpss_result[0]:.4f}")
        st.metric("P-value", f"{kpss_result[1]:.4f}")
        st.metric("Critical Value (5%)", f"{kpss_result[3]['5%']:.4f}")

        if kpss_result[1] > 0.05:
            st.success("✅ Fail to reject H₀: Series is STATIONARY")
        else:
            st.error("❌ Reject H₀: Series is NON-STATIONARY")

    # Conclusion
    st.markdown("#### Combined Conclusion")
    adf_stationary = adf_result[1] < 0.05
    kpss_stationary = kpss_result[1] > 0.05

    if adf_stationary and kpss_stationary:
        st.success("✅ **Both tests agree**: Series is STATIONARY (d=0)")
    elif not adf_stationary and not kpss_stationary:
        st.error("❌ **Both tests agree**: Series is NON-STATIONARY (need differencing)")
        st.info("Recommendation: Try d=1 (first difference)")
    else:
        st.warning("⚠️ **Tests disagree**: Results are inconclusive")
        st.info("Recommendation: Try differencing and test again")

    st.subheader("Step 2: Autocorrelation Analysis")

    st.markdown("""
    ### 2.1 Autocorrelation Function (ACF)

    The ACF measures correlation between $Y_t$ and $Y_{t-k}$:
    """)

    st.latex(r"\rho_k = \frac{Cov(Y_t, Y_{t-k})}{Var(Y_t)} = \frac{\gamma_k}{\gamma_0}")

    st.markdown("""
    **Sample ACF**:
    """)

    st.latex(
        r"\hat{\rho}_k = \frac{\sum_{t=k+1}^{n}(Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum_{t=1}^{n}(Y_t - \bar{Y})^2}")

    st.markdown("""
    **95% Confidence Interval**:
    """)

    st.latex(r"\pm \frac{1.96}{\sqrt{n}}")

    st.markdown("""
    **Interpretation**:
    - Spikes outside confidence bands → Significant correlation at that lag
    - Pattern reveals model structure

    ### 2.2 Partial Autocorrelation Function (PACF)

    PACF measures correlation between $Y_t$ and $Y_{t-k}$ after removing effects of lags 1 to k-1.

    **Definition**: PACF at lag k is the coefficient $\\phi_{kk}$ from regression:
    """)

    st.latex(r"Y_t = \phi_{k1} Y_{t-1} + \phi_{k2} Y_{t-2} + ... + \phi_{kk} Y_{t-k} + \epsilon_t")

    st.markdown("""
    **95% Confidence Interval** (same as ACF):
    """)

    st.latex(r"\pm \frac{1.96}{\sqrt{n}}")

    st.subheader("Step 3: Model Identification Rules")

    # Create comprehensive identification table
    identification_df = pd.DataFrame({
        'Model Type': ['AR(p)', 'MA(q)', 'ARMA(p,q)', 'ARIMA(p,d,q)'],
        'ACF Pattern': [
            'Tails off exponentially or with damped sine wave',
            'Cuts off after lag q (zero for k > q)',
            'Tails off gradually',
            'After differencing: depends on (p,q)'
        ],
        'PACF Pattern': [
            'Cuts off after lag p (zero for k > p)',
            'Tails off exponentially or with damped sine wave',
            'Tails off gradually',
            'After differencing: depends on (p,q)'
        ],
        'How to Identify': [
            'Count significant PACF lags',
            'Count significant ACF lags',
            'Use AIC/BIC to test different (p,q)',
            'First find d, then apply AR/MA/ARMA rules'
        ]
    })

    st.dataframe(identification_df, use_container_width=True, hide_index=True)

    st.subheader("Step 4: Information Criteria")

    st.markdown("""
    When ACF/PACF patterns are unclear, use information criteria to compare models:

    ### 4.1 Akaike Information Criterion (AIC)
    """)

    st.latex(r"AIC = -2\ln(L) + 2k")

    st.markdown("""
    ### 4.2 Corrected AIC (AICc) for small samples
    """)

    st.latex(r"AICc = AIC + \frac{2k(k+1)}{n-k-1}")

    st.markdown("""
    Use AICc when $n/k < 40$

    ### 4.3 Bayesian Information Criterion (BIC)
    """)

    st.latex(r"BIC = -2\ln(L) + k\ln(n)")

    st.markdown("""
    Where:
    - $L$ = Likelihood of the model
    - $k$ = Number of parameters
    - $n$ = Sample size

    **For ARIMA(p,d,q)**:
    """)

    st.latex(r"k = p + q + 1 \text{ (if constant included)}")

    st.markdown("""
    **Decision rule**: Choose model with **minimum** AIC or BIC

    **BIC vs AIC**:
    - BIC penalizes complexity more heavily (has $\\ln(n)$ term)
    - BIC tends to choose simpler models
    - AIC better for prediction
    - BIC better for model selection

    **Practical approach**:
    1. Identify candidate models using ACF/PACF
    2. Fit all candidates
    3. Compare AIC and BIC
    4. Choose model with lowest values
    5. Verify with diagnostic checks (next section)
    """)

    # Example model comparison
    st.markdown("### Example: Model Comparison")

    st.markdown("""
    Suppose we have these candidate models for a differenced series (d=1):
    """)

    comparison_df = pd.DataFrame({
        'Model': ['ARIMA(1,1,0)', 'ARIMA(0,1,1)', 'ARIMA(1,1,1)',
                  'ARIMA(2,1,0)', 'ARIMA(0,1,2)', 'ARIMA(2,1,1)'],
        'Parameters (k)': [2, 2, 3, 3, 3, 4],
        'AIC': [245.3, 243.1, 241.8, 244.2, 242.9, 243.5],
        'BIC': [252.1, 249.9, 251.4, 253.8, 252.5, 256.0],
        'Selected?': ['', '✓ (Lowest AIC & BIC)', '', '', '', '']
    })


    # Highlight best model
    def highlight_best(s):
        if s['Selected?'] == '✓ (Lowest AIC & BIC)':
            return ['background-color: #90EE90'] * len(s)
        return [''] * len(s)


    styled_df = comparison_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("""
    In this example, **ARIMA(0,1,1)** has:
    - Lowest AIC (243.1)
    - Lowest BIC (249.9)
    - Fewest parameters (k=2)

    → Best choice by parsimony principle!
    """)

# ============================================================================
# SECTION 9: COMPLETE EXAMPLE
# ============================================================================
elif section == "Complete Example":
    st.header("9. Complete Box-Jenkins Example")

    st.markdown("""
    Let's walk through the entire Box-Jenkins methodology with a complete example, 
    step by step.
    """)

    st.subheader("Dataset: Simulated Monthly Sales Data")

    st.markdown("""
    We'll analyze 5 years of monthly sales data (60 observations) with:
    - Upward trend
    - Some seasonal pattern
    - Random fluctuations
    """)

    # Generate realistic sales data
    np.random.seed(123)
    n = 60
    t = np.arange(n)

    # Components
    trend = 100 + 2 * t
    seasonal = 15 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)
    sales = trend + seasonal + noise

    dates = pd.date_range(start='2020-01-01', periods=n, freq='M')
    df = pd.DataFrame({'Date': dates, 'Sales': sales})

    # Plot original data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Sales'], mode='lines+markers',
                             name='Sales', line=dict(color='#2E86AB', width=2),
                             marker=dict(size=6)))
    fig.update_layout(title="Original Sales Data",
                      xaxis_title="Date", yaxis_title="Sales", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## STEP 1: Identification")

    st.markdown("### 1.1 Check Stationarity")

    # ADF test
    adf_original = adfuller(sales, regression='ct')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Visual Assessment")
        st.write("✗ Clear upward trend visible")
        st.write("✗ Mean is not constant")
        st.write("→ Series appears non-stationary")

    with col2:
        st.markdown("#### ADF Test on Original Series")
        st.metric("ADF Statistic", f"{adf_original[0]:.4f}")
        st.metric("P-value", f"{adf_original[1]:.4f}")
        if adf_original[1] > 0.05:
            st.error("❌ Non-stationary (p > 0.05)")
            st.write("**Conclusion**: Need differencing (d ≥ 1)")

    st.markdown("### 1.2 Apply First Differencing")

    diff1 = np.diff(sales)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Original Series', 'First Difference'),
                        vertical_spacing=0.15)

    fig.add_trace(go.Scatter(x=df['Date'], y=sales, mode='lines',
                             line=dict(color='#E63946', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['Date'][1:], y=diff1, mode='lines',
                             line=dict(color='#06D6A0', width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sales", row=1, col=1)
    fig.update_yaxes(title_text="Change in Sales", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Test differenced series
    adf_diff = adfuller(diff1, regression='c')

    st.markdown("#### ADF Test on Differenced Series")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ADF Statistic", f"{adf_diff[0]:.4f}")
    with col2:
        st.metric("P-value", f"{adf_diff[1]:.6f}")

    if adf_diff[1] < 0.05:
        st.success("✅ Differenced series is stationary (p < 0.05)")
        st.write("**Conclusion**: d = 1 is sufficient")

    st.markdown("### 1.3 ACF and PACF Analysis")

    from statsmodels.tsa.stattools import acf, pacf

    acf_vals = acf(diff1, nlags=20)
    pacf_vals = pacf(diff1, nlags=20)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF of Differenced Series',
                                                        'PACF of Differenced Series'))

    # ACF
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals,
                         marker_color='#118AB2'), row=1, col=1)

    # PACF
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals,
                         marker_color='#EF476F'), row=1, col=2)

    # Confidence intervals
    conf = 1.96 / np.sqrt(len(diff1))
    for col in [1, 2]:
        fig.add_hline(y=conf, line_dash="dash", line_color="gray", row=1, col=col)
        fig.add_hline(y=-conf, line_dash="dash", line_color="gray", row=1, col=col)

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="Correlation")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    #### Interpretation:

    **ACF Pattern**:
    - Significant spike at lag 1
    - Quickly decays to insignificance
    - Suggests MA(1) component

    **PACF Pattern**:
    - Significant spike at lag 1
    - Some significance at later lags
    - Suggests AR(1) component

    **Preliminary Models to Consider**:
    - ARIMA(1,1,0): Pure AR after differencing
    - ARIMA(0,1,1): Pure MA after differencing
    - ARIMA(1,1,1): Mixed ARMA after differencing
    """)

    st.markdown("---")
    st.markdown("## STEP 2: Estimation")

    st.markdown("""
    Fit the candidate models and compare using AIC/BIC:
    """)

    # Fit models
    models_to_fit = [
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 0),
        (0, 1, 2)
    ]

    results = []
    fitted_models = {}

    for order in models_to_fit:
        try:
            model = ARIMA(sales, order=order)
            fitted = model.fit()
            fitted_models[order] = fitted
            results.append({
                'Model': f'ARIMA{order}',
                'AIC': fitted.aic,
                'BIC': fitted.bic,
                'Parameters': order[0] + order[2] + 1
            })
        except:
            pass

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AIC')

    # Highlight best model
    best_idx = results_df['AIC'].idxmin()


    def highlight_best_model(row):
        if row.name == best_idx:
            return ['background-color: #90EE90'] * len(row)
        return [''] * len(row)


    styled_results = results_df.style.apply(highlight_best_model, axis=1)
    st.dataframe(styled_results, use_container_width=True, hide_index=True)

    best_model_order = tuple(map(int, results_df.loc[best_idx, 'Model'].replace('ARIMA', '').strip('()').split(',')))
    best_model = fitted_models[best_model_order]

    st.success(f"✅ Best Model Selected: ARIMA{best_model_order}")

    st.markdown("### Model Parameters")

    st.write(best_model.summary().tables[1])

    st.markdown("""
    **Parameter Interpretation**:
    - All coefficients should be statistically significant (p < 0.05)
    - Standard errors should be relatively small
    """)

    st.markdown("---")
    st.markdown("## STEP 3: Diagnostic Checking")

    st.markdown("### 3.1 Residual Analysis")

    residuals = best_model.resid

    # Plot residuals
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Residuals Over Time',
                                        'Residual Distribution',
                                        'ACF of Residuals',
                                        'Q-Q Plot'),
                        specs=[[{"type": "scatter"}, {"type": "histogram"}],
                               [{"type": "bar"}, {"type": "scatter"}]])

    # Residuals over time
    fig.add_trace(go.Scatter(y=residuals, mode='lines', line=dict(color='#2E86AB')),
                  row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(x=residuals, nbinsx=20, marker_color='#06D6A0'),
                  row=1, col=2)

    # ACF of residuals
    acf_resid = acf(residuals, nlags=20)
    fig.add_trace(go.Bar(x=list(range(len(acf_resid))), y=acf_resid,
                         marker_color='#118AB2'), row=2, col=1)
    conf = 1.96 / np.sqrt(len(residuals))
    fig.add_hline(y=conf, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="gray", row=2, col=1)

    # Q-Q plot
    from scipy import stats

    qq = stats.probplot(residuals, dist="norm")
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                             marker=dict(color='#EF476F')), row=2, col=2)
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1],
                             mode='lines', line=dict(color='red', dash='dash')),
                  row=2, col=2)

    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="ACF", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    #### Diagnostic Checks:

    **1. Residuals Over Time**:
    - ✓ Should fluctuate randomly around zero
    - ✓ No apparent patterns or trends
    - ✓ Constant variance (homoscedastic)

    **2. Residual Distribution**:
    - ✓ Should be approximately normal
    - ✓ Centered around zero
    - ✓ No heavy tails or skewness

    **3. ACF of Residuals**:
    - ✓ All lags should be within confidence bands
    - ✓ Indicates white noise (no autocorrelation left)

    **4. Q-Q Plot**:
    - ✓ Points should lie close to diagonal line
    - ✓ Confirms normality assumption
    """)

    st.markdown("### 3.2 Statistical Tests")

    # Ljung-Box test
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)

    st.markdown("#### Ljung-Box Test for Residual Autocorrelation")

    st.markdown("""
    **Null Hypothesis**: Residuals are independently distributed (white noise)

    **Test Statistic**:
    """)

    st.latex(r"Q = n(n+2)\sum_{k=1}^{h}\frac{\hat{\rho}_k^2}{n-k}")

    st.markdown("""
    Where:
    - $n$ = sample size
    - $h$ = number of lags tested
    - $\\hat{\\rho}_k$ = sample autocorrelation at lag $k$
    """)

    st.dataframe(lb_test.head(10), use_container_width=True)

    if (lb_test['lb_pvalue'] > 0.05).all():
        st.success("✅ All p-values > 0.05: Residuals are white noise!")
    else:
        st.warning("⚠️ Some p-values < 0.05: May indicate model inadequacy")

    # Shapiro-Wilk test for normality
    from scipy.stats import shapiro

    shapiro_stat, shapiro_p = shapiro(residuals)

    st.markdown("#### Shapiro-Wilk Test for Normality")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Statistic", f"{shapiro_stat:.4f}")
    with col2:
        st.metric("P-value", f"{shapiro_p:.4f}")

    if shapiro_p > 0.05:
        st.success("✅ p-value > 0.05: Residuals are normally distributed")
    else:
        st.warning("⚠️ p-value < 0.05: Residuals may not be normal")

    st.markdown("---")
    st.markdown("## STEP 4: Forecasting")

    st.markdown("### Generate Forecasts")

    forecast_steps = st.slider("Number of steps to forecast:", 1, 24, 12)

    # Make forecast
    forecast = best_model.forecast(steps=forecast_steps)
    forecast_df = best_model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast_df.conf_int()

    # Create forecast dates
    last_date = df['Date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                   periods=forecast_steps, freq='M')

    # Plot
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=df['Date'], y=sales, mode='lines+markers',
                             name='Historical', line=dict(color='#2E86AB', width=2)))

    # Forecast
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines+markers',
                             name='Forecast', line=dict(color='#E63946', width=2, dash='dash')))

    # Confidence interval
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_ci.iloc[:, 0],
                             mode='lines', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_ci.iloc[:, 1],
                             mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(230, 57, 70, 0.2)',
                             name='95% CI'))

    fig.update_layout(title=f"Sales Forecast using ARIMA{best_model_order}",
                      xaxis_title="Date", yaxis_title="Sales", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown("### Forecast Values")

    forecast_table = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast,
        'Lower 95% CI': forecast_ci.iloc[:, 0],
        'Upper 95% CI': forecast_ci.iloc[:, 1]
    })

    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Forecast Interpretation

    **Point Forecast**: 
    - Best estimate of future values
    - Based on model parameters and historical patterns

    **Confidence Interval**:
    - 95% probability true value falls within this range
    - Widens as we forecast further into future
    - Reflects forecast uncertainty

    **Forecast Uncertainty** increases with:
    - Longer forecast horizon
    - Model parameter uncertainty
    - Random error variance
    """)

    st.markdown("---")
    st.markdown("## Summary of Analysis")

    st.markdown(f"""
    ### Complete Box-Jenkins Process Applied:

    **1. Identification**:
    - ✅ Tested for stationarity (ADF test)
    - ✅ Applied first differencing (d=1)
    - ✅ Analyzed ACF and PACF
    - ✅ Identified candidate models

    **2. Estimation**:
    - ✅ Fit {len(models_to_fit)} candidate models
    - ✅ Compared AIC and BIC
    - ✅ Selected best model: **ARIMA{best_model_order}**

    **3. Diagnostic Checking**:
    - ✅ Residuals appear as white noise
    - ✅ Passed Ljung-Box test
    - ✅ Residuals approximately normal
    - ✅ Model is adequate

    **4. Forecasting**:
    - ✅ Generated {forecast_steps}-step ahead forecast
    - ✅ Provided 95% confidence intervals
    - ✅ Forecast shows continuation of trend

    **Final Model**: ARIMA{best_model_order} is suitable for forecasting this sales data.
    """)

# ============================================================================
# SECTION 10: MODEL DIAGNOSTICS
# ============================================================================
else:  # Model Diagnostics
    st.header("10. Model Diagnostics in Detail")

    st.markdown("""
    After estimating a model, we must verify it's adequate. This section covers 
    all diagnostic checks in detail.
    """)

    st.subheader("Why Diagnostics Matter")

    st.markdown("""
    A model might have good fit statistics (low AIC/BIC) but still be inadequate if:
    - Residuals show patterns (not white noise)
    - Assumptions are violated
    - Model is over-fitted or under-fitted

    **Goal of diagnostics**: Verify the model has captured all systematic patterns, 
    leaving only random noise in residuals.
    """)

    st.subheader("1. Residual Analysis")

    st.markdown("""
    ### 1.1 What are Residuals?

    Residuals are the difference between observed and fitted values:
    """)

    st.latex(r"\hat{\epsilon}_t = Y_t - \hat{Y}_t")

    st.markdown("""
    For a good model, residuals should behave like **white noise**:
    """)

    st.latex(r"\epsilon_t \sim WN(0, \sigma^2)")

    st.markdown("""
    **White noise properties**:
    1. Mean = 0: $E[\\epsilon_t] = 0$
    2. Constant variance: $Var(\\epsilon_t) = \\sigma^2$
    3. No autocorrelation: $Cov(\\epsilon_t, \\epsilon_{t-k}) = 0$ for all $k \\neq 0$
    4. Normally distributed: $\\epsilon_t \\sim N(0, \\sigma^2)$ (for inference)
    """)

    # Generate example good and bad residuals
    st.markdown("### 1.2 Good vs Bad Residuals")

    np.random.seed(42)
    n = 200

    good_residuals = np.random.randn(n)
    bad_residuals = np.random.randn(n) + 0.3 * np.sin(np.linspace(0, 4 * np.pi, n))

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Good Residuals (White Noise)',
                                        'Bad Residuals (Pattern Present)'))

    fig.add_trace(go.Scatter(y=good_residuals, mode='lines',
                             line=dict(color='#06D6A0', width=1)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Scatter(y=bad_residuals, mode='lines',
                             line=dict(color='#EF476F', width=1)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Residual")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Left (Good)**:
    - Random fluctuations around zero
    - No visible patterns
    - Constant spread

    **Right (Bad)**:
    - Clear cyclical pattern
    - Model hasn't captured all information
    - Need to revise model
    """)

    st.subheader("2. ACF of Residuals")

    st.markdown("""
    ### 2.1 Theory

    If model is adequate, residual ACF should show **no significant autocorrelation**:
    """)

    st.latex(r"\rho_k(\hat{\epsilon}) \approx 0 \text{ for all } k > 0")

    st.markdown("""
    **95% confidence bands**:
    """)

    st.latex(r"\pm \frac{1.96}{\sqrt{n}}")

    st.markdown("""
    **Interpretation**:
    - If lags fall within bands → No autocorrelation (good!)
    - If lags outside bands → Autocorrelation present (bad!)
    """)

    # Plot ACF comparison
    from statsmodels.tsa.stattools import acf

    acf_good = acf(good_residuals, nlags=20)
    acf_bad = acf(bad_residuals, nlags=20)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('ACF: Good Residuals',
                                        'ACF: Bad Residuals'))

    fig.add_trace(go.Bar(x=list(range(len(acf_good))), y=acf_good,
                         marker_color='#118AB2'), row=1, col=1)

    fig.add_trace(go.Bar(x=list(range(len(acf_bad))), y=acf_bad,
                         marker_color='#EF476F'), row=1, col=2)

    conf = 1.96 / np.sqrt(n)
    for col in [1, 2]:
        fig.add_hline(y=conf, line_dash="dash", line_color="gray", row=1, col=col)
        fig.add_hline(y=-conf, line_dash="dash", line_color="gray", row=1, col=col)

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="ACF")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. Ljung-Box Test")

    st.markdown("""
    ### 3.1 Mathematical Formulation

    The Ljung-Box test formally tests if residuals are white noise.

    **Test Statistic**:
    """)

    st.latex(r"Q_{LB} = n(n+2)\sum_{k=1}^{h}\frac{\hat{\rho}_k^2}{n-k}")

    st.markdown("""
    Where:
    - $n$ = sample size
    - $h$ = number of lags tested (typically 10 or 20)
    - $\\hat{\\rho}_k$ = sample ACF of residuals at lag $k$

    **Distribution**: Under $H_0$, $Q_{LB} \\sim \\chi^2_{h-p-q}$

    **Hypotheses**:
    - $H_0$: Residuals are independently distributed (no autocorrelation)
    - $H_1$: Residuals are autocorrelated

    **Decision Rule**:
    - If p-value > 0.05 → Fail to reject $H_0$ → Residuals are white noise ✓
    - If p-value < 0.05 → Reject $H_0$ → Residuals show autocorrelation ✗

    **Rule of Thumb**: Test at lags $h = \\min(10, n/5)$ or $h = \\min(20, n/5)$
    """)

    # Perform test on examples
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_good = acorr_ljungbox(good_residuals, lags=10, return_df=True)
    lb_bad = acorr_ljungbox(bad_residuals, lags=10, return_df=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Good Residuals")
        st.dataframe(lb_good[['lb_stat', 'lb_pvalue']].head(), use_container_width=True)
        if (lb_good['lb_pvalue'] > 0.05).all():
            st.success("✅ All p > 0.05: White noise confirmed")

    with col2:
        st.markdown("#### Bad Residuals")
        st.dataframe(lb_bad[['lb_stat', 'lb_pvalue']].head(), use_container_width=True)
        if not (lb_bad['lb_pvalue'] > 0.05).all():
            st.error("❌ Some p < 0.05: Autocorrelation detected")

    st.subheader("4. Normality Tests")

    st.markdown("""
    ### 4.1 Why Test Normality?

    Many ARIMA inference procedures (confidence intervals, hypothesis tests) assume:
    """)

    st.latex(r"\epsilon_t \sim N(0, \sigma^2)")

    st.markdown("""
    While forecasts don't require normality, **statistical inference** does.

    ### 4.2 Q-Q Plot

    **Quantile-Quantile plot** compares residual quantiles to theoretical normal quantiles.

    **Interpretation**:
    - Points on diagonal line → Normal distribution
    - S-shaped curve → Skewed distribution
    - Points far from line at tails → Heavy or light tails
    """)

    # Generate non-normal residuals for comparison
    skewed_residuals = np.random.gamma(2, 2, n) - 4
    heavy_tail_residuals = np.random.standard_t(3, n)

    from scipy import stats

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=('Normal', 'Skewed', 'Heavy Tails'))

    for i, (data, name) in enumerate([(good_residuals, 'Normal'),
                                      (skewed_residuals, 'Skewed'),
                                      (heavy_tail_residuals, 'Heavy')], 1):
        qq = stats.probplot(data, dist="norm")
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                                 marker=dict(size=4, color='#2E86AB')), row=1, col=i)
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1],
                                 mode='lines', line=dict(color='red', dash='dash')),
                      row=1, col=i)

    fig.update_layout(height=400, showlegend=False)
    for i in range(1, 4):
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=i)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=i)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 4.3 Shapiro-Wilk Test

    **Test Statistic**: Measures how well data fits normal distribution
    """)

    st.latex(r"W = \frac{\left(\sum_{i=1}^{n}a_i x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}")

    st.markdown("""
    Where $x_{(i)}$ are ordered residuals and $a_i$ are weights.

    **Hypotheses**:
    - $H_0$: Data are normally distributed
    - $H_1$: Data are not normally distributed

    **Decision**: p-value > 0.05 → Accept normality
    """)

    # Tests
    from scipy.stats import shapiro

    sw_good = shapiro(good_residuals)
    sw_skewed = shapiro(skewed_residuals)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Normal Residuals")
        st.metric("Statistic", f"{sw_good[0]:.4f}")
        st.metric("P-value", f"{sw_good[1]:.4f}")
        if sw_good[1] > 0.05:
            st.success("✅ p > 0.05: Normal")

    with col2:
        st.markdown("#### Skewed Residuals")
        st.metric("Statistic", f"{sw_skewed[0]:.4f}")
        st.metric("P-value", f"{sw_skewed[1]:.6f}")
        if sw_skewed[1] < 0.05:
            st.error("❌ p < 0.05: Not normal")

    st.subheader("5. Heteroscedasticity Tests")

    st.markdown("""
    ### 5.1 What is Heteroscedasticity?

    **Heteroscedasticity** = Non-constant variance of residuals

    **Homoscedastic** (good):
    """)
    st.latex(r"Var(\epsilon_t) = \sigma^2 \text{ for all } t")

    st.markdown("""
    **Heteroscedastic** (bad):
    """)
    st.latex(r"Var(\epsilon_t) = \sigma_t^2 \text{ (varies with } t\text{)}")

    st.markdown("""
    ### 5.2 Visual Detection

    Plot residuals vs fitted values or time. Look for:
    - Funnel shape (increasing/decreasing spread)
    - Clustering of variance in certain periods
    """)

    # Example
    hetero_residuals = good_residuals * (1 + 0.5 * np.linspace(0, 1, n))

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Homoscedastic', 'Heteroscedastic'))

    fig.add_trace(go.Scatter(y=good_residuals, mode='markers',
                             marker=dict(size=4, color='#06D6A0')), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Scatter(y=hetero_residuals, mode='markers',
                             marker=dict(size=4, color='#EF476F')), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Residual")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("6. Diagnostic Checklist")

    st.markdown("""
    ### Complete Checklist for Model Adequacy

    | Check | Method | What to Look For | Action if Failed |
    |-------|--------|-----------------|------------------|
    | **No Autocorrelation** | ACF plot | All lags within confidence bands | Increase AR or MA order |
    | **White Noise** | Ljung-Box | All p-values > 0.05 | Revise model specification |
    | **Normality** | Q-Q plot | Points on diagonal | Usually OK if minor violation |
    | **Normality** | Shapiro-Wilk | p-value > 0.05 | Consider transformation |
    | **Constant Variance** | Residual plot | Even spread over time | Consider ARCH/GARCH models |
    | **No Outliers** | Standardized residuals | All within ±3 standard deviations | Investigate data points |
    | **Model Stability** | Recursive residuals | Stable over time | Check for structural breaks |

    ### What to Do When Diagnostics Fail

    **Problem: Residuals show autocorrelation**
    - Solution: Increase p or q
    - Try different model orders
    - Check if differencing is adequate

    **Problem: Non-normal residuals**
    - Minor deviations: Usually OK for forecasting
    - Severe skewness: Try Box-Cox transformation
    - Heavy tails: Consider robust methods

    **Problem: Heteroscedasticity**
    - Solution: Use ARCH/GARCH models
    - Apply variance stabilizing transformation
    - Use weighted least squares

    **Problem: All checks fail**
    - Return to identification step
    - Consider different model class
    - Check for structural breaks or outliers
    """)

    st.markdown("---")
    st.markdown("""
    ### Final Note on Diagnostics

    Remember:
    1. **Perfect fit is rare** in real data
    2. **Minor violations** are often acceptable
    3. **Focus on residual patterns** more than individual tests
    4. **Iterate** until diagnostics are satisfactory
    5. **Forecasting** is often robust to minor assumption violations

    The goal is a model that:
    - Captures systematic patterns
    - Leaves random residuals
    - Provides reliable forecasts
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📚 Complete Guide to Box-Jenkins Models</p>
    <p>Navigate through sections using the sidebar to explore each topic in detail</p>
</div>
""", unsafe_allow_html=True)