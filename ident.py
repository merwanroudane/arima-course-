import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="ARMA & ARIMA Model Identification Guide",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2ca02c;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Complete Guide to ARMA & ARIMA Model Identification</p>',
            unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📑 Navigation")
section = st.sidebar.radio(
  "Choose Section:",
  ["🏠 Introduction",
   "📈 ACF - Autocorrelation Function",
   "📉 PACF - Partial Autocorrelation Function",
   "🔬 Statistical Tests",
   "🎯 ARIMA Order Selection",
   "💡 Practical Cases & Examples",
   "🧪 Interactive Simulation"]
)

# ============================================================================
# SECTION 1: INTRODUCTION
# ============================================================================
if section == "🏠 Introduction":
  st.markdown('<p class="section-header">Introduction to Time Series Analysis</p>',
              unsafe_allow_html=True)

  st.markdown("""
    ### What is Time Series Analysis?

    Time series analysis involves analyzing data points collected over time to understand patterns, 
    trends, and make forecasts. ARMA and ARIMA models are powerful tools for this purpose.
    """)

  col1, col2 = st.columns(2)

  with col1:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
        **ARMA Model (AutoRegressive Moving Average)**

        Combines two components:
        - **AR (p)**: AutoRegressive - uses past values
        - **MA (q)**: Moving Average - uses past errors

        **Formula:**
        """)
    st.latex(
      r"y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}")
    st.markdown('</div>', unsafe_allow_html=True)

  with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
        **ARIMA Model (AutoRegressive Integrated Moving Average)**

        Extends ARMA by adding:
        - **I (d)**: Integration - differencing to achieve stationarity

        **ARIMA(p, d, q)** where:
        - p = AR order
        - d = degree of differencing
        - q = MA order
        """)
    st.latex(
      r"\Delta^d y_t = c + \phi_1 \Delta^d y_{t-1} + \cdots + \phi_p \Delta^d y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}")
    st.markdown('</div>', unsafe_allow_html=True)

  st.markdown("""
    ### Key Concepts

    #### 1. **Stationarity**
    A time series is stationary if its statistical properties (mean, variance) don't change over time.
    """)

  st.latex(r"""
    \begin{aligned}
    &\text{Constant Mean: } E[y_t] = \mu \text{ for all } t \\
    &\text{Constant Variance: } Var[y_t] = \sigma^2 \text{ for all } t \\
    &\text{Autocovariance depends only on lag: } Cov[y_t, y_{t-k}] = \gamma_k
    \end{aligned}
    """)

  st.markdown("""
    #### 2. **White Noise**
    A sequence of random variables with:
    - Zero mean: E[εₜ] = 0
    - Constant variance: Var[εₜ] = σ²
    - No autocorrelation: Cov[εₜ, εₛ] = 0 for t ≠ s
    """)

  st.markdown('<div class="warning-box">', unsafe_allow_html=True)
  st.markdown("""
    **Why Stationarity Matters:**
    - ARMA models require stationary data
    - Non-stationary data can lead to spurious correlations
    - Differencing (I component in ARIMA) transforms non-stationary data to stationary
    """)
  st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# SECTION 2: ACF - AUTOCORRELATION FUNCTION
# ============================================================================
elif section == "📈 ACF - Autocorrelation Function":
  st.markdown('<p class="section-header">ACF - Autocorrelation Function</p>',
              unsafe_allow_html=True)

  st.markdown("""
    ### What is ACF?

    The Autocorrelation Function (ACF) measures the correlation between a time series and its lagged values.
    It helps identify the moving average (MA) component of the model.
    """)

  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("**Mathematical Definition:**")
  st.latex(r"""
    \rho_k = \frac{Cov(y_t, y_{t-k})}{\sqrt{Var(y_t) \cdot Var(y_{t-k})}} = \frac{\gamma_k}{\gamma_0}
    """)
  st.markdown("where:")
  st.latex(r"""
    \begin{aligned}
    &\gamma_k = E[(y_t - \mu)(y_{t-k} - \mu)] \text{ (autocovariance at lag k)} \\
    &\gamma_0 = Var(y_t) \text{ (variance)} \\
    &\rho_k \in [-1, 1] \text{ (correlation coefficient)}
    \end{aligned}
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  st.markdown("### Sample ACF Estimation")
  st.latex(r"""
    \hat{\rho}_k = \frac{\sum_{t=k+1}^{n}(y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{n}(y_t - \bar{y})^2}
    """)

  st.markdown("""
    ### Interpretation of ACF Patterns

    The ACF pattern helps identify the MA(q) order:
    """)

  col1, col2, col3 = st.columns(3)

  with col1:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **MA(q) Process**

        - **Sharp cutoff** after lag q
        - ACF = 0 for lags > q
        - Significant spikes up to lag q

        Example: MA(2) has non-zero ACF only at lags 1 and 2
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  with col2:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **AR(p) Process**

        - **Gradual decay** (exponential or sinusoidal)
        - Does NOT cut off sharply
        - Tails off slowly

        ACF alone cannot determine AR order
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  with col3:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **ARMA(p,q) Process**

        - Mixed pattern
        - Tails off gradually
        - May show both decay and oscillation

        Need PACF for complete identification
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  st.markdown("### Confidence Intervals")
  st.markdown("""
    ACF values within the confidence bounds suggest no significant autocorrelation.
    """)
  st.latex(r"""
    \text{95\% Confidence Interval: } \pm \frac{1.96}{\sqrt{n}}
    """)

  st.markdown('<div class="warning-box">', unsafe_allow_html=True)
  st.markdown("""
    **Important Notes:**
    - Large sample approximation (valid for n > 30)
    - Values outside bounds indicate significant autocorrelation
    - First lag often significant in economic/financial data
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Interactive ACF Example
  st.markdown('<p class="subsection-header">Interactive ACF Demonstration</p>',
              unsafe_allow_html=True)

  process_type = st.selectbox("Select Process Type:",
                              ["MA(1)", "MA(2)", "AR(1)", "AR(2)", "ARMA(1,1)"])
  n_samples = st.slider("Number of samples:", 100, 1000, 500)

  np.random.seed(42)

  if process_type == "MA(1)":
    theta = 0.7
    epsilon = np.random.normal(0, 1, n_samples + 1)
    y = np.zeros(n_samples)
    for t in range(n_samples):
      y[t] = epsilon[t] + theta * epsilon[t - 1]
    st.latex(r"y_t = \varepsilon_t + 0.7\varepsilon_{t-1}")

  elif process_type == "MA(2)":
    theta1, theta2 = 0.7, 0.4
    epsilon = np.random.normal(0, 1, n_samples + 2)
    y = np.zeros(n_samples)
    for t in range(n_samples):
      y[t] = epsilon[t] + theta1 * epsilon[t - 1] + theta2 * epsilon[t - 2]
    st.latex(r"y_t = \varepsilon_t + 0.7\varepsilon_{t-1} + 0.4\varepsilon_{t-2}")

  elif process_type == "AR(1)":
    phi = 0.7
    y = np.zeros(n_samples)
    epsilon = np.random.normal(0, 1, n_samples)
    for t in range(1, n_samples):
      y[t] = phi * y[t - 1] + epsilon[t]
    st.latex(r"y_t = 0.7y_{t-1} + \varepsilon_t")

  elif process_type == "AR(2)":
    phi1, phi2 = 0.7, -0.3
    y = np.zeros(n_samples)
    epsilon = np.random.normal(0, 1, n_samples)
    for t in range(2, n_samples):
      y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + epsilon[t]
    st.latex(r"y_t = 0.7y_{t-1} - 0.3y_{t-2} + \varepsilon_t")

  else:  # ARMA(1,1)
    phi, theta = 0.7, 0.4
    y = np.zeros(n_samples)
    epsilon = np.random.normal(0, 1, n_samples + 1)
    for t in range(1, n_samples):
      y[t] = phi * y[t - 1] + epsilon[t] + theta * epsilon[t - 1]
    st.latex(r"y_t = 0.7y_{t-1} + \varepsilon_t + 0.4\varepsilon_{t-1}")

  # Calculate ACF
  acf_values = acf(y, nlags=20, fft=True)
  conf_int = 1.96 / np.sqrt(len(y))

  # Plot
  fig = make_subplots(rows=1, cols=2,
                      subplot_titles=('Time Series', 'ACF Plot'))

  fig.add_trace(go.Scatter(y=y, mode='lines', name='Series'), row=1, col=1)

  fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values,
                       name='ACF', marker_color='steelblue'), row=1, col=2)
  fig.add_hline(y=conf_int, line_dash="dash", line_color="red",
                annotation_text="95% CI", row=1, col=2)
  fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=1, col=2)
  fig.add_hline(y=0, line_color="black", row=1, col=2)

  fig.update_xaxes(title_text="Time", row=1, col=1)
  fig.update_xaxes(title_text="Lag", row=1, col=2)
  fig.update_yaxes(title_text="Value", row=1, col=1)
  fig.update_yaxes(title_text="ACF", row=1, col=2)

  fig.update_layout(height=400, showlegend=False)
  st.plotly_chart(fig, use_container_width=True)

  # Interpretation
  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("**Interpretation:**")
  significant_lags = np.where(np.abs(acf_values[1:]) > conf_int)[0] + 1
  if len(significant_lags) > 0:
    st.write(f"Significant autocorrelations detected at lags: {significant_lags.tolist()}")
  else:
    st.write("No significant autocorrelations detected (resembles white noise)")
  st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# SECTION 3: PACF - PARTIAL AUTOCORRELATION FUNCTION
# ============================================================================
elif section == "📉 PACF - Partial Autocorrelation Function":
  st.markdown('<p class="section-header">PACF - Partial Autocorrelation Function</p>',
              unsafe_allow_html=True)

  st.markdown("""
    ### What is PACF?

    The Partial Autocorrelation Function (PACF) measures the correlation between yₜ and yₜ₋ₖ 
    **after removing the effect of intermediate lags** (yₜ₋₁, yₜ₋₂, ..., yₜ₋ₖ₊₁).

    It helps identify the autoregressive (AR) component of the model.
    """)

  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("**Mathematical Definition:**")
  st.markdown("""
    PACF at lag k is the correlation between yₜ and yₜ₋ₖ after controlling for 
    yₜ₋₁, yₜ₋₂, ..., yₜ₋ₖ₊₁.
    """)
  st.latex(r"""
    \phi_{kk} = Corr(y_t - \hat{y}_t, y_{t-k} - \hat{y}_{t-k})
    """)
  st.markdown("where ŷₜ is the best linear prediction of yₜ using yₜ₋₁, ..., yₜ₋ₖ₊₁")
  st.markdown('</div>', unsafe_allow_html=True)

  st.markdown("### Computing PACF via Yule-Walker Equations")
  st.markdown("""
    For an AR(p) process, the PACF can be computed by solving:
    """)
  st.latex(r"""
    \begin{bmatrix}
    1 & \rho_1 & \rho_2 & \cdots & \rho_{k-1} \\
    \rho_1 & 1 & \rho_1 & \cdots & \rho_{k-2} \\
    \rho_2 & \rho_1 & 1 & \cdots & \rho_{k-3} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \rho_{k-1} & \rho_{k-2} & \rho_{k-3} & \cdots & 1
    \end{bmatrix}
    \begin{bmatrix}
    \phi_{k1} \\
    \phi_{k2} \\
    \phi_{k3} \\
    \vdots \\
    \phi_{kk}
    \end{bmatrix}
    =
    \begin{bmatrix}
    \rho_1 \\
    \rho_2 \\
    \rho_3 \\
    \vdots \\
    \rho_k
    \end{bmatrix}
    """)
  st.markdown("The PACF at lag k is φₖₖ (the last element)")

  st.markdown("### Interpretation of PACF Patterns")

  col1, col2, col3 = st.columns(3)

  with col1:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **AR(p) Process**

        - **Sharp cutoff** after lag p
        - PACF = 0 for lags > p
        - Significant spikes up to lag p

        Example: AR(2) has non-zero PACF only at lags 1 and 2
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  with col2:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **MA(q) Process**

        - **Gradual decay** (exponential or damped sinusoidal)
        - Does NOT cut off sharply
        - Tails off slowly

        PACF alone cannot determine MA order
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  with col3:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **ARMA(p,q) Process**

        - Mixed pattern
        - Tails off gradually
        - Both ACF and PACF tail off

        Need both ACF and PACF for identification
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  st.markdown("### ACF vs PACF Summary Table")

  comparison_df = pd.DataFrame({
    'Process': ['AR(p)', 'MA(q)', 'ARMA(p,q)'],
    'ACF Behavior': ['Tails off (decays)', 'Cuts off after lag q', 'Tails off'],
    'PACF Behavior': ['Cuts off after lag p', 'Tails off (decays)', 'Tails off'],
    'Order Identification': ['Use PACF cutoff', 'Use ACF cutoff', 'Use both + criteria']
  })

  st.dataframe(comparison_df, use_container_width=True)

  st.markdown('<div class="warning-box">', unsafe_allow_html=True)
  st.markdown("""
    **Key Differences:**

    - **ACF**: Total correlation including indirect effects through intermediate lags
    - **PACF**: Direct correlation after removing intermediate lag effects
    - **ACF** is symmetric: ρₖ = ρ₋ₖ
    - **PACF** cutoff identifies AR order; **ACF** cutoff identifies MA order
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Interactive PACF Example
  st.markdown('<p class="subsection-header">Interactive PACF Demonstration</p>',
              unsafe_allow_html=True)

  process_type = st.selectbox("Select Process Type:",
                              ["AR(1)", "AR(2)", "MA(1)", "MA(2)", "ARMA(1,1)"],
                              key="pacf_process")
  n_samples = st.slider("Number of samples:", 100, 1000, 500, key="pacf_samples")

  np.random.seed(42)

  if process_type == "AR(1)":
    phi = 0.7
    y = np.zeros(n_samples)
    epsilon = np.random.normal(0, 1, n_samples)
    for t in range(1, n_samples):
      y[t] = phi * y[t - 1] + epsilon[t]
    st.latex(r"y_t = 0.7y_{t-1} + \varepsilon_t")
    expected_cutoff = 1

  elif process_type == "AR(2)":
    phi1, phi2 = 0.7, -0.3
    y = np.zeros(n_samples)
    epsilon = np.random.normal(0, 1, n_samples)
    for t in range(2, n_samples):
      y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + epsilon[t]
    st.latex(r"y_t = 0.7y_{t-1} - 0.3y_{t-2} + \varepsilon_t")
    expected_cutoff = 2

  elif process_type == "MA(1)":
    theta = 0.7
    epsilon = np.random.normal(0, 1, n_samples + 1)
    y = np.zeros(n_samples)
    for t in range(n_samples):
      y[t] = epsilon[t] + theta * epsilon[t - 1]
    st.latex(r"y_t = \varepsilon_t + 0.7\varepsilon_{t-1}")
    expected_cutoff = None

  elif process_type == "MA(2)":
    theta1, theta2 = 0.7, 0.4
    epsilon = np.random.normal(0, 1, n_samples + 2)
    y = np.zeros(n_samples)
    for t in range(n_samples):
      y[t] = epsilon[t] + theta1 * epsilon[t - 1] + theta2 * epsilon[t - 2]
    st.latex(r"y_t = \varepsilon_t + 0.7\varepsilon_{t-1} + 0.4\varepsilon_{t-2}")
    expected_cutoff = None

  else:  # ARMA(1,1)
    phi, theta = 0.7, 0.4
    y = np.zeros(n_samples)
    epsilon = np.random.normal(0, 1, n_samples + 1)
    for t in range(1, n_samples):
      y[t] = phi * y[t - 1] + epsilon[t] + theta * epsilon[t - 1]
    st.latex(r"y_t = 0.7y_{t-1} + \varepsilon_t + 0.4\varepsilon_{t-1}")
    expected_cutoff = None

  # Calculate ACF and PACF
  acf_values = acf(y, nlags=20, fft=True)
  pacf_values = pacf(y, nlags=20)
  conf_int = 1.96 / np.sqrt(len(y))

  # Plot
  fig = make_subplots(rows=2, cols=2,
                      subplot_titles=('Time Series', 'ACF Plot', '', 'PACF Plot'))

  fig.add_trace(go.Scatter(y=y, mode='lines', name='Series',
                           line=dict(color='darkblue')), row=1, col=1)

  fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values,
                       name='ACF', marker_color='steelblue'), row=1, col=2)
  fig.add_hline(y=conf_int, line_dash="dash", line_color="red", row=1, col=2)
  fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=1, col=2)
  fig.add_hline(y=0, line_color="black", row=1, col=2)

  fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values,
                       name='PACF', marker_color='darkgreen'), row=2, col=2)
  fig.add_hline(y=conf_int, line_dash="dash", line_color="red", row=2, col=2)
  fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=2, col=2)
  fig.add_hline(y=0, line_color="black", row=2, col=2)

  fig.update_xaxes(title_text="Time", row=1, col=1)
  fig.update_xaxes(title_text="Lag", row=1, col=2)
  fig.update_xaxes(title_text="Lag", row=2, col=2)
  fig.update_yaxes(title_text="Value", row=1, col=1)
  fig.update_yaxes(title_text="ACF", row=1, col=2)
  fig.update_yaxes(title_text="PACF", row=2, col=2)

  fig.update_layout(height=600, showlegend=False)
  st.plotly_chart(fig, use_container_width=True)

  # Interpretation
  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("**Interpretation:**")

  significant_acf = np.where(np.abs(acf_values[1:]) > conf_int)[0] + 1
  significant_pacf = np.where(np.abs(pacf_values[1:]) > conf_int)[0] + 1

  col1, col2 = st.columns(2)
  with col1:
    st.write("**ACF Analysis:**")
    if len(significant_acf) > 0:
      st.write(f"Significant at lags: {significant_acf.tolist()}")
      if len(significant_acf) <= 3 and max(significant_acf) == len(significant_acf):
        st.write(f"→ Suggests MA({len(significant_acf)}) component")
    else:
      st.write("No significant autocorrelations")

  with col2:
    st.write("**PACF Analysis:**")
    if len(significant_pacf) > 0:
      st.write(f"Significant at lags: {significant_pacf.tolist()}")
      if expected_cutoff:
        st.write(f"→ Expected cutoff at lag {expected_cutoff}")
        st.write(f"→ Suggests AR({expected_cutoff}) component")
    else:
      st.write("No significant partial autocorrelations")

  st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# SECTION 4: STATISTICAL TESTS
# ============================================================================
elif section == "🔬 Statistical Tests":
  st.markdown('<p class="section-header">Statistical Tests for Time Series</p>',
              unsafe_allow_html=True)

  st.markdown("""
    Before fitting ARIMA models, we need to test for stationarity and validate model assumptions.
    Here are the essential statistical tests:
    """)

  # Test 1: Augmented Dickey-Fuller (ADF) Test
  st.markdown('<p class="subsection-header">1. Augmented Dickey-Fuller (ADF) Test</p>',
              unsafe_allow_html=True)

  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("""
    **Purpose:** Test for the presence of a unit root (non-stationarity)

    **Hypotheses:**
    - H₀: The series has a unit root (non-stationary)
    - H₁: The series is stationary

    **Test Statistic:**
    """)
  st.latex(r"\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t")
  st.markdown("""
    We test if γ = 0 (unit root) vs γ < 0 (stationary)

    **Decision Rule:**
    - If p-value < 0.05: Reject H₀ → Series is stationary
    - If p-value ≥ 0.05: Fail to reject H₀ → Series is non-stationary (needs differencing)
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Test 2: KPSS Test
  st.markdown('<p class="subsection-header">2. KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)</p>',
              unsafe_allow_html=True)

  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("""
    **Purpose:** Complementary test to ADF (opposite hypotheses)

    **Hypotheses:**
    - H₀: The series is stationary
    - H₁: The series has a unit root (non-stationary)

    **Test Statistic:**
    """)
  st.latex(r"KPSS = \frac{1}{T^2} \sum_{t=1}^{T} \frac{S_t^2}{\hat{\sigma}^2}")
  st.markdown("""
    where Sₜ is the partial sum of residuals

    **Decision Rule:**
    - If p-value < 0.05: Reject H₀ → Series is non-stationary
    - If p-value ≥ 0.05: Fail to reject H₀ → Series is stationary

    **Comparison with ADF:**
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  comparison_table = pd.DataFrame({
    'Test': ['ADF', 'KPSS'],
    'H₀': ['Non-stationary (unit root)', 'Stationary'],
    'H₁': ['Stationary', 'Non-stationary'],
    'When to use': ['Primary test', 'Confirmatory test'],
    'Stationary if': ['p-value < 0.05', 'p-value > 0.05']
  })
  st.dataframe(comparison_table, use_container_width=True)

  st.markdown('<div class="warning-box">', unsafe_allow_html=True)
  st.markdown("""
    **Best Practice:** Use both tests together!

    | ADF Result | KPSS Result | Interpretation |
    |------------|-------------|----------------|
    | Stationary | Stationary | ✅ Clearly stationary |
    | Non-stationary | Non-stationary | ❌ Clearly non-stationary |
    | Stationary | Non-stationary | ⚠️ Difference stationary |
    | Non-stationary | Stationary | ⚠️ Trend stationary |
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Test 3: Ljung-Box Test
  st.markdown('<p class="subsection-header">3. Ljung-Box Test</p>',
              unsafe_allow_html=True)

  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("""
    **Purpose:** Test for autocorrelation in residuals (model adequacy check)

    **Hypotheses:**
    - H₀: No autocorrelation in residuals (model is adequate)
    - H₁: Autocorrelation exists in residuals (model is inadequate)

    **Test Statistic:**
    """)
  st.latex(r"Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k}")
  st.markdown("""
    where:
    - n = sample size
    - h = number of lags tested
    - ρ̂ₖ = sample autocorrelation at lag k

    **Distribution:** Q ~ χ²(h - p - q) under H₀

    **Decision Rule:**
    - If p-value > 0.05: Fail to reject H₀ → Residuals are white noise ✅
    - If p-value < 0.05: Reject H₀ → Residuals have autocorrelation ❌
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Test 4: Normality Tests
  st.markdown('<p class="subsection-header">4. Normality Tests for Residuals</p>',
              unsafe_allow_html=True)

  col1, col2 = st.columns(2)

  with col1:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
        **Jarque-Bera Test**

        Tests if residuals follow normal distribution
        """)
    st.latex(r"JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)")
    st.markdown("""
        where:
        - S = skewness
        - K = kurtosis

        **Decision:**
        - p > 0.05: Residuals are normal
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
        **Shapiro-Wilk Test**

        More powerful for small samples
        """)
    st.latex(r"W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}")
    st.markdown("""
        where x₍ᵢ₎ are ordered sample values

        **Decision:**
        - p > 0.05: Residuals are normal
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  # Interactive Testing
  st.markdown('<p class="subsection-header">Interactive Statistical Testing</p>',
              unsafe_allow_html=True)

  test_process = st.selectbox("Select Process for Testing:",
                              ["Stationary AR(1)", "Non-stationary Random Walk",
                               "Trend + Noise", "Seasonal"],
                              key="test_process")
  n_test = st.slider("Sample size:", 100, 1000, 300, key="test_n")

  np.random.seed(42)

  if test_process == "Stationary AR(1)":
    y_test = np.zeros(n_test)
    epsilon = np.random.normal(0, 1, n_test)
    for t in range(1, n_test):
      y_test[t] = 0.6 * y_test[t - 1] + epsilon[t]
    description = "Stationary AR(1): yₜ = 0.6yₜ₋₁ + εₜ"

  elif test_process == "Non-stationary Random Walk":
    epsilon = np.random.normal(0, 1, n_test)
    y_test = np.cumsum(epsilon)
    description = "Random Walk: yₜ = yₜ₋₁ + εₜ"

  elif test_process == "Trend + Noise":
    trend = np.linspace(0, 10, n_test)
    noise = np.random.normal(0, 1, n_test)
    y_test = trend + noise
    description = "Linear Trend: yₜ = 0.1t + εₜ"

  else:  # Seasonal
    t = np.arange(n_test)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 1, n_test)
    y_test = seasonal + noise
    description = "Seasonal: yₜ = 10sin(2πt/12) + εₜ"

  st.write(f"**Process:** {description}")

  # Plot series
  fig = go.Figure()
  fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Series',
                           line=dict(color='darkblue')))
  fig.update_layout(title='Time Series', xaxis_title='Time',
                    yaxis_title='Value', height=300)
  st.plotly_chart(fig, use_container_width=True)

  # Perform tests
  st.markdown("### Test Results")

  col1, col2, col3 = st.columns(3)

  with col1:
    st.markdown("**ADF Test**")
    adf_result = adfuller(y_test)
    st.metric("Test Statistic", f"{adf_result[0]:.4f}")
    st.metric("p-value", f"{adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
      st.success("✅ Stationary")
    else:
      st.error("❌ Non-stationary")
    st.write(f"Critical Values:")
    for key, value in adf_result[4].items():
      st.write(f"{key}: {value:.4f}")

  with col2:
    st.markdown("**KPSS Test**")
    kpss_result = kpss(y_test, regression='c')
    st.metric("Test Statistic", f"{kpss_result[0]:.4f}")
    st.metric("p-value", f"{kpss_result[1]:.4f}")
    if kpss_result[1] > 0.05:
      st.success("✅ Stationary")
    else:
      st.error("❌ Non-stationary")
    st.write(f"Critical Values:")
    for key, value in kpss_result[3].items():
      st.write(f"{key}: {value:.4f}")

  with col3:
    st.markdown("**Normality Test**")
    jb_stat, jb_pvalue = stats.jarque_bera(y_test)
    st.metric("JB Statistic", f"{jb_stat:.4f}")
    st.metric("p-value", f"{jb_pvalue:.4f}")
    if jb_pvalue > 0.05:
      st.success("✅ Normal")
    else:
      st.warning("⚠️ Non-normal")

    sw_stat, sw_pvalue = stats.shapiro(y_test[:min(5000, len(y_test))])
    st.write(f"**Shapiro-Wilk:**")
    st.write(f"p-value: {sw_pvalue:.4f}")

  # ACF of series
  acf_test = acf(y_test, nlags=20, fft=True)

  st.markdown("**Ljung-Box Test (Multiple Lags)**")
  lb_result = acorr_ljungbox(y_test, lags=10, return_df=True)
  st.dataframe(lb_result.style.highlight_between(subset=['lb_pvalue'],
                                                 left=0, right=0.05,
                                                 color='#ffcccc'),
               use_container_width=True)

  st.markdown('<div class="success-box">', unsafe_allow_html=True)
  st.markdown("**Interpretation Summary:**")

  is_stationary_adf = adf_result[1] < 0.05
  is_stationary_kpss = kpss_result[1] > 0.05

  if is_stationary_adf and is_stationary_kpss:
    st.write("✅ **Both tests confirm: Series is STATIONARY**")
    st.write("→ Can use ARMA model (d=0)")
  elif not is_stationary_adf and not is_stationary_kpss:
    st.write("❌ **Both tests confirm: Series is NON-STATIONARY**")
    st.write("→ Need differencing (d≥1) - use ARIMA")
  else:
    st.write("⚠️ **Tests disagree - further investigation needed**")
    st.write("→ Try differencing and re-test")

  st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# SECTION 5: ARIMA ORDER SELECTION
# ============================================================================
elif section == "🎯 ARIMA Order Selection":
  st.markdown('<p class="section-header">ARIMA Order Selection Strategy</p>',
              unsafe_allow_html=True)

  st.markdown("""
    Selecting the correct ARIMA(p, d, q) order is crucial for model performance. 
    Here's a comprehensive strategy:
    """)

  # Step-by-step process
  st.markdown('<p class="subsection-header">Step-by-Step Selection Process</p>',
              unsafe_allow_html=True)

  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("""
    ### **Step 1: Determine d (Differencing Order)**

    **Goal:** Make the series stationary

    **Method:**
    1. Plot the series and check for trend/non-stationarity
    2. Run ADF and KPSS tests
    3. If non-stationary, difference the series: Δyₜ = yₜ - yₜ₋₁
    4. Re-test until stationary

    **Common values:**
    - d = 0: Already stationary
    - d = 1: Non-stationary with linear trend (most common)
    - d = 2: Non-stationary with quadratic trend (rare)

    ⚠️ **Warning:** Over-differencing can introduce artificial autocorrelation!
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("""
    ### **Step 2: Determine p and q (AR and MA Orders)**

    After achieving stationarity, use ACF and PACF plots:
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Decision Table
  st.markdown("### ACF/PACF Pattern Recognition Table")

  pattern_df = pd.DataFrame({
    'Model': ['AR(p)', 'MA(q)', 'ARMA(p,q)', 'White Noise'],
    'ACF Pattern': [
      'Decays exponentially or with damped sine wave',
      'Cuts off sharply after lag q',
      'Decays gradually',
      'All lags insignificant'
    ],
    'PACF Pattern': [
      'Cuts off sharply after lag p',
      'Decays exponentially or with damped sine wave',
      'Decays gradually',
      'All lags insignificant'
    ],
    'Order Selection': [
      'p = lag where PACF cuts off',
      'q = lag where ACF cuts off',
      'Try different combinations',
      'No model needed'
    ]
  })

  st.dataframe(pattern_df, use_container_width=True)

  # Information Criteria
  st.markdown('<p class="subsection-header">Step 3: Model Selection Criteria</p>',
              unsafe_allow_html=True)

  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("""
    When patterns are unclear or multiple models seem plausible, use information criteria:
    """)

  col1, col2, col3 = st.columns(3)

  with col1:
    st.markdown("**AIC (Akaike Information Criterion)**")
    st.latex(r"AIC = -2\ln(L) + 2k")
    st.markdown("""
        - L = likelihood
        - k = number of parameters
        - Lower is better
        - Tends to prefer complex models
        """)

  with col2:
    st.markdown("**BIC (Bayesian Information Criterion)**")
    st.latex(r"BIC = -2\ln(L) + k\ln(n)")
    st.markdown("""
        - n = sample size
        - Lower is better
        - Stronger penalty for complexity
        - Preferred for larger samples
        """)

  with col3:
    st.markdown("**AICc (Corrected AIC)**")
    st.latex(r"AICc = AIC + \frac{2k(k+1)}{n-k-1}")
    st.markdown("""
        - Corrects for small samples
        - Use when n/k < 40
        - Lower is better
        - Prevents overfitting
        """)

  st.markdown('</div>', unsafe_allow_html=True)

  st.markdown('<div class="warning-box">', unsafe_allow_html=True)
  st.markdown("""
    **Selection Strategy:**

    1. **Initial Candidates:** Based on ACF/PACF, identify 3-5 candidate models
    2. **Fit Models:** Estimate parameters for each candidate
    3. **Compare Criteria:** Compare AIC, BIC, AICc
    4. **Check Residuals:** Verify residuals are white noise (Ljung-Box test)
    5. **Select Best:** Choose model with:
       - Lowest information criterion
       - White noise residuals
       - Parsimony (fewer parameters if similar performance)
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Detailed Selection Cases
  st.markdown('<p class="subsection-header">Common Pattern Cases</p>',
              unsafe_allow_html=True)

  case_tabs = st.tabs(["Case 1: Clear AR", "Case 2: Clear MA",
                       "Case 3: Mixed ARMA", "Case 4: Seasonal"])

  with case_tabs[0]:
    st.markdown("### Case 1: Pure AR Process")
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **Indicators:**
        - PACF cuts off sharply after lag p
        - ACF decays gradually (exponential or sinusoidal)

        **Example:** AR(2) identification
        - PACF: Significant at lags 1 and 2, then cuts off
        - ACF: Gradual decay

        **Decision:** p = 2, q = 0
        **Model:** ARIMA(2, d, 0)
        """)

    # Generate AR(2) example
    np.random.seed(42)
    n = 500
    y_ar = np.zeros(n)
    epsilon = np.random.normal(0, 1, n)
    for t in range(2, n):
      y_ar[t] = 0.7 * y_ar[t - 1] - 0.3 * y_ar[t - 2] + epsilon[t]

    acf_ar = acf(y_ar, nlags=20)
    pacf_ar = pacf(y_ar, nlags=20)
    conf = 1.96 / np.sqrt(n)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'PACF'))

    fig.add_trace(go.Bar(x=list(range(len(acf_ar))), y=acf_ar,
                         marker_color='steelblue'), row=1, col=1)
    fig.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Bar(x=list(range(len(pacf_ar))), y=pacf_ar,
                         marker_color='darkgreen'), row=1, col=2)
    fig.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=2)

    # Highlight cutoff
    fig.add_vrect(x0=2.5, x1=20, fillcolor="yellow", opacity=0.2,
                  line_width=0, row=1, col=2)
    fig.add_annotation(x=10, y=0.5, text="PACF cuts off after lag 2",
                       showarrow=True, row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

  with case_tabs[1]:
    st.markdown("### Case 2: Pure MA Process")
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **Indicators:**
        - ACF cuts off sharply after lag q
        - PACF decays gradually

        **Example:** MA(1) identification
        - ACF: Significant only at lag 1, then cuts off
        - PACF: Gradual decay

        **Decision:** p = 0, q = 1
        **Model:** ARIMA(0, d, 1)
        """)

    # Generate MA(1) example
    np.random.seed(42)
    epsilon = np.random.normal(0, 1, n + 1)
    y_ma = np.zeros(n)
    for t in range(n):
      y_ma[t] = epsilon[t] + 0.7 * epsilon[t - 1]

    acf_ma = acf(y_ma, nlags=20)
    pacf_ma = pacf(y_ma, nlags=20)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'PACF'))

    fig.add_trace(go.Bar(x=list(range(len(acf_ma))), y=acf_ma,
                         marker_color='steelblue'), row=1, col=1)
    fig.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=1)

    # Highlight cutoff
    fig.add_vrect(x0=1.5, x1=20, fillcolor="yellow", opacity=0.2,
                  line_width=0, row=1, col=1)
    fig.add_annotation(x=10, y=0.3, text="ACF cuts off after lag 1",
                       showarrow=True, row=1, col=1)

    fig.add_trace(go.Bar(x=list(range(len(pacf_ma))), y=pacf_ma,
                         marker_color='darkgreen'), row=1, col=2)
    fig.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

  with case_tabs[2]:
    st.markdown("### Case 3: Mixed ARMA Process")
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **Indicators:**
        - Both ACF and PACF decay gradually
        - No clear cutoff in either

        **Strategy:**
        1. Try combinations: ARMA(1,1), ARMA(2,1), ARMA(1,2), ARMA(2,2)
        2. Compare using AIC/BIC
        3. Check residual diagnostics

        **Example:** ARMA(1,1) identification
        - ACF: Gradual decay
        - PACF: Gradual decay
        - Try multiple models and use information criteria
        """)

    # Generate ARMA(1,1) example
    np.random.seed(42)
    y_arma = np.zeros(n)
    epsilon = np.random.normal(0, 1, n + 1)
    for t in range(1, n):
      y_arma[t] = 0.7 * y_arma[t - 1] + epsilon[t] + 0.4 * epsilon[t - 1]

    acf_arma = acf(y_arma, nlags=20)
    pacf_arma = pacf(y_arma, nlags=20)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'PACF'))

    fig.add_trace(go.Bar(x=list(range(len(acf_arma))), y=acf_arma,
                         marker_color='steelblue'), row=1, col=1)
    fig.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_annotation(x=10, y=0.5, text="Gradual decay", showarrow=False,
                       row=1, col=1)

    fig.add_trace(go.Bar(x=list(range(len(pacf_arma))), y=pacf_arma,
                         marker_color='darkgreen'), row=1, col=2)
    fig.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_annotation(x=10, y=0.5, text="Gradual decay", showarrow=False,
                       row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Model comparison
    st.markdown("**Model Comparison Example:**")

    models_to_try = [(1, 0, 1), (2, 0, 1), (1, 0, 2), (2, 0, 2)]
    results = []

    for order in models_to_try:
      try:
        model = ARIMA(y_arma, order=order)
        fit = model.fit()
        results.append({
          'Model': f'ARIMA{order}',
          'AIC': fit.aic,
          'BIC': fit.bic,
          'Parameters': order[0] + order[2] + 1
        })
      except:
        pass

    results_df = pd.DataFrame(results)
    results_df['Best_AIC'] = results_df['AIC'] == results_df['AIC'].min()
    results_df['Best_BIC'] = results_df['BIC'] == results_df['BIC'].min()

    st.dataframe(results_df.style.highlight_min(subset=['AIC', 'BIC'],
                                                color='lightgreen'),
                 use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

  with case_tabs[3]:
    st.markdown("### Case 4: Seasonal Patterns")
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
        **Indicators:**
        - ACF/PACF show spikes at seasonal lags (12, 24, 36 for monthly data)
        - Regular pattern repeating at seasonal frequency

        **Model:** SARIMA(p,d,q)(P,D,Q)ₛ
        - (p,d,q): Non-seasonal components
        - (P,D,Q)ₛ: Seasonal components at period s

        **Strategy:**
        1. Identify seasonal period (s)
        2. Check for seasonal differencing need (D)
        3. Look for spikes at seasonal lags in ACF/PACF
        4. Seasonal AR order (P) from PACF at seasonal lags
        5. Seasonal MA order (Q) from ACF at seasonal lags

        **Example:** Monthly data with yearly seasonality
        - If spike at lag 12 in ACF → Q=1
        - If spike at lag 12 in PACF → P=1
        - Model: SARIMA(p,d,q)(P,D,Q)₁₂
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  # Practical Workflow
  st.markdown('<p class="subsection-header">Complete Workflow Summary</p>',
              unsafe_allow_html=True)

  st.markdown('<div class="info-box">', unsafe_allow_html=True)
  st.markdown("""
    ### **Recommended Workflow:**

    **1. Data Preparation**
    - Plot time series
    - Check for missing values, outliers
    - Transform if needed (log, Box-Cox)

    **2. Stationarity Testing**
    - Visual inspection
    - ADF test and KPSS test
    - Difference if necessary (determine d)

    **3. Model Identification**
    - Plot ACF and PACF of stationary series
    - Identify potential p and q values
    - Create 3-5 candidate models

    **4. Model Estimation**
    - Fit candidate models
    - Compare AIC, BIC, AICc
    - Check parameter significance

    **5. Diagnostic Checking**
    - Ljung-Box test on residuals
    - Plot residual ACF/PACF
    - Normality tests
    - Check for heteroscedasticity

    **6. Model Selection**
    - Choose model with:
      * Best information criteria
      * White noise residuals
      * Fewest parameters (parsimony)
      * All parameters significant

    **7. Forecasting**
    - Generate forecasts
    - Calculate prediction intervals
    - Validate on test set if available
    """)
  st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# SECTION 6: PRACTICAL CASES & EXAMPLES
# ============================================================================
elif section == "💡 Practical Cases & Examples":
  st.markdown('<p class="section-header">Practical Cases & Examples</p>',
              unsafe_allow_html=True)

  st.markdown("""
    Let's walk through complete real-world examples of ARIMA model identification.
    """)

  case_type = st.selectbox("Select Case Study:",
                           ["Case 1: Stock Returns (Stationary)",
                            "Case 2: GDP Growth (Trending)",
                            "Case 3: Temperature (Seasonal)",
                            "Case 4: Sales Data (Mixed)"])

  np.random.seed(42)
  n = 500

  if case_type == "Case 1: Stock Returns (Stationary)":
    st.markdown("### Case 1: Daily Stock Returns")
    st.markdown("""
        **Context:** Daily stock returns are typically stationary (no unit root in returns)

        **Expected:** ARMA model (d=0)
        """)

    # Generate stock returns-like data (ARMA(1,1))
    data = np.zeros(n)
    epsilon = np.random.normal(0, 0.02, n + 1)
    for t in range(1, n):
      data[t] = 0.1 * data[t - 1] + epsilon[t] + 0.3 * epsilon[t - 1]

    title = "Simulated Daily Stock Returns"

  elif case_type == "Case 2: GDP Growth (Trending)":
    st.markdown("### Case 2: GDP with Linear Trend")
    st.markdown("""
        **Context:** Economic indicators often have trends requiring differencing

        **Expected:** ARIMA with d=1
        """)

    # Generate trending data
    trend = 0.01 * np.arange(n)
    ar_component = np.zeros(n)
    epsilon = np.random.normal(0, 0.5, n)
    for t in range(1, n):
      ar_component[t] = 0.8 * ar_component[t - 1] + epsilon[t]
    data = trend + ar_component

    title = "Simulated GDP (with trend)"

  elif case_type == "Case 3: Temperature (Seasonal)":
    st.markdown("### Case 3: Monthly Temperature")
    st.markdown("""
        **Context:** Temperature has strong seasonal patterns

        **Expected:** SARIMA model with seasonal components
        """)

    # Generate seasonal data
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 12) + 5 * np.cos(2 * np.pi * t / 12)
    ar = np.zeros(n)
    epsilon = np.random.normal(0, 2, n)
    for i in range(1, n):
      ar[i] = 0.6 * ar[i - 1] + epsilon[i]
    data = seasonal + ar + 20

    title = "Simulated Monthly Temperature"

  else:  # Sales Data
    st.markdown("### Case 4: Monthly Sales Data")
    st.markdown("""
        **Context:** Sales data with trend and irregular fluctuations

        **Expected:** ARIMA(p,1,q) or ARMA after detrending
        """)

    # Generate sales-like data
    trend = 0.05 * np.arange(n)
    arma = np.zeros(n)
    epsilon = np.random.normal(0, 1, n + 1)
    for t in range(2, n):
      arma[t] = 0.6 * arma[t - 1] + epsilon[t] + 0.4 * epsilon[t - 1]
    data = 100 + trend + arma * 10

    title = "Simulated Monthly Sales"

  # Analysis
  st.markdown("### Step 1: Visualize the Data")

  fig = go.Figure()
  fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Series',
                           line=dict(color='darkblue')))
  fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Value',
                    height=300)
  st.plotly_chart(fig, use_container_width=True)

  # Statistics
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric("Mean", f"{np.mean(data):.3f}")
  with col2:
    st.metric("Std Dev", f"{np.std(data):.3f}")
  with col3:
    st.metric("Min", f"{np.min(data):.3f}")
  with col4:
    st.metric("Max", f"{np.max(data):.3f}")

  # Step 2: Stationarity Tests
  st.markdown("### Step 2: Test for Stationarity")

  col1, col2 = st.columns(2)

  with col1:
    st.markdown("**ADF Test (Original Series)**")
    adf_result = adfuller(data)

    results_dict = {
      'Metric': ['Test Statistic', 'p-value', '# Lags Used', '# Observations'],
      'Value': [f"{adf_result[0]:.4f}", f"{adf_result[1]:.4f}",
                adf_result[2], adf_result[3]]
    }
    st.dataframe(pd.DataFrame(results_dict), use_container_width=True)

    if adf_result[1] < 0.05:
      st.success("✅ Series is stationary (p < 0.05)")
      d_suggested = 0
    else:
      st.error("❌ Series is non-stationary (p ≥ 0.05)")
      st.info("→ Differencing required")
      d_suggested = 1

  with col2:
    st.markdown("**KPSS Test (Original Series)**")
    kpss_result = kpss(data, regression='c')

    results_dict = {
      'Metric': ['Test Statistic', 'p-value', '# Lags Used'],
      'Value': [f"{kpss_result[0]:.4f}", f"{kpss_result[1]:.4f}",
                kpss_result[2]]
    }
    st.dataframe(pd.DataFrame(results_dict), use_container_width=True)

    if kpss_result[1] > 0.05:
      st.success("✅ Series is stationary (p > 0.05)")
    else:
      st.error("❌ Series is non-stationary (p ≤ 0.05)")

  # Differencing if needed
  if d_suggested > 0:
    st.markdown("### Differencing Applied")
    data_diff = np.diff(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data_diff, mode='lines',
                             name='Differenced Series',
                             line=dict(color='green')))
    fig.update_layout(title='First Difference', xaxis_title='Time',
                      yaxis_title='Δ Value', height=250)
    st.plotly_chart(fig, use_container_width=True)

    # Re-test
    adf_diff = adfuller(data_diff)
    st.info(f"ADF test on differenced series: p-value = {adf_diff[1]:.4f}")
    if adf_diff[1] < 0.05:
      st.success("✅ Differenced series is now stationary")
      data_stationary = data_diff
    else:
      st.warning("May need second differencing")
      data_stationary = np.diff(data_diff)
      d_suggested = 2
  else:
    data_stationary = data

  # Step 3: ACF and PACF
  st.markdown(f"### Step 3: ACF and PACF Analysis (d={d_suggested})")

  acf_vals = acf(data_stationary, nlags=24, fft=True)
  pacf_vals = pacf(data_stationary, nlags=24)
  conf_int = 1.96 / np.sqrt(len(data_stationary))

  fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'PACF'))

  fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals,
                       marker_color='steelblue', name='ACF'), row=1, col=1)
  fig.add_hline(y=conf_int, line_dash="dash", line_color="red", row=1, col=1)
  fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=1, col=1)
  fig.add_hline(y=0, line_color="black", row=1, col=1)

  fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals,
                       marker_color='darkgreen', name='PACF'), row=1, col=2)
  fig.add_hline(y=conf_int, line_dash="dash", line_color="red", row=1, col=2)
  fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=1, col=2)
  fig.add_hline(y=0, line_color="black", row=1, col=2)

  fig.update_xaxes(title_text="Lag", row=1, col=1)
  fig.update_xaxes(title_text="Lag", row=1, col=2)
  fig.update_yaxes(title_text="ACF", row=1, col=1)
  fig.update_yaxes(title_text="PACF", row=1, col=2)

  fig.update_layout(height=400, showlegend=False)
  st.plotly_chart(fig, use_container_width=True)

  # Identify orders
  sig_acf = np.where(np.abs(acf_vals[1:]) > conf_int)[0] + 1
  sig_pacf = np.where(np.abs(pacf_vals[1:]) > conf_int)[0] + 1

  st.markdown("**Pattern Analysis:**")
  col1, col2 = st.columns(2)

  with col1:
    st.write("**ACF significant lags:**", sig_acf[:5].tolist() if len(sig_acf) > 0 else "None")

    # Determine if cutoff
    if len(sig_acf) > 0:
      max_sig_acf = max(sig_acf[:5]) if len(sig_acf[:5]) > 0 else 0
      if len(sig_acf) <= 3:
        st.success(f"→ Sharp cutoff after lag {max_sig_acf}")
        st.write(f"→ Suggests MA({max_sig_acf})")
        q_suggested = max_sig_acf
      else:
        st.info("→ Gradual decay (no clear cutoff)")
        q_suggested = 0
    else:
      q_suggested = 0

  with col2:
    st.write("**PACF significant lags:**", sig_pacf[:5].tolist() if len(sig_pacf) > 0 else "None")

    # Determine if cutoff
    if len(sig_pacf) > 0:
      max_sig_pacf = max(sig_pacf[:5]) if len(sig_pacf[:5]) > 0 else 0
      if len(sig_pacf) <= 3:
        st.success(f"→ Sharp cutoff after lag {max_sig_pacf}")
        st.write(f"→ Suggests AR({max_sig_pacf})")
        p_suggested = max_sig_pacf
      else:
        st.info("→ Gradual decay (no clear cutoff)")
        p_suggested = 1
    else:
      p_suggested = 0

  # Step 4: Model Selection
  st.markdown("### Step 4: Candidate Models and Selection")

  # Generate candidates
  candidates = []

  # Based on ACF/PACF
  if p_suggested > 0 and q_suggested == 0:
    candidates.append((p_suggested, d_suggested, 0))
  elif q_suggested > 0 and p_suggested == 0:
    candidates.append((0, d_suggested, q_suggested))
  else:
    # Mixed - try combinations
    for p in range(max(p_suggested, 1), min(p_suggested + 2, 4)):
      for q in range(max(q_suggested, 1), min(q_suggested + 2, 4)):
        candidates.append((p, d_suggested, q))

  # Also try simple models
  candidates.append((1, d_suggested, 0))
  candidates.append((0, d_suggested, 1))
  candidates.append((1, d_suggested, 1))

  # Remove duplicates
  candidates = list(set(candidates))

  st.write(f"**Testing {len(candidates)} candidate models...**")

  results = []
  for order in candidates:
    try:
      model = ARIMA(data, order=order)
      fit = model.fit()

      # Ljung-Box test
      residuals = fit.resid
      lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
      lb_pvalue = lb_test['lb_pvalue'].iloc[-1]

      results.append({
        'Model': f'ARIMA{order}',
        'p': order[0],
        'd': order[1],
        'q': order[2],
        'AIC': fit.aic,
        'BIC': fit.bic,
        'LB p-value': lb_pvalue,
        'White Noise': '✅' if lb_pvalue > 0.05 else '❌'
      })
    except:
      pass

  results_df = pd.DataFrame(results).sort_values('AIC')

  # Highlight best
  st.dataframe(results_df.style.highlight_min(subset=['AIC', 'BIC'],
                                              color='lightgreen'),
               use_container_width=True)

  st.markdown('<div class="success-box">', unsafe_allow_html=True)
  best_model = results_df.iloc[0]
  st.markdown(f"""
    **Recommended Model:** {best_model['Model']}

    **Justification:**
    - Lowest AIC: {best_model['AIC']:.2f}
    - BIC: {best_model['BIC']:.2f}
    - Ljung-Box p-value: {best_model['LB p-value']:.4f} {best_model['White Noise']}
    - Parameters: p={int(best_model['p'])}, d={int(best_model['d'])}, q={int(best_model['q'])}
    """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Step 5: Fit best model and diagnostics
  st.markdown("### Step 5: Model Diagnostics")

  best_order = (int(best_model['p']), int(best_model['d']), int(best_model['q']))
  final_model = ARIMA(data, order=best_order).fit()

  residuals = final_model.resid

  # Residual plots
  fig = make_subplots(rows=2, cols=2,
                      subplot_titles=('Residuals', 'Residual ACF',
                                      'Residual Distribution', 'Q-Q Plot'))

  # Residuals over time
  fig.add_trace(go.Scatter(y=residuals, mode='lines', name='Residuals',
                           line=dict(color='red')), row=1, col=1)
  fig.add_hline(y=0, line_color='black', row=1, col=1)

  # Residual ACF
  resid_acf = acf(residuals, nlags=20)
  fig.add_trace(go.Bar(x=list(range(len(resid_acf))), y=resid_acf,
                       marker_color='steelblue'), row=1, col=2)
  conf = 1.96 / np.sqrt(len(residuals))
  fig.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=2)
  fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=2)

  # Histogram
  fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Distribution',
                             marker_color='lightblue'), row=2, col=1)

  # Q-Q plot
  from scipy.stats import probplot

  qq = probplot(residuals, dist="norm")
  fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                           name='Q-Q', marker=dict(color='blue')), row=2, col=2)
  fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1],
                           mode='lines', name='Theoretical',
                           line=dict(color='red')), row=2, col=2)

  fig.update_layout(height=600, showlegend=False)
  st.plotly_chart(fig, use_container_width=True)

  # Diagnostic tests
  col1, col2, col3 = st.columns(3)

  with col1:
    st.markdown("**Ljung-Box Test**")
    lb_full = acorr_ljungbox(residuals, lags=10, return_df=True)
    st.dataframe(lb_full[['lb_stat', 'lb_pvalue']], use_container_width=True)

    if all(lb_full['lb_pvalue'] > 0.05):
      st.success("✅ No autocorrelation")
    else:
      st.warning("⚠️ Some autocorrelation detected")

  with col2:
    st.markdown("**Normality Tests**")
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    sw_stat, sw_pval = stats.shapiro(residuals[:min(5000, len(residuals))])

    norm_results = pd.DataFrame({
      'Test': ['Jarque-Bera', 'Shapiro-Wilk'],
      'Statistic': [f"{jb_stat:.4f}", f"{sw_stat:.4f}"],
      'p-value': [f"{jb_pval:.4f}", f"{sw_pval:.4f}"]
    })
    st.dataframe(norm_results, use_container_width=True)

    if jb_pval > 0.05 and sw_pval > 0.05:
      st.success("✅ Residuals normal")
    else:
      st.info("ℹ️ Slight deviation from normality")

  with col3:
    st.markdown("**Residual Statistics**")
    resid_stats = pd.DataFrame({
      'Metric': ['Mean', 'Std Dev', 'Skewness', 'Kurtosis'],
      'Value': [
        f"{np.mean(residuals):.4f}",
        f"{np.std(residuals):.4f}",
        f"{stats.skew(residuals):.4f}",
        f"{stats.kurtosis(residuals):.4f}"
      ]
    })
    st.dataframe(resid_stats, use_container_width=True)

  # Model Summary
  st.markdown("### Model Summary")
  st.text(final_model.summary())

# ============================================================================
# SECTION 7: INTERACTIVE SIMULATION
# ============================================================================
else:  # Interactive Simulation
  st.markdown('<p class="section-header">Interactive ARIMA Simulation</p>',
              unsafe_allow_html=True)

  st.markdown("""
    Build your own ARIMA model and see how ACF/PACF patterns emerge!
    """)

  col1, col2 = st.columns(2)

  with col1:
    st.markdown("### Model Parameters")

    model_type = st.radio("Model Type:", ["AR", "MA", "ARMA", "Custom ARIMA"])

    if model_type == "AR":
      p = st.slider("AR Order (p):", 0, 5, 1)
      d = 0
      q = 0

      st.markdown("**AR Coefficients:**")
      ar_coefs = []
      for i in range(p):
        coef = st.slider(f"φ_{i + 1}:", -0.95, 0.95, 0.5, 0.05, key=f"ar_{i}")
        ar_coefs.append(coef)
      ma_coefs = []

    elif model_type == "MA":
      p = 0
      d = 0
      q = st.slider("MA Order (q):", 0, 5, 1)

      st.markdown("**MA Coefficients:**")
      ma_coefs = []
      for i in range(q):
        coef = st.slider(f"θ_{i + 1}:", -0.95, 0.95, 0.5, 0.05, key=f"ma_{i}")
        ma_coefs.append(coef)
      ar_coefs = []

    elif model_type == "ARMA":
      p = st.slider("AR Order (p):", 0, 3, 1)
      d = 0
      q = st.slider("MA Order (q):", 0, 3, 1)

      ar_coefs = []
      ma_coefs = []

      if p > 0:
        st.markdown("**AR Coefficients:**")
        for i in range(p):
          coef = st.slider(f"φ_{i + 1}:", -0.95, 0.95, 0.5, 0.05, key=f"ar_{i}")
          ar_coefs.append(coef)

      if q > 0:
        st.markdown("**MA Coefficients:**")
        for i in range(q):
          coef = st.slider(f"θ_{i + 1}:", -0.95, 0.95, 0.5, 0.05, key=f"ma_{i}")
          ma_coefs.append(coef)

    else:  # Custom ARIMA
      p = st.slider("AR Order (p):", 0, 3, 1)
      d = st.slider("Differencing (d):", 0, 2, 0)
      q = st.slider("MA Order (q):", 0, 3, 1)

      ar_coefs = []
      ma_coefs = []

      if p > 0:
        st.markdown("**AR Coefficients:**")
        for i in range(p):
          coef = st.slider(f"φ_{i + 1}:", -0.95, 0.95, 0.5, 0.05, key=f"ar_{i}")
          ar_coefs.append(coef)

      if q > 0:
        st.markdown("**MA Coefficients:**")
        for i in range(q):
          coef = st.slider(f"θ_{i + 1}:", -0.95, 0.95, 0.5, 0.05, key=f"ma_{i}")
          ma_coefs.append(coef)

    n_sim = st.slider("Sample Size:", 100, 2000, 500)
    noise_std = st.slider("Noise Std Dev:", 0.1, 5.0, 1.0, 0.1)

  with col2:
    st.markdown("### Model Equation")

    # Display equation
    if model_type == "AR" or (model_type in ["ARMA", "Custom ARIMA"] and p > 0 and q == 0):
      eq_parts = ["y_t = "]
      for i, coef in enumerate(ar_coefs):
        eq_parts.append(f"{coef:.2f}y_{{t-{i + 1}}} + ")
      eq_parts.append(r"\varepsilon_t")
      st.latex("".join(eq_parts))

    elif model_type == "MA" or (model_type in ["ARMA", "Custom ARIMA"] and p == 0 and q > 0):
      eq_parts = ["y_t = \varepsilon_t + "]
      for i, coef in enumerate(ma_coefs):
        eq_parts.append(f"{coef:.2f}\varepsilon_{{t-{i + 1}}}")
        if i < len(ma_coefs) - 1:
          eq_parts.append(" + ")
      st.latex("".join(eq_parts))

    else:
      ar_part = " + ".join([f"{c:.2f}y_{{t-{i + 1}}}" for i, c in enumerate(ar_coefs)])
      ma_part = " + ".join([f"{c:.2f}\varepsilon_{{t-{i + 1}}}" for i, c in enumerate(ma_coefs)])

      if d > 0:
        st.latex(f"\Delta^{d} y_t = {ar_part} + \varepsilon_t + {ma_part}")
      else:
        st.latex(f"y_t = {ar_part} + \varepsilon_t + {ma_part}")

    st.markdown("### Theoretical Properties")

    # Stationarity check for AR
    if p > 0:
      # Check AR polynomial roots
      ar_poly = np.array([1] + [-c for c in ar_coefs])
      roots = np.roots(ar_poly)
      root_mags = np.abs(roots)

      if all(root_mags > 1):
        st.success("✅ AR process is stationary")
        st.write(f"All roots outside unit circle: {root_mags}")
      else:
        st.error("❌ AR process is non-stationary")
        st.write(f"Roots: {root_mags}")

    # Invertibility check for MA
    if q > 0:
      ma_poly = np.array([1] + ma_coefs)
      roots = np.roots(ma_poly)
      root_mags = np.abs(roots)

      if all(root_mags > 1):
        st.success("✅ MA process is invertible")
      else:
        st.warning("⚠️ MA process is non-invertible")

  # Generate data
  np.random.seed(42)

  epsilon = np.random.normal(0, noise_std, n_sim + max(p, q) + 10)
  y = np.zeros(n_sim + max(p, q) + 10)

  for t in range(max(p, q), n_sim + max(p, q)):
    # AR component
    ar_sum = sum(ar_coefs[i] * y[t - i - 1] for i in range(p)) if p > 0 else 0

    # MA component
    ma_sum = sum(ma_coefs[i] * epsilon[t - i - 1] for i in range(q)) if q > 0 else 0

    y[t] = ar_sum + epsilon[t] + ma_sum

  # Remove burn-in
  y = y[max(p, q):]

  # Apply differencing
  if d > 0:
    y_original = y.copy()
    for _ in range(d):
      y = np.diff(y)
  else:
    y_original = y

  # Analysis
  st.markdown("### Simulated Series")

  fig = make_subplots(rows=2, cols=1,
                      subplot_titles=('Original Series' if d == 0 else 'After Differencing',
                                      'ACF and PACF'))

  fig.add_trace(go.Scatter(y=y, mode='lines', name='Series',
                           line=dict(color='darkblue')), row=1, col=1)

  # Calculate ACF and PACF
  acf_vals = acf(y, nlags=min(40, len(y) // 2), fft=True)
  pacf_vals = pacf(y, nlags=min(40, len(y) // 2))
  conf = 1.96 / np.sqrt(len(y))

  fig.update_layout(height=600, showlegend=True)
  st.plotly_chart(fig, use_container_width=True)

  # Separate ACF/PACF plot
  fig2 = make_subplots(rows=1, cols=2,
                       subplot_titles=('ACF', 'PACF'))

  fig2.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals,
                        marker_color='steelblue', name='ACF'), row=1, col=1)
  fig2.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=1)
  fig2.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=1)

  fig2.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals,
                        marker_color='darkgreen', name='PACF'), row=1, col=2)
  fig2.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=2)
  fig2.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=2)

  fig2.update_xaxes(title_text="Lag", row=1, col=1)
  fig2.update_xaxes(title_text="Lag", row=1, col=2)
  fig2.update_layout(height=400, showlegend=False)
  st.plotly_chart(fig2, use_container_width=True)

  # Analysis
  st.markdown("### Pattern Analysis")

  sig_acf = np.where(np.abs(acf_vals[1:]) > conf)[0] + 1
  sig_pacf = np.where(np.abs(pacf_vals[1:]) > conf)[0] + 1

  col1, col2, col3 = st.columns(3)

  with col1:
    st.markdown("**True Model:**")
    st.write(f"ARIMA({p}, {d}, {q})")
    if p > 0:
      st.write(f"AR coefs: {[f'{c:.2f}' for c in ar_coefs]}")
    if q > 0:
      st.write(f"MA coefs: {[f'{c:.2f}' for c in ma_coefs]}")

  with col2:
    st.markdown("**ACF Pattern:**")
    if len(sig_acf) > 0:
      st.write(f"Significant lags: {sig_acf[:5].tolist()}")
      if len(sig_acf) <= q + 2 and q > 0:
        st.success(f"✅ Cutoff near q={q}")
      elif len(sig_acf) > 5:
        st.info("Gradual decay (AR pattern)")
    else:
      st.write("No significant lags")

  with col3:
    st.markdown("**PACF Pattern:**")
    if len(sig_pacf) > 0:
      st.write(f"Significant lags: {sig_pacf[:5].tolist()}")
      if len(sig_pacf) <= p + 2 and p > 0:
        st.success(f"✅ Cutoff near p={p}")
      elif len(sig_pacf) > 5:
        st.info("Gradual decay (MA pattern)")
    else:
      st.write("No significant lags")

  # Statistical tests
  st.markdown("### Statistical Tests")

  col1, col2 = st.columns(2)

  with col1:
    adf_sim = adfuller(y)
    st.markdown("**ADF Test:**")
    st.write(f"Statistic: {adf_sim[0]:.4f}")
    st.write(f"p-value: {adf_sim[1]:.4f}")
    if adf_sim[1] < 0.05:
      st.success("✅ Stationary")
    else:
      st.error("❌ Non-stationary")

  with col2:
    st.markdown("**Summary Statistics:**")
    st.write(f"Mean: {np.mean(y):.4f}")
    st.write(f"Std: {np.std(y):.4f}")
    st.write(f"Skewness: {stats.skew(y):.4f}")
    st.write(f"Kurtosis: {stats.kurtosis(y):.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 Complete ARMA & ARIMA Identification Guide</p>
    <p>For educational purposes - Understanding Time Series Model Selection</p>
</div>
""", unsafe_allow_html=True)