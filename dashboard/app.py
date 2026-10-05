import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from data_loader import get_dashboard_data
from dashboard.logger import log_prediction

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="SolarSentinel-X",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# Dashboard Constants
# ==================================================

AUTO_REFRESH_INTERVAL = 30 * 1000

BACKGROUND_COLOR = "#0E1117"
GRID_COLOR = "#333333"
ACCENT_GREEN = "#00FF99"

GAUGE_HEIGHT = 350
GRAPH_HEIGHT = 450
SHAP_HEIGHT = 500

# ==================================================
# Status Constants
# ==================================================

STATUS_HEALTHY = "Healthy"
STATUS_WARNING = "Warning"
STATUS_CRITICAL = "Critical"

OVERALL_HEALTHY = "HEALTHY"
OVERALL_WARNING = "WARNING"
OVERALL_CRITICAL = "CRITICAL"

# --------------------------------------------------
# Auto Refresh (Every 30 Seconds)
# --------------------------------------------------

st_autorefresh(
    interval=AUTO_REFRESH_INTERVAL,  # 30 seconds
    key="dashboard_refresh"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
    color:white;
}

.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
}

div[data-testid="metric-container"]{
    background:#1b1f2a;
    border:1px solid #2d3748;
    padding:15px;
    border-radius:12px;
}

h1,h2,h3,h4{
    color:white;
}

hr{
    border-color:#333;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("☀️ SolarSentinel-X")

st.sidebar.markdown("---")

# ----------------------------------------
# Live Alert Status
# ----------------------------------------

dashboard_data = get_dashboard_data()
log_prediction(dashboard_data)

alert = dashboard_data["alert"]

if alert["level"] == "INFO":

    st.sidebar.success("🟢 System Healthy")

elif alert["level"] == "WARNING":

    st.sidebar.warning("🟡 Warning")

elif alert["level"] == "CRITICAL":

    st.sidebar.error("🔴 Critical")

else:

    st.sidebar.info("Unknown Status")

st.sidebar.write(alert["message"])

st.sidebar.markdown("---")

st.sidebar.subheader("🕒 Last Updated")

st.sidebar.markdown(
    f"""
**Date**
{dashboard_data["last_updated_date"]}

**Time**
{dashboard_data["last_updated_time"]}
"""
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard"
    ]
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("☀️ SolarSentinel-X")

st.caption(
    "Explainable AI-Driven Predictive Digital Twin Framework for Autonomous Solar Microgrid Health Management"
)

st.markdown("---")

# --------------------------------------------------
# Top KPI Cards
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    health = dashboard_data["health"]

    if health == STATUS_HEALTHY:
        delta = "🟢 Healthy"

    elif health == STATUS_WARNING:
        delta = "🟡 Warning"

    else:
        delta = "🔴 Critical"

    st.metric(
        label="System Health",
        value=health,
        delta=delta
    )
    
with col2:

    soc = dashboard_data["soc"]

    if soc >= 50:
        delta = "🟢 Good"

    elif soc >= 20:
        delta = "🟡 Medium"

    else:
        delta = "🔴 Low"

    st.metric(
        label="Battery SOC",
        value=f"{soc}%",
        delta=delta
    )

with col3:

    soh = dashboard_data["soh"]

    if soh >= 80:
        delta = "🟢 Healthy"

    elif soh >= 60:
        delta = "🟡 Aging"

    else:
        delta = "🔴 Replace"

    st.metric(
        label="Battery SOH",
        value=f"{soh}%",
        delta=delta
    )

with col4:

    rul = dashboard_data["rul"]

    if rul >= 2500:
        delta = "🟢 Excellent"

    elif rul >= 1200:
        delta = "🟡 Moderate"

    else:
        delta = "🔴 Low"

    st.metric(
        label="Battery RUL",
        value=f"{rul} Cycles",
        delta=delta
    )

st.markdown("---")

# ==================================================
# AI Prediction Results
# ==================================================

st.subheader("🧠 AI Prediction Results")

col1, col2 = st.columns(2)

with col1:

    st.success("🤖 Random Forest")

    health = dashboard_data["health"]
    confidence = dashboard_data["health_confidence"]

    if health == STATUS_HEALTHY:
        badge = "🟢 HEALTHY"

    elif health == STATUS_WARNING:
        badge = "🟡 WARNING"

    else:
        badge = "🔴 CRITICAL"

    st.markdown(f"### {badge}")

    st.progress(confidence)

    st.markdown(f"**Model Confidence:** {confidence}%")

    st.caption("Prediction Status : Successful")

with col2:

    st.info("🧠 Autoencoder")

    fault = dashboard_data["unknown_fault"]
    error = dashboard_data["reconstruction_error"]

    if fault:
        badge = "🔴 UNKNOWN FAULT DETECTED"
        status = "Anomaly Detected"
    else:
        badge = "🟢 NO UNKNOWN FAULT"
        status = "Normal Behaviour"

    st.markdown(f"### {badge}")

    st.progress(min(int(error * 100), 100))

    st.markdown(f"**Reconstruction Error:** {error:.3f}")

    st.caption(f"Detection Status : {status}")

st.markdown("")

col3, col4 = st.columns(2)

with col3:

    st.warning("📈 LSTM Forecast")

    soc = dashboard_data["soc"]
    soh = dashboard_data["soh"]
    rul = dashboard_data["rul"]

    st.write(f"🔋 **Battery SOC:** {soc}%")
    st.write(f"❤️ **Battery SOH:** {soh}%")
    st.write(f"♻️ **Battery RUL:** {rul} Cycles")

    st.caption("Forecast Status : Completed")

with col4:

    st.success("🌐 Digital Twin")

    twin = dashboard_data["digital_twin"]

    if twin == "Synced":
        badge = "🟢 TWIN SYNCED"
    else:
        badge = "🟡 TWIN UPDATING"

    st.markdown(f"### {badge}")

    st.write("📡 **Sensor Status:** Online")
    st.write("🤖 **Prediction:** Updated")
    st.write("⚙️ **Simulation:** Running")

    st.caption("Digital Twin Status : Active")
    
st.markdown("---")

sensor = dashboard_data["sensor_values"]

# ==================================================
# Live Sensor Values
# ==================================================

st.subheader("📡 Live Sensor Values")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Panel Voltage",
        f'{sensor["Panel_Voltage"]:.2f} V'
    )

with col2:
    st.metric(
        "Panel Current",
        f'{sensor["Panel_Current"]:.2f} A'
    )

with col3:
    st.metric(
        "Panel Power",
        f'{sensor["Panel_Power"]:.2f} W'
    )

with col4:
    st.metric(
        "Battery Voltage",
        f'{sensor["Battery_Voltage"]:.2f} V'
    )


col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        "Battery Current",
        f'{sensor["Battery_Current"]:.2f} A'
    )

with col6:
    st.metric(
        "Battery Temperature",
        f'{sensor["Battery_Temperature"]:.2f} °C'
    )

with col7:
    st.metric(
        "Panel Temperature",
        f'{sensor["Panel_Temperature"]:.2f} °C'
    )

with col8:
    st.metric(
        "Load Power",
        f'{sensor["Load_Power"]:.2f} W'
    )

st.markdown("---")

# ==================================================
# Battery Health Gauges
# ==================================================

st.subheader("🔋 Battery Health Overview")

col1, col2 = st.columns(2)

with col1:

    fig_soc = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=dashboard_data["soc"],
            title={"text": "Battery SOC (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "limegreen"},

                "steps": [
                    {"range": [0, 20], "color": "#ff4d4d"},
                    {"range": [20, 50], "color": "#ffcc00"},
                    {"range": [50, 100], "color": "#00cc66"}
                ]
            }
        )
    )

    fig_soc.update_layout(
        height=GAUGE_HEIGHT,
        paper_bgcolor=BACKGROUND_COLOR,
        font_color="white"
    )

    st.plotly_chart(
        fig_soc,
        width="stretch"
    )

with col2:

    fig_soh = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=dashboard_data["soh"],
            title={"text": "Battery SOH (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "deepskyblue"},

                "steps": [
                    {"range": [0, 60], "color": "#ff4d4d"},
                    {"range": [60, 80], "color": "#ffcc00"},
                    {"range": [80, 100], "color": "#00cc66"}
                ]
            }
        )
    )

    fig_soh.update_layout(
        height=GAUGE_HEIGHT,
        paper_bgcolor=BACKGROUND_COLOR,
        font_color="white"
    )

    st.plotly_chart(
        fig_soh,
        width="stretch"
    )

st.markdown("---")

# ==================================================
# Solar Power Generation
# ==================================================

st.subheader("📈 Solar Power Generation")

history = dashboard_data["history"]

df = history[["Hour", "Panel_Power"]].copy()

df.rename(
    columns={"Panel_Power": "Power (W)"},
    inplace=True
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Hour"],
        y=df["Power (W)"],
        mode="lines+markers",
        name="Solar Power",
        line=dict(
            color=ACCENT_GREEN,
            width=4,
            shape="spline"
        ),
        marker=dict(
            size=8,
            color=ACCENT_GREEN
        ),
        hovertemplate=
        "<b>Hour</b>: %{x}<br>"
        "<b>Power</b>: %{y:.2f} W"
        "<extra></extra>"
    )
)

fig.update_layout(

    title="☀️ Solar Power Output Throughout the Day",

    height=GRAPH_HEIGHT,

    paper_bgcolor=BACKGROUND_COLOR,
    plot_bgcolor=BACKGROUND_COLOR,

    font_color="white",

    hovermode="x unified",

    xaxis=dict(
        title="Hour of Day",
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False
    ),

    yaxis=dict(
        title="Power (Watts)",
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),

    margin=dict(
        l=40,
        r=40,
        t=60,
        b=40
    )
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.markdown("---")

# ==================================================
# System Status & Recommendations
# ==================================================

st.subheader("🛡️ System Status & Recommendations")

col1, col2 = st.columns(2)

with col1:

    st.success("Overall System Status")

    status = dashboard_data["overall_status"]

    if status == OVERALL_HEALTHY:
        st.write("### 🟢 HEALTHY")
        st.write("Solar microgrid operating normally.")
        st.progress(100)

    elif status == "WARNING":
        st.write("### 🟡 WARNING")
        st.write("The system requires attention.")
        st.progress(70)

    else:
        st.write("### 🔴 CRITICAL")
        st.write("Immediate maintenance required.")
        st.progress(30)

with col2:

    st.warning("💡 AI Recommendation")

    recommendation = dashboard_data["recommendation"]

    st.markdown(
        f"""
<div style="
background:#1b1f2a;
padding:16px;
border-radius:10px;
border-left:5px solid #00cc66;
">

<h4 style="margin-top:0; margin-bottom:12px;">
Recommendation
</h4>

<p style="
font-size:16px;
margin-bottom:18px;
line-height:1.6;
">
{recommendation}
</p>

<hr style="border:1px solid #333;">

<p style="
font-size:15px;
margin:0;
">
<b>Priority:</b>
<span style="color:#00cc66;">LOW</span>
</p>

</div>
""",
        unsafe_allow_html=True
    )

st.markdown("---")

# ==================================================
# SHAP Feature Importance
# ==================================================

shap_df = dashboard_data["shap_data"]

features = shap_df["Feature"]

importance = shap_df["Importance"]

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=importance,
        y=features,
        orientation="h",

        marker=dict(
            color=ACCENT_GREEN
        ),

        text=[f"{v:.3f}" for v in importance],

        textposition="outside",

        hovertemplate=
        "<b>%{y}</b><br>"
        "Importance: %{x:.3f}"
        "<extra></extra>"
    )
)

fig.update_layout(

    title="🧠 SHAP Feature Importance Analysis",

    height=SHAP_HEIGHT,

    paper_bgcolor=BACKGROUND_COLOR,
    plot_bgcolor=BACKGROUND_COLOR,

    font_color="white",

    hovermode="y",

    xaxis=dict(
        title="Feature Importance",
        showgrid=True,
        gridcolor=GRID_COLOR
    ),

    yaxis=dict(
        autorange="reversed",
        title=""
    ),

    margin=dict(
        l=120,
        r=40,
        t=60,
        b=40
    )
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.markdown("---")

# ==================================================
# Footer
# ==================================================

st.markdown("---")

st.markdown(
    """
<div style="text-align:center; color:#9CA3AF; padding:20px;">

<h4 style="color:white; margin-bottom:5px;">
☀️ SolarSentinel-X v1.0
</h4>

<p style="margin:0;">
Explainable AI-Driven Predictive Digital Twin Framework
</p>

<p style="margin:5px 0 15px 0;">
for Autonomous Solar Microgrid Health Management
</p>

<p style="font-size:14px;">
B.Tech Major Project • Amrita Vishwa Vidyapeetham
</p>

<hr style="border:1px solid #333; width:35%;">

<p style="font-size:13px; color:gray;">
Developed using Random Forest • LSTM • Autoencoder • SHAP • Digital Twin • Streamlit
</p>

<p style="font-size:12px; color:#666;">
© 2026 SolarSentinel-X | All Rights Reserved
</p>

</div>
""",
    unsafe_allow_html=True
)