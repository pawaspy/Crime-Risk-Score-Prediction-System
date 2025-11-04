import streamlit as st
import pandas as pd
import requests
from streamlit_folium import st_folium
import folium
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    layout="wide", 
    page_title="Crime Risk Dashboard",
    page_icon="🚨",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        agg = pd.read_csv("models/grid_aggregated.csv")
        if "top_crime_type" in agg.columns:
            agg["top_crime_type"] = agg["top_crime_type"].astype(str)
        return agg
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

agg = load_data()

if agg is None:
    st.error("⚠️ Could not load data. Please run crime_pipeline_fixed.py first!")
    st.stop()

# Session state
if 'selected_lat' not in st.session_state:
    st.session_state.selected_lat = float(agg["lat_grid"].mean())
if 'selected_lon' not in st.session_state:
    st.session_state.selected_lon = float(agg["lon_grid"].mean())
if 'selected_hour' not in st.session_state:
    st.session_state.selected_hour = 12
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚨 Crime Risk Prediction Dashboard")
    st.caption("Visualize area risk and predict safety level for any location and time in Delhi")
with col2:
    st.metric("Total Grid Cells", f"{len(agg):,}")

# Sidebar
st.sidebar.header("🎯 Prediction Controls")

st.sidebar.subheader("📍 Location")
selected_lat = st.sidebar.number_input(
    "Latitude", 
    value=st.session_state.selected_lat, 
    step=0.001,
    format="%.4f"
)
selected_lon = st.sidebar.number_input(
    "Longitude", 
    value=st.session_state.selected_lon, 
    step=0.001,
    format="%.4f"
)

st.sidebar.subheader("⏰ Time of Day")
selected_hour = st.sidebar.slider(
    "Hour (24-hour format)", 
    0, 23, 
    st.session_state.selected_hour,
    help="Drag to select the hour of the day"
)

time_of_day = "🌙 Night" if 0 <= selected_hour <= 6 else \
              "🌅 Morning" if 7 <= selected_hour <= 11 else \
              "☀️ Afternoon" if 12 <= selected_hour <= 17 else \
              "🌆 Evening" if 18 <= selected_hour <= 20 else "🌃 Night"
st.sidebar.info(f"**{time_of_day}** ({selected_hour}:00)")

st.sidebar.subheader("🔍 Crime Type")
crime_types = sorted(agg["top_crime_type"].unique())
st.sidebar.write(f"Available: {len(crime_types)} types")

selected_crime = st.sidebar.selectbox(
    "Select Crime Type",
    crime_types,
    help="Choose a specific crime type for prediction"
)

predict_clicked = st.sidebar.button("🔮 Predict Risk Level", use_container_width=True)

st.session_state.selected_lat = selected_lat
st.session_state.selected_lon = selected_lon
st.session_state.selected_hour = selected_hour

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Map", "📊 Analytics", "🎯 Prediction Details", "📈 Insights"])

with tab1:
    st.subheader("🗺️ Crime Risk Heat Map - Click on map to update location")
    
    m = folium.Map(
        location=[selected_lat, selected_lon],
        zoom_start=12,
        tiles="cartodb dark_matter"
    )
    for _, r in agg.iterrows():
        color = "red" if r["risk_score"] > 0.66 else ("orange" if r["risk_score"] > 0.33 else "green")
    
        area_name = r["nm_pol"] if "nm_pol" in r and pd.notna(r["nm_pol"]) else "Unknown"

        popup_html = f"""
        <div style="font-family: Arial; width: 220px;">
            <h4 style="color: {color}; margin-bottom:4px;">⚠️ {r['risk_type'].upper()} RISK</h4>
            <p><b>Area:</b> {area_name}</p>
            <p><b>Score:</b> {r['risk_score']:.2f}</p>
            <p><b>Total Crimes:</b> {int(r['total_crimes'])}</p>
            <p><b>Top Crime:</b> {r['top_crime_type']}</p>
            <p><b>Lat:</b> {r['lat_grid']:.4f}, <b>Lon:</b> {r['lon_grid']:.4f}</p>
        </div>
        """

        folium.Marker(
            location=[r["lat_grid"], r["lon_grid"]],
            popup=popup_html,
            icon=folium.Icon(
                color="red" if r["risk_type"] == "high" else 
                    "orange" if r["risk_type"] == "medium" else 
                    "green"
            )
        ).add_to(m)



    
    folium.Marker(
        location=[selected_lat, selected_lon],
        popup="📍 Selected Location",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    
    m.location = [st.session_state.selected_lat, st.session_state.selected_lon]

    map_data = st_folium(m, width=None, height=600, key="main_map")
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        if clicked_lat and clicked_lon:
            # Find nearest point from aggregated data
            nearest_idx = ((agg["lat_grid"] - clicked_lat).abs() + (agg["lon_grid"] - clicked_lon).abs()).idxmin()
            nearest_row = agg.loc[nearest_idx]

            # Update session state
            st.session_state.selected_lat = float(nearest_row["lat_grid"])
            st.session_state.selected_lon = float(nearest_row["lon_grid"])
            area_name = nearest_row.get("nm_pol", "Unknown Area")

            # Show area name in sidebar (only after click)
            st.sidebar.success(f"📍 Selected Area: {area_name}")
            st.sidebar.info(f"Lat: {nearest_row['lat_grid']:.4f}, Lon: {nearest_row['lon_grid']:.4f}")

            st.rerun()

with tab2:
    st.subheader("📊 Crime Analytics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔴 High Risk", len(agg[agg["risk_type"] == "high"]))
    with col2:
        st.metric("🟠 Medium Risk", len(agg[agg["risk_type"] == "medium"]))
    with col3:
        st.metric("🟢 Low Risk", len(agg[agg["risk_type"] == "low"]))
    with col4:
        st.metric("📍 Total Crimes", f"{agg['total_crimes'].sum():,}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        crime_dist = agg["top_crime_type"].value_counts().head(10)
        fig1 = px.bar(
            x=crime_dist.values,
            y=crime_dist.index,
            orientation='h',
            title="Top 10 Crime Types by Area",
            labels={'x': 'Number of Grid Cells', 'y': 'Crime Type'},
            color=crime_dist.values,
            color_continuous_scale='Reds'
        )
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        risk_dist = agg["risk_type"].value_counts()
        fig2 = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title="Risk Level Distribution",
            color_discrete_map={'high': '#FF4B4B', 'medium': '#FFA500', 'low': '#00CC00'}
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("⏰ Crime Risk by Time of Day")
    hour_data = pd.DataFrame({
        'Hour': range(24),
        'Relative Risk': [0.3, 0.2, 0.15, 0.1, 0.1, 0.15, 0.3, 0.5, 0.6, 0.5, 0.4, 0.4,
                         0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.9, 0.7, 0.5, 0.4]
    })
    fig3 = px.line(
        hour_data,
        x='Hour',
        y='Relative Risk',
        title='Crime Risk Pattern Throughout the Day',
        markers=True
    )
    fig3.update_traces(line_color='#FF4B4B', line_width=3)
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("🎯 Prediction Results")
    
    if predict_clicked:
        payload = {
            "lat": selected_lat, 
            "lon": selected_lon, 
            "hour": selected_hour, 
            "top_crime_type": selected_crime
        }
        
        api_url = "https://crime-risk-score-prediction-system.onrender.com"
        
        with st.spinner("🔄 Analyzing crime risk..."):
            try:
                resp = requests.post(api_url, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.prediction_result = data
                else:
                    st.error(f"❌ API Error: {resp.status_code} - {resp.text}")
                    st.session_state.prediction_result = None
            except Exception as e:
                st.error(f"⚠️ Could not connect to API: {e}")
                st.error("Make sure the API is running: `uvicorn predict_api:app --reload`")
                st.session_state.prediction_result = None
    
    if st.session_state.prediction_result:
        data = st.session_state.prediction_result
        pred = data["prediction"]
        
        risk_type = pred['risk_type'].upper()
        risk_score = pred['risk_score']
        
        if risk_type == "HIGH":
            st.markdown(f'<div class="risk-high">⚠️ HIGH RISK AREA - Score: {risk_score:.2%}</div>', unsafe_allow_html=True)
        elif risk_type == "MEDIUM":
            st.markdown(f'<div class="risk-medium">⚡ MEDIUM RISK AREA - Score: {risk_score:.2%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ LOW RISK AREA - Score: {risk_score:.2%}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📍 Grid Latitude", f"{data['nearest_grid']['lat_grid']:.4f}")
        with col2:
            st.metric("📍 Grid Longitude", f"{data['nearest_grid']['lon_grid']:.4f}")
        with col3:
            st.metric("🔍 Top Crime", data['nearest_grid']['top_crime_type'])
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 24}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "darkred" if risk_score > 0.66 else "orange" if risk_score > 0.33 else "green"},
                'steps': [
                    {'range': [0, 33], 'color': 'lightgreen'},
                    {'range': [33, 66], 'color': 'lightyellow'},
                    {'range': [66, 100], 'color': 'lightcoral'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 66
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("💡 Safety Recommendations")
        if risk_type == "HIGH":
            st.warning("""
            - 🚨 **Exercise extreme caution** in this area
            - 👥 Avoid traveling alone, especially at night
            - 📱 Keep emergency contacts readily available
            - 🚗 Use well-lit main roads and avoid shortcuts
            """)
        elif risk_type == "MEDIUM":
            st.info("""
            - ⚡ **Stay alert** and aware of your surroundings
            - 🌙 Avoid late-night travel if possible
            - 👥 Travel in groups when feasible
            """)
        else:
            st.success("""
            - ✅ **Relatively safe** area
            - 👍 Standard safety precautions recommended
            - 📱 Stay aware of your surroundings
            """)
    else:
        st.info("👆 Click 'Predict Risk Level' in the sidebar to see results")

with tab4:
    st.subheader("📈 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 High Risk Hot Spots")
        high_risk = agg[agg["risk_type"] == "high"].nlargest(5, "risk_score")
        for idx, row in high_risk.iterrows():
            st.markdown(f"""
            **Location:** {row['lat_grid']:.4f}, {row['lon_grid']:.4f}  
            **Risk Score:** {row['risk_score']:.2%}  
            **Top Crime:** {row['top_crime_type']}  
            **Total Crimes:** {row['total_crimes']}
            ---
            """)
    
    with col2:
        st.markdown("### ✅ Safe Zones")
        low_risk = agg[agg["risk_type"] == "low"].nsmallest(5, "risk_score")
        for idx, row in low_risk.iterrows():
            st.markdown(f"""
            **Location:** {row['lat_grid']:.4f}, {row['lon_grid']:.4f}  
            **Risk Score:** {row['risk_score']:.2%}  
            **Top Crime:** {row['top_crime_type']}  
            **Total Crimes:** {row['total_crimes']}
            ---
            """)

st.divider()
st.caption("🔒 Crime Risk Prediction Dashboard | Built with Streamlit & ML")
