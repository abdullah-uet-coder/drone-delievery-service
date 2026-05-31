import streamlit as st
import pandas as pd
import pydeck as pdk
import math
import time

# =====================================================================
# 1. APP CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    layout="wide", 
    page_title="Drone ATC Dashboard", 
    page_icon="🛸"
)

# Custom CSS injection for professional dark-mode UI enhancement
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_with_html=True)

st.title("🛸 Autonomous Drone Delivery Air-Traffic Control Terminal")
st.caption("ICT in Transportation | Advanced Multi-Agent Flight Management System")
st.write("---")

# =====================================================================
# 2. SEED DATA & DATABASE SIMULATION (No-Fly Zones)
# =====================================================================
# Coordinates centered around a busy urban logistics hub region
BASE_LAT = 33.6500
BASE_LON = 73.0500

# Defining geometric circular boundaries for restricted airspaces
NO_FLY_ZONES = [
    {"name": "International Airport Airspace (Zone Alpha)", "lat": 33.6844, "lon": 73.0479, "radius_deg": 0.022, "alt_floor": 0, "alt_ceiling": 1000},
    {"name": "Military Cantonment (Zone Bravo)", "lat": 33.6007, "lon": 73.0678, "radius_deg": 0.015, "alt_floor": 0, "alt_ceiling": 1500},
    {"name": "Downtown High-Rise Grid (Zone Charlie)", "lat": 33.6410, "lon": 73.0790, "radius_deg": 0.009, "alt_floor": 0, "alt_ceiling": 300},
]

# Initialize Session State Variables to store persistent data
if 'flight_log' not in st.session_state:
    st.session_state['flight_log'] = []
if 'sim_running' not in st.session_state:
    st.session_state['sim_running'] = False

# =====================================================================
# 3. CORE CORE ENGINE LOGIC (Math, Physics & Geo-fencing)
# =====================================================================

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the accurate great-circle distance between two points in km."""
    R = 6371.0 # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_3d_flight_path(start_lat, start_lon, end_lat, end_lon, peak_alt=150, steps=60):
    """Generates an interpolated 3D parabolic path matrix for the flight vector."""
    points = []
    for i in range(steps + 1):
        t = i / steps
        # Linear spatial interpolation
        curr_lat = start_lat + (end_lat - start_lat) * t
        curr_lon = start_lon + (end_lon - start_lon) * t
        
        # Parabolic curve modeling flight altitude profile
        altitude = peak_alt * (1 - 4 * (t - 0.5)**2)
        altitude = max(0.0, altitude) # Ensure no sub-surface telemetry
        
        points.append([curr_lon, curr_lat, altitude])
    return points

def evaluate_geofence_breach(path_points):
    """Mathematical collision detection matrix checking path vectors against NFZ coordinates."""
    breached_zones = []
    for pt in path_points:
        lon, lat, alt = pt
        for zone in NO_FLY_ZONES:
            # Check 2D radial footprint boundary
            distance = math.sqrt((lat - zone["lat"])**2 + (lon - zone["lon"])**2)
            if distance <= zone["radius_deg"]:
                # Check 3D altitude envelope validation
                if zone["alt_floor"] <= alt <= zone["alt_ceiling"]:
                    if zone["name"] not in breached_zones:
                        breached_zones.append(zone["name"])
    return breached_zones

def compute_advanced_battery_drain(distance_km, wind_spd, wind_dir, flight_dir, payload_kg, drone_type):
    """
    Mechanical Engineering drag and battery depletion model.
    Accounts for payload mass, headwind resistance, and structural aircraft specifications.
    """
    # Drone configuration matrix constants
    spec_sheet = {
        "Quadcopter (Lightweight)": {"base_eff": 3.0, "mass": 2.5, "drag_coeff": 0.42},
        "Hexacopter (Heavy-Lift)": {"base_eff": 4.5, "mass": 6.8, "drag_coeff": 0.55}
    }
    
    cfg = spec_sheet[drone_type]
    
    # Calculate angular wind resistance vectors (relative wind direction)
    angle_rad = math.radians(abs(wind_dir - flight_dir))
    relative_wind_impact = wind_spd * math.cos(angle_rad) # Headwind yields positive drag acceleration
    
    # Mathematical expression representing aerodynamic and mass energy consumption
    mass_penalty = 0.35 * (payload_kg / cfg["mass"])
    drag_penalty = 0.08 * (relative_wind_impact * cfg["drag_coeff"]) if relative_wind_impact > 0 else 0.02 * relative_wind_impact
    
    energy_per_km = cfg["base_eff"] + mass_penalty + drag_penalty
    total_consumption = energy_per_km * distance_km
    
    # Constrain boundary limits safely between 1% and 100% capacity
    return max(1.0, min(100.0, round(total_consumption, 2)))

# =====================================================================
# 4. STREAMLIT INTERFACE AND CONTROL PANEL LAYOUT
# =====================================================================
col1, col2 = st.columns()

with col1:
    st.header("🎛️ Mission Dispatched Parameters")
    
    with st.expander("1. Navigation Waypoints", expanded=True):
        st.caption("Input absolute GPS coordinates for origin and target destinations.")
        start_lat = st.number_input("Origin Latitude", value=33.6350, format="%.4f")
        start_lon = st.number_input("Origin Longitude", value=73.0150, format="%.4f")
        end_lat = st.number_input("Destination Latitude", value=33.6650, format="%.4f")
        end_lon = st.number_input("Destination Longitude", value=73.0850, format="%.4f")

    with st.expander("2. Aircraft Specifications", expanded=True):
        drone_model = st.selectbox("UAV Fleet Classification", ["Quadcopter (Lightweight)", "Hexacopter (Heavy-Lift)"])
        payload_mass = st.slider("Payload Freight Cargo (kg)", 0.0, 10.0, 1.5, step=0.5)
        cruise_alt = st.slider("Target Flight Altitude Ceiling (meters)", 50, 300, 120)

    with st.expander("3. Micro-Meteorological Telemetry", expanded=True):
        wind_speed = st.slider("Atmospheric Wind Velocity (km/h)", 0, 60, 22)
        wind_direction = st.slider("Wind Vector Bearing (Degrees)", 0, 360, 120)

    # Calculate Flight Vectors
    calc_distance = calculate_haversine_distance(start_lat, start_lon, end_lat, end_lon)
    # Approximate bearing mathematically
    calc_bearing = math.degrees(math.atan2(end_lon - start_lon, end_lat - start_lat)) % 360
    
    projected_drain = compute_advanced_battery_drain(
        calc_distance, wind_speed, wind_direction, calc_bearing, payload_mass, drone_model
    )
    
    generated_path = generate_3d_flight_path(start_lat, start_lon, end_lat, end_lon, peak_alt=cruise_alt)
    airspace_breaches = evaluate_geofence_breach(generated_path)
    
    # Submission Interface Controls
    st.write("---")
    if airspace_breaches:
        st.error("🚨 MISSION REJECTED: Flight trajectory violates restricted municipal airspaces.")
        for breach in airspace_breaches:
            st.write(f"- Critical Entry Points Detected: `{breach}`")
        st.button("🚀 Core Simulator Engaged (Disabled due to Breach)", disabled=True)
    else:
        st.success("✅ MISSION COMPLIANT: Flight pathway meets active security grid regulations.")
        
        if st.button("🚀 Deploy UAV & Begin Real-Time Simulation", type="primary"):
            st.session_state['sim_running'] = True
            # Log current valid flight profile into historical data array
            st.session_state['flight_log'].append({
                "Timestamp": time.strftime("%H:%M:%S"),
                "UAV Profile": drone_model,
                "Range Vector (km)": round(calc_distance, 2),
                "Est. Battery Cost": f"{projected_drain}%",
                "Status": "COMPLETED"
            })

# =====================================================================
# 5. DATA VISUALIZATION AND MAP OVERLAYS
# =====================================================================
with col2:
    st.header("🛰️ Airspace Telemetry & Spatial Viewport")
    
    # Render High-Level Operational Metrics Dashboard Cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Calculated Vector Path", f"{round(calc_distance, 2)} km")
    m2.metric("Predicted Battery Overhead", f"{projected_drain} %")
    m3.metric("Calculated Track Bearing", f"{round(calc_bearing, 1)}° N")
    
    # Transform database arrays into Pandas structures for mapping pipelines
    nfz_data = pd.DataFrame(NO_FLY_ZONES)
    # Scale degrees into physical radius projection constants for rendering engines
    nfz_data['radius_meters'] = nfz_data['radius_deg'] * 111000 
    
    # Set up interactive multi-dimensional mapping engine structural parameters
    nfz_render_layer = pdk.Layer(
        "CylinderLayer",
        nfz_data,
        get_position=["lon", "lat"],
        get_radius="radius_meters",
        get_elevation="alt_ceiling",
        get_fill_color=,  # Semi-transparent Crimson warning matrix
        pickable=True,
        auto_highlight=True
    )
    
    path_render_layer = pdk.Layer(
        "PathLayer",
        [{"path": generated_path}],
        get_path="path",
        width_scale=15,
        width_min_pixels=4,
        get_color= if not airspace_breaches else, # Emerald green vs Amber Warning
    )
    
    # Display the Primary Map Object Canvas Environment
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/navigation-dark-v9",
        initial_view_state=pdk.ViewState(
            latitude=BASE_LAT,
            longitude=BASE_LON,
            zoom=11.5,
            pitch=50,
            bearing=-10
        ),
        layers=[nfz_render_layer, path_render_layer],
        tooltip={"text": "{name}\nOperational Altitude Window: {alt_floor}m - {alt_ceiling}m"}
    ))
    
    # Real-Time Telemetry Simulation Feedback Loop
    if st.session_state['sim_running']:
        st.write("---")
        st.subheader("📡 Live UAV Feed Processing Streams")
        progress_bar = st.progress(0)
        status_update_box = st.empty()
        
        # Simulate step-by-step progress along coordinates matrix array
        for percent_complete in range(0, 101, 20):
            time.sleep(0.4) # Artificially throttles execution speed to mimic actual flight duration frames
            progress_bar.progress(percent_complete)
            status_update_box.code(
                f"SYSTEM STATUS: UAV cruising at altitude... Coordinate Vector Step Index Progress: {percent_complete}% | "
                f"Battery Status: {round(100 - (projected_drain * (percent_complete/100)), 1)}% remaining"
            )
        status_update_box.success("🎯 Mission Success: Package delivered securely. Aircraft landed safely at destination.")
        st.session_state['sim_running'] = False

    # Historical Database Execution Logs Component Visualizer
    if st.session_state['flight_log']:
        st.write("---")
        st.subheader("🗄️ Terminal Local Flight Registry History Log")
        st.dataframe(pd.DataFrame(st.session_state['flight_log']), use_container_width=True)
