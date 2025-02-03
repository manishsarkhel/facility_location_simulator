# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:52:52 2025

@author: manis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import folium
from streamlit_folium import folium_static

def generate_random_locations(n_points, center_lat, center_lon, radius_km):
    # Generate random points within radius
    r = radius_km * np.sqrt(np.random.random(n_points))
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    # Convert to lat/lon
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    
    # Convert km to degrees (approximate)
    lat = center_lat + (dy / 111)
    lon = center_lon + (dx / (111 * np.cos(np.radians(center_lat))))
    
    return lat, lon

def calculate_costs(n_facilities, demand_points, facility_cost, transport_cost_per_km):
    # Use KMeans to optimize facility locations
    kmeans = KMeans(n_clusters=n_facilities, random_state=42)
    kmeans.fit(demand_points)
    
    # Calculate distances to nearest facilities
    distances = np.min(kmeans.transform(demand_points), axis=1)
    
    # Calculate total costs
    total_transport_cost = np.sum(distances) * transport_cost_per_km
    total_facility_cost = n_facilities * facility_cost
    
    return (kmeans.cluster_centers_, 
            total_transport_cost, 
            total_facility_cost,
            distances)

st.title("Facility Location Optimization Simulator")

# Sidebar inputs
st.sidebar.header("Parameters")
center_lat = st.sidebar.number_input("Center Latitude", value=40.7128)
center_lon = st.sidebar.number_input("Center Longitude", value=-74.0060)
radius = st.sidebar.number_input("Region Radius (km)", value=10)
n_demand_points = st.sidebar.number_input("Number of Demand Points", value=100, min_value=10)
n_facilities = st.sidebar.number_input("Number of Facilities", value=5, min_value=1)
facility_cost = st.sidebar.number_input("Facility Opening Cost", value=1000)
transport_cost = st.sidebar.number_input("Transport Cost per km", value=10)

# Generate random demand points
demand_lats, demand_lons = generate_random_locations(
    n_demand_points, center_lat, center_lon, radius
)
demand_points = np.column_stack([demand_lats, demand_lons])

# Calculate optimal facility locations and costs
facility_locations, transport_costs, facility_costs, distances = calculate_costs(
    n_facilities, demand_points, facility_cost, transport_cost
)

# Create map
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# Plot demand points
for lat, lon in zip(demand_lats, demand_lons):
    folium.CircleMarker(
        [lat, lon],
        radius=3,
        color="blue",
        fill=True
    ).add_to(m)

# Plot facility locations
for lat, lon in facility_locations:
    folium.Marker(
        [lat, lon],
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

# Display map
st.write("### Map of Demand Points and Facility Locations")
folium_static(m)

# Display costs
st.write("### Cost Analysis")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Transport Cost", f"${transport_costs:,.2f}")
with col2:
    st.metric("Facility Cost", f"${facility_costs:,.2f}")
with col3:
    st.metric("Total Cost", f"${(transport_costs + facility_costs):,.2f}")

# Plot cost distribution
df_costs = pd.DataFrame({
    'Distance to Nearest Facility (km)': distances,
    'Count': 1
})

fig = px.histogram(
    df_costs, 
    x='Distance to Nearest Facility (km)',
    title='Distribution of Distances to Nearest Facility'
)
st.plotly_chart(fig)