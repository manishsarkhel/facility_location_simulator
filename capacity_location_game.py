# -*- coding: utf-8 -*-
"""
Facility Location & Network Design Simulator — Quick-Commerce Dark-Store Edition
==================================================================================
Teaching companion for: "Why does Blinkit's dark-store model work in Chandigarh
and Dehradun, but not in Paonta Sahib?" — a facility-location / network-design
exercise built around demand density, delivery service-radius constraints, and
dark-store unit economics.

Author: Manish Sarkhel, IIM Sirmaur (PGPEX-LSM)

WHAT CHANGED FROM THE ORIGINAL SCRIPT (and why):
1. BUG FIX — coordinate projection: the original ran KMeans directly on raw
   (lat, lon) and then multiplied the resulting "distances" by a cost-per-km
   figure. But 1 degree of longitude does not equal 1 degree of latitude in
   real km (it shrinks by cos(latitude)). That silently distorted every
   facility placement and every cost number. This version projects points to
   a local (x, y) km grid before clustering, and un-projects facility
   locations back to lat/lon only for drawing the map.
2. BUG FIX — units: the original computed `distances` in *degree* units from
   kmeans.transform() but then treated them as km. Now transform() operates
   on the km-projected coordinates, so distances are genuinely in km.
3. NEW — demand is density-weighted and shape-aware, not uniform-random in a
   circle. A compact city (Chandigarh) concentrates demand near a centre; a
   highway/corridor town (Paonta Sahib) spreads demand along an axis. This is
   the geometric heart of the teaching point: density AND shape both matter.
4. NEW — a delivery *service radius* (the 10-15 minute promise). Demand
   outside this radius from every facility is NOT captured by the
   quick-commerce channel, regardless of how "optimal" the facility
   locations are. This is what the original cost-minimizing KMeans model
   could never show.
5. NEW — dark-store breakeven economics (fixed cost, contribution margin per
   order) so viability is a number students can check, not a vibe.
6. NEW — a three-city "Compare" mode with a shared bar chart against the
   breakeven line, and discussion prompts for class.

NOTE ON DATA: city population, penetration, and cost figures below are
ILLUSTRATIVE planning assumptions for classroom use, not verified census or
company data. Everything is editable in the sidebar — encourage students to
challenge the assumptions and re-run.

To run:
    pip install streamlit numpy pandas plotly scikit-learn folium streamlit-folium
    streamlit run facility_location_sim.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="Facility Location: Quick-Commerce Edition", layout="wide")

# ----------------------------------------------------------------------------
# 1. GEOGRAPHY HELPERS  (equirectangular projection — good enough at city scale)
# ----------------------------------------------------------------------------

KM_PER_DEG_LAT = 111.0


def project_to_km(lat, lon, center_lat, center_lon):
    """Convert lat/lon to local (x, y) km coordinates around a centre point."""
    y = (lat - center_lat) * KM_PER_DEG_LAT
    x = (lon - center_lon) * KM_PER_DEG_LAT * np.cos(np.radians(center_lat))
    return x, y


def unproject_from_km(x, y, center_lat, center_lon):
    """Inverse of project_to_km: local km coordinates back to lat/lon."""
    lat = center_lat + y / KM_PER_DEG_LAT
    lon = center_lon + x / (KM_PER_DEG_LAT * np.cos(np.radians(center_lat)))
    return lat, lon


def zoom_for_radius(radius_km):
    if radius_km <= 8:
        return 12
    elif radius_km <= 12:
        return 11
    return 10


# ----------------------------------------------------------------------------
# 2. CITY PRESETS — the three-city teaching narrative
# ----------------------------------------------------------------------------

CITY_PRESETS = {
    "Chandigarh — dense, planned grid": {
        "lat": 30.7333, "lon": 76.7794, "radius_km": 10,
        "population": 1_055_000, "penetration": 0.35, "orders_per_user_day": 0.25,
        "shape": "radial", "concentration": 0.65,
        "story": ("Compact, planned city with a dense core, high smartphone/UPI "
                  "penetration and short average distances — near-ideal geography "
                  "for a 10-minute delivery promise."),
    },
    "Dehradun — larger, but urban core still dense": {
        "lat": 30.3165, "lon": 78.0322, "radius_km": 13,
        "population": 800_000, "penetration": 0.28, "orders_per_user_day": 0.20,
        "shape": "radial", "concentration": 0.80,
        "story": ("More sprawl than Chandigarh, but the urban core is still dense "
                  "enough to support several viable dark stores near the centre."),
    },
    "Paonta Sahib — small, dispersed corridor town": {
        "lat": 30.4359, "lon": 77.6248, "radius_km": 8,
        "population": 35_000, "penetration": 0.12, "orders_per_user_day": 0.12,
        "shape": "corridor", "concentration": 1.0,
        "story": ("Small industrial town strung along a highway/river-valley "
                  "corridor rather than a compact core; lower digital penetration "
                  "and far fewer households per sq km than a state-capital market."),
    },
}

# ----------------------------------------------------------------------------
# 3. DEMAND GENERATION — density-weighted, shape-aware
# ----------------------------------------------------------------------------


def generate_demand_points(n_points, radius_km, shape="radial", concentration=0.7):
    """
    Sample n_points demand locations in local km coords, centred at (0, 0).

    shape:
        'radial'   — concentric city: points cluster toward the centre.
        'corridor' — linear highway-town: points spread along an axis with a
                     narrow lateral band, NOT concentrated at one centre.
    concentration: 0 = uniform over the disc, 1 = strongly pulled toward the
                   centre. Only used for 'radial' shape.
    """
    if shape == "radial":
        power = 1.0 + concentration  # >1 pulls samples toward r = 0
        r = radius_km * (np.random.random(n_points) ** power)
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    else:  # corridor
        along = np.random.uniform(-radius_km, radius_km, n_points)
        lateral_sigma = radius_km * 0.12
        lateral = np.random.normal(0, lateral_sigma, n_points)
        angle = np.radians(35)  # arbitrary corridor bearing
        x = along * np.cos(angle) - lateral * np.sin(angle)
        y = along * np.sin(angle) + lateral * np.cos(angle)
    return x, y


# ----------------------------------------------------------------------------
# 4. FACILITY LOCATION MODEL
# ----------------------------------------------------------------------------


def solve_facility_location(x, y, weights, n_facilities, seed=42):
    """
    Weighted KMeans in km-space. Fixes the original bug of clustering on raw
    lat/lon (which distorts distances since 1 degree of longitude != 1 degree
    of latitude in real km away from the equator).
    """
    n_facilities = max(1, min(n_facilities, len(x)))
    pts = np.column_stack([x, y])
    km = KMeans(n_clusters=n_facilities, random_state=seed, n_init=10)
    km.fit(pts, sample_weight=weights)
    dist_matrix = km.transform(pts)  # genuinely in km, since input was km
    nearest_dist = np.min(dist_matrix, axis=1)
    assignment = km.labels_
    return km.cluster_centers_, nearest_dist, assignment


# ----------------------------------------------------------------------------
# 5. STREAMLIT APP
# ----------------------------------------------------------------------------

st.title("📍 Facility Location & Network Design: Quick-Commerce Edition")
st.caption(
    "Why does Blinkit's dark-store model work in Chandigarh and Dehradun, but "
    "not in Paonta Sahib? Adjust demand density, the delivery-radius promise, "
    "and dark-store economics below, and see where the model breaks."
)

st.sidebar.header("1. Scenario")
mode = st.sidebar.radio("Mode", ["Single scenario", "Compare all three cities"])

if mode == "Single scenario":
    scenario_name = st.sidebar.selectbox("City scenario", list(CITY_PRESETS.keys()))
    scenarios_to_run = {scenario_name: CITY_PRESETS[scenario_name]}
else:
    scenarios_to_run = CITY_PRESETS

n_facilities = st.sidebar.slider("Number of dark stores", 1, 10, 3)
n_points_slider = st.sidebar.slider(
    "Demand sample points (simulation resolution)", 50, 400, 200, 25,
    help="More points = smoother density pattern. Total order volume stays "
         "fixed by population × penetration × orders/user — this slider only "
         "controls simulation resolution, not demand itself."
)

st.sidebar.header("2. Quick-commerce delivery promise")
service_radius = st.sidebar.slider(
    "Delivery service radius (km)", 1.0, 6.0, 2.0, 0.25,
    help="The distance within which the platform commits to fast delivery. "
         "Demand outside this radius from every facility is NOT captured by "
         "the quick-commerce channel."
)

st.sidebar.header("3. Dark-store economics (₹)")
fixed_cost_month = st.sidebar.number_input("Fixed cost per store / month (₹)", value=450_000, step=25_000)
contribution_per_order = st.sidebar.number_input("Contribution margin per order (₹)", value=30, step=1,
    help="Revenue per order minus variable delivery/packing cost.")

daily_fixed_cost = fixed_cost_month / 30
breakeven_orders = daily_fixed_cost / contribution_per_order
st.sidebar.metric("→ Breakeven orders / store / day", f"{breakeven_orders:,.0f}")


def render_scenario(name, cfg, n_facilities, n_points, service_radius, breakeven_orders, daily_fixed_cost, contribution_per_order):
    st.subheader(name)
    st.write(cfg["story"])

    x, y = generate_demand_points(n_points, cfg["radius_km"], cfg["shape"], cfg.get("concentration", 0.7))
    total_orders_day = cfg["population"] * cfg["penetration"] * cfg["orders_per_user_day"]
    weight_per_point = total_orders_day / n_points
    weights = np.full(n_points, weight_per_point)

    centers, nearest_dist, assignment = solve_facility_location(x, y, weights, n_facilities)

    covered = nearest_dist <= service_radius
    captured_orders = weights[covered].sum()
    lost_orders = total_orders_day - captured_orders
    coverage_pct = 100 * captured_orders / total_orders_day if total_orders_day > 0 else 0

    store_rows = []
    for i in range(len(centers)):
        store_mask = (assignment == i) & covered
        store_orders = weights[store_mask].sum()
        profit = store_orders * contribution_per_order - daily_fixed_cost
        store_rows.append({
            "Store": f"Store {i + 1}",
            "Orders/day (within radius)": round(store_orders),
            "Viable?": "✅ Yes" if store_orders >= breakeven_orders else "❌ No",
            "Daily profit (₹)": round(profit),
        })
    store_df = pd.DataFrame(store_rows)

    lat0, lon0 = cfg["lat"], cfg["lon"]
    m = folium.Map(location=[lat0, lon0], zoom_start=zoom_for_radius(cfg["radius_km"]))
    for xi, yi, cov in zip(x, y, covered):
        lat_p, lon_p = unproject_from_km(xi, yi, lat0, lon0)
        folium.CircleMarker(
            [lat_p, lon_p], radius=2,
            color="#2E86AB" if cov else "#B0B0B0",
            fill=True, fill_opacity=0.6
        ).add_to(m)
    for i, (cx, cy) in enumerate(centers):
        lat_c, lon_c = unproject_from_km(cx, cy, lat0, lon0)
        store_orders = store_df.loc[i, "Orders/day (within radius)"]
        viable = store_orders >= breakeven_orders
        folium.Marker(
            [lat_c, lon_c],
            icon=folium.Icon(color="green" if viable else "red", icon="home", prefix="fa"),
            popup=f"Store {i + 1}: {store_orders} orders/day"
        ).add_to(m)
        folium.Circle(
            [lat_c, lon_c], radius=service_radius * 1000,
            color="green" if viable else "red", fill=False, weight=1, dash_array="4"
        ).add_to(m)
    folium_static(m, height=420)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total orders/day (city)", f"{total_orders_day:,.0f}")
    c2.metric("Captured within delivery radius", f"{captured_orders:,.0f}", f"{coverage_pct:.0f}% coverage")
    c3.metric("Lost / unserved demand", f"{lost_orders:,.0f}")
    network_profit = store_df["Daily profit (₹)"].sum()
    c4.metric("Network daily profit (₹)", f"{network_profit:,.0f}")

    st.dataframe(store_df, use_container_width=True)

    return {
        "name": name,
        "total_orders": total_orders_day,
        "captured": captured_orders,
        "coverage_pct": coverage_pct,
        "avg_orders_per_store": store_df["Orders/day (within radius)"].mean(),
        "n_viable": (store_df["Viable?"] == "✅ Yes").sum(),
        "n_stores": len(store_df),
        "network_profit": network_profit,
    }


summary_rows = []
for name, cfg in scenarios_to_run.items():
    result = render_scenario(
        name, cfg, n_facilities, n_points_slider, service_radius,
        breakeven_orders, daily_fixed_cost, contribution_per_order
    )
    summary_rows.append(result)
    st.divider()

# ----------------------------------------------------------------------------
# 6. CROSS-CITY TAKEAWAY (only meaningful with more than one scenario)
# ----------------------------------------------------------------------------
if len(summary_rows) > 1:
    st.header("📊 The Cross-City Takeaway")
    summary_df = pd.DataFrame(summary_rows)[
        ["name", "avg_orders_per_store", "n_viable", "n_stores", "coverage_pct", "network_profit"]
    ]
    summary_df.columns = ["City", "Avg orders/store/day", "Viable stores", "Total stores",
                           "Demand coverage %", "Network daily profit (₹)"]
    st.dataframe(summary_df, use_container_width=True)

    fig = px.bar(summary_df, x="City", y="Avg orders/store/day",
                 title="Average orders captured per store per day vs breakeven line")
    fig.add_hline(y=breakeven_orders, line_dash="dash", line_color="red",
                  annotation_text="Breakeven threshold")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Discussion prompts for class:**
1. Try increasing the number of dark stores in Paonta Sahib. Does adding facilities fix
   the problem, or does it just divide a small demand pie into smaller, less viable slices?
2. Compare the *shape* of demand in Paonta Sahib (a highway corridor) with Chandigarh's
   compact radial city. Why does geometry — not just population — change how many
   facilities you need to cover a given service radius?
3. What would have to change in Paonta Sahib for a 10-minute delivery model to become
   viable — penetration, order frequency, service radius, or the store's cost structure?
4. This is a simplified teaching model — which real-world factors (actual road network vs.
   straight-line distance, rider availability, SKU assortment, cold-chain needs) would
   change the answer?
""")
