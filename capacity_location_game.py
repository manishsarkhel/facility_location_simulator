# -*- coding: utf-8 -*-
"""
Facility Location & Network Design Simulator — Quick-Commerce Dark-Store Edition
Author: Manish Sarkhel, IIM Sirmaur (PGPEX-LSM)

Fixes a coordinate-projection bug in the original script (KMeans was run on
raw lat/lon, whose degree-to-km ratio is not constant, so distances and costs
were wrong). Points are now projected to a local (x, y) km grid before
clustering and un-projected only for map display.

To run:
    pip install -r requirements.txt
    streamlit run facility_location_sim.py

See instructor_notes.md for teaching context and discussion material.
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
    y = (lat - center_lat) * KM_PER_DEG_LAT
    x = (lon - center_lon) * KM_PER_DEG_LAT * np.cos(np.radians(center_lat))
    return x, y


def unproject_from_km(x, y, center_lat, center_lon):
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
# 2. CITY PRESETS
# ----------------------------------------------------------------------------

CITY_PRESETS = {
    "Chandigarh — dense, planned grid": {
        "lat": 30.7333, "lon": 76.7794, "radius_km": 10, "seed": 1,
        "population": 1_055_000, "penetration": 0.35, "orders_per_user_day": 0.25,
        "shape": "radial", "concentration": 0.65,
        "context": ("Compact, planned city with a dense core, high smartphone/UPI "
                    "penetration and short average distances."),
    },
    "Dehradun — larger, but urban core still dense": {
        "lat": 30.3165, "lon": 78.0322, "radius_km": 13, "seed": 2,
        "population": 800_000, "penetration": 0.28, "orders_per_user_day": 0.20,
        "shape": "radial", "concentration": 0.80,
        "context": ("More sprawl than Chandigarh, but the urban core is still dense "
                    "enough to support several dark stores near the centre."),
    },
    "Paonta Sahib — small, dispersed corridor town": {
        "lat": 30.4359, "lon": 77.6248, "radius_km": 8, "seed": 3,
        "population": 35_000, "penetration": 0.12, "orders_per_user_day": 0.12,
        "shape": "corridor", "concentration": 1.0,
        "context": ("Small industrial town strung along a highway/river-valley "
                    "corridor rather than a compact core; lower digital penetration "
                    "and far fewer households per sq km than a state-capital market."),
    },
}

# ----------------------------------------------------------------------------
# 3. DEMAND GENERATION — density-weighted, shape-aware
# ----------------------------------------------------------------------------


def generate_demand_points(n_points, radius_km, shape="radial", concentration=0.7, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if shape == "radial":
        power = 1.0 + concentration
        r = radius_km * (np.random.random(n_points) ** power)
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    else:  # corridor
        along = np.random.uniform(-radius_km, radius_km, n_points)
        lateral_sigma = radius_km * 0.12
        lateral = np.random.normal(0, lateral_sigma, n_points)
        angle = np.radians(35)
        x = along * np.cos(angle) - lateral * np.sin(angle)
        y = along * np.sin(angle) + lateral * np.cos(angle)
    return x, y


# ----------------------------------------------------------------------------
# 4. FACILITY LOCATION MODEL
# ----------------------------------------------------------------------------


def solve_facility_location(x, y, weights, n_facilities, seed=42):
    n_facilities = max(1, min(n_facilities, len(x)))
    pts = np.column_stack([x, y])
    km = KMeans(n_clusters=n_facilities, random_state=seed, n_init=10)
    km.fit(pts, sample_weight=weights)
    dist_matrix = km.transform(pts)
    nearest_dist = np.min(dist_matrix, axis=1)
    assignment = km.labels_
    return km.cluster_centers_, nearest_dist, assignment


# ----------------------------------------------------------------------------
# 5. STREAMLIT APP
# ----------------------------------------------------------------------------

st.title("📍 Facility Location & Network Design: Quick-Commerce Edition")

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
    help="More points = smoother density pattern. Total order volume is fixed "
         "by population × penetration × orders/user; this only controls resolution."
)

st.sidebar.header("2. Quick-commerce delivery promise")
service_radius = st.sidebar.slider(
    "Delivery service radius (km)", 1.0, 6.0, 2.0, 0.25,
    help="Demand outside this radius from every facility is not captured by "
         "the quick-commerce channel."
)

st.sidebar.header("3. Cost structure (₹)")
fixed_cost_month = st.sidebar.number_input("Fixed cost per store / month (₹)", value=450_000, step=25_000,
    help="Rent, staff, equipment — cost of simply keeping a store open, independent of order volume.")
contribution_per_order = st.sidebar.number_input("Contribution margin per order (₹)", value=30, step=1,
    help="Revenue per order minus variable delivery/packing cost.")
transport_cost_per_order_km = st.sidebar.number_input("Delivery cost per order per km (₹)", value=5, step=1,
    help="Variable cost of physically delivering one order one km. This is what "
         "rises as stores get farther apart and falls as you add more, closer stores.")

daily_fixed_cost = fixed_cost_month / 30
breakeven_orders = daily_fixed_cost / contribution_per_order
st.sidebar.metric("→ Breakeven orders / store / day", f"{breakeven_orders:,.0f}")


def render_scenario(name, cfg, n_facilities, n_points, service_radius, breakeven_orders, daily_fixed_cost, contribution_per_order):
    st.subheader(name)
    st.write(cfg["context"])

    x, y = generate_demand_points(n_points, cfg["radius_km"], cfg["shape"], cfg.get("concentration", 0.7), seed=cfg["seed"])
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

    result = {
        "name": name,
        "total_orders": total_orders_day,
        "captured": captured_orders,
        "coverage_pct": coverage_pct,
        "avg_orders_per_store": store_df["Orders/day (within radius)"].mean(),
        "n_viable": (store_df["Viable?"] == "✅ Yes").sum(),
        "n_stores": len(store_df),
        "network_profit": network_profit,
    }
    return result, x, y, weights


def render_cost_tradeoff(name, x, y, weights, daily_fixed_cost, transport_cost_per_order_km, current_n):
    """
    Classic facility-location trade-off: as the number of facilities rises,
    total fixed cost rises linearly while average delivery distance — and
    therefore total variable/transport cost — falls. This sweep ignores the
    quick-commerce service-radius promise on purpose: it shows the general
    network-design trade-off, separate from the viability check above.
    """
    ks = list(range(1, 11))
    rows = []
    for k in ks:
        _, nearest_dist, _ = solve_facility_location(x, y, weights, k)
        total_transport = np.sum(nearest_dist * weights) * transport_cost_per_order_km
        total_fixed = k * daily_fixed_cost
        rows.append({
            "Facilities": k,
            "Fixed cost (₹/day)": total_fixed,
            "Transport cost (₹/day)": total_transport,
            "Total cost (₹/day)": total_fixed + total_transport,
        })
    df = pd.DataFrame(rows)
    optimal_k = int(df.loc[df["Total cost (₹/day)"].idxmin(), "Facilities"])

    fig = px.line(
        df, x="Facilities",
        y=["Fixed cost (₹/day)", "Transport cost (₹/day)", "Total cost (₹/day)"],
        markers=True, title=f"{name}: fixed vs. delivery cost trade-off (network-wide, unconstrained by service radius)"
    )
    fig.add_vline(x=current_n, line_dash="dot", line_color="gray", annotation_text="Current setting")
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="green", annotation_text=f"Cost-minimizing = {optimal_k}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Cost-minimizing network size here: **{optimal_k} facilities**. "
        f"Fewer stores concentrate fixed cost but stretch delivery distance; "
        f"more stores shrink delivery distance but multiply fixed cost."
    )
    return df, optimal_k


summary_rows = []
for name, cfg in scenarios_to_run.items():
    result, x, y, weights = render_scenario(
        name, cfg, n_facilities, n_points_slider, service_radius,
        breakeven_orders, daily_fixed_cost, contribution_per_order
    )
    summary_rows.append(result)
    render_cost_tradeoff(name, x, y, weights, daily_fixed_cost, transport_cost_per_order_km, n_facilities)
    st.divider()

# ----------------------------------------------------------------------------
# 6. CROSS-CITY SUMMARY (only meaningful with more than one scenario)
# ----------------------------------------------------------------------------
if len(summary_rows) > 1:
    st.header("📊 Cross-City Summary")
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
