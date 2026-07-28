# -*- coding: utf-8 -*-
"""
Facility Location & Network Design Simulator — Quick-Commerce Dark-Store Edition
Author: Manish Sarkhel, IIM Sirmaur (PGPEX-LSM)

Three pages:
  1. Single scenario   — one city's dark-store viability (map + breakeven check)
  2. Compare all three — Chandigarh / Dehradun / Paonta Sahib side by side
  3. Core Concepts     — response time, inventory, transport, and facility cost
                         curves vs. number of facilities, grounded in real 2026
                         Indian quick-commerce dark-store data (Blinkit, Zepto,
                         Swiggy Instamart), plus a network-design archetype
                         reference table.

To run:
    pip install -r requirements.txt
    streamlit run facility_location_sim.py

See instructor_notes.md for teaching context and discussion material.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# 5. CORE-CONCEPTS COST MODEL (illustrative, not fitted to any company's books)
# ----------------------------------------------------------------------------
#
# These four functions translate the classic network-design cost/response
# trade-offs into simple, well-known scaling laws:
#   - Response time distance ~ 1/sqrt(n)   (facility catchment area ~ A/n)
#   - Inventory (safety stock) ~ sqrt(n)   (Maister's square-root law)
#   - Outbound/last-mile transport ~ 1/sqrt(n) (shorter average delivery hop)
#   - Inbound transport ~ n^0.3            (smaller, less-than-truckload lots)
#   - Facility cost ~ n                    (roughly linear, mildly convex)
#
# Coefficients are illustrative teaching assumptions, chosen to produce a
# visible cost-minimizing facility count well below where real platforms
# operate — NOT derived from any company's actual financial statements.

def facility_cost_cr(n, per_facility_cr=0.04):
    return n * per_facility_cr


def inventory_cost_cr(n, base_cr=0.5):
    return base_cr * np.sqrt(n)


def transport_inbound_cr(n, base_cr=2.0, exponent=0.3):
    return base_cr * n ** exponent


def transport_outbound_cr(n, base_cr=50.0):
    return base_cr / np.sqrt(n)


def response_time_minutes(n, base_hours=10.0):
    return (base_hours / np.sqrt(n)) * 60


# Real, publicly reported dark-store counts (India quick commerce, FY23–FY26).
# Source: industry reporting on quarterly dark-store counts; figures change
# quickly as platforms keep expanding — treat as directional, not exact.
BLINKIT_GROWTH = {
    "Period": ["Q2 FY23", "Q4 FY24", "Q1 FY25", "Q2 FY25", "Dec 2024", "Q4 FY25", "Q2 FY26", "Q4 FY26"],
    "Dark stores": [383, 526, 639, 791, 1000, 1301, 1816, 2243],
}
MARCH_2026_SNAPSHOT = {
    "Platform": ["Blinkit", "Zepto", "Swiggy Instamart"],
    "Dark stores (Mar 2026)": [2243, 1089, 1038],
}

NETWORK_ARCHETYPES = pd.DataFrame({
    "Archetype": [
        "Manufacturer storage, direct shipping",
        "Manufacturer storage, direct shipping + in-transit merge",
        "Distributor storage, carrier delivery",
        "Distributor storage, last-mile delivery",
        "Manufacturer/distributor storage, customer pickup",
        "Retail storage, customer pickup",
    ],
    "How it works": [
        "Product ships straight from the factory to the customer; no intermediate stocking.",
        "Parts from multiple plants are combined by the carrier en route into one shipment.",
        "A distributor holds stock and ships to retail outlets via a common carrier.",
        "The distributor's own (or partnered) fleet delivers straight to the end customer.",
        "Stock sits upstream; the customer collects it themselves.",
        "Traditional retail: goods sit on a store shelf, customer visits and buys.",
    ],
    "Contemporary Indian example": [
        "D2C brands shipping made-to-order or personalized items factory-direct.",
        "Large-appliance + accessory bundles merged in transit by logistics partners.",
        "FMCG C&F agents supplying kirana stores; publisher distributors to bookstores.",
        "Blinkit, Zepto, Swiggy Instamart dark stores — the model this simulator studies.",
        "Car dealership pickup (Maruti, Tata Motors); large-appliance showroom pickup.",
        "DMart, kirana stores, traditional grocery and department stores.",
    ],
})


def render_core_concepts(n_current, show_company_lines):
    st.header("Core Concepts: Response Time, Cost, and the Facility-Count Trade-off")
    st.write(
        "Every additional facility buys a shorter response time — but it changes "
        "inventory cost, transportation cost, and facility cost in different, "
        "sometimes opposite, directions. This page uses India's 2026 quick-commerce "
        "dark-store race as a real, contemporary anchor for that trade-off."
    )

    growth_df = pd.DataFrame(BLINKIT_GROWTH)
    fig_growth = px.line(growth_df, x="Period", y="Dark stores", markers=True,
                          title="Blinkit's real facility count growth, FY23-FY26")
    st.plotly_chart(fig_growth, use_container_width=True)
    st.caption("Source: industry reporting on Blinkit's quarterly dark-store counts, FY23-FY26. "
               "Illustrates 'shorter response time = more facilities needed' as a real expansion path.")

    snap_df = pd.DataFrame(MARCH_2026_SNAPSHOT)
    fig_snap = px.bar(snap_df, x="Platform", y="Dark stores (Mar 2026)",
                       title="India quick-commerce dark-store counts, March 2026")
    st.plotly_chart(fig_snap, use_container_width=True)
    st.caption("Source: industry reporting, March 2026. Store counts change quickly as platforms keep expanding.")

    st.info(
        "In January 2026, Blinkit shifted its own marketing away from a fixed "
        "'10-minute' promise toward a broader doorstep-delivery message, reportedly "
        "after discussions with the labour ministry about time-bound delivery "
        "pressure on gig workers. The 'response time' variable below is also a real "
        "operational and human one, not just a modelling convenience."
    )

    ns = np.unique(np.round(np.logspace(0, np.log10(2500), 60)).astype(int))
    curve_df = pd.DataFrame({
        "Facilities": ns,
        "Facility cost (Rs cr/mo)": facility_cost_cr(ns),
        "Inventory cost (Rs cr/mo)": inventory_cost_cr(ns),
        "Inbound transport (Rs cr/mo)": transport_inbound_cr(ns),
        "Outbound/last-mile transport (Rs cr/mo)": transport_outbound_cr(ns),
        "Response time (min)": response_time_minutes(ns),
    })
    curve_df["Total logistics cost (Rs cr/mo)"] = (
        curve_df["Facility cost (Rs cr/mo)"] + curve_df["Inventory cost (Rs cr/mo)"]
        + curve_df["Inbound transport (Rs cr/mo)"] + curve_df["Outbound/last-mile transport (Rs cr/mo)"]
    )

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Response time vs. facilities",
        "Inventory cost vs. facilities (square-root law)",
        "Transport cost vs. facilities (inbound vs. outbound)",
        "Facility cost vs. facilities",
    ))
    fig.add_trace(go.Scatter(x=curve_df["Facilities"], y=curve_df["Response time (min)"],
                              mode="lines", name="Response time"), row=1, col=1)
    fig.add_trace(go.Scatter(x=curve_df["Facilities"], y=curve_df["Inventory cost (Rs cr/mo)"],
                              mode="lines", name="Inventory cost"), row=1, col=2)
    fig.add_trace(go.Scatter(x=curve_df["Facilities"], y=curve_df["Inbound transport (Rs cr/mo)"],
                              mode="lines", name="Inbound transport"), row=2, col=1)
    fig.add_trace(go.Scatter(x=curve_df["Facilities"], y=curve_df["Outbound/last-mile transport (Rs cr/mo)"],
                              mode="lines", name="Outbound / last-mile transport"), row=2, col=1)
    fig.add_trace(go.Scatter(x=curve_df["Facilities"], y=curve_df["Facility cost (Rs cr/mo)"],
                              mode="lines", name="Facility cost"), row=2, col=2)
    fig.update_xaxes(type="log")
    fig.update_layout(height=650, showlegend=True,
                       title_text="Facility-count cost & responsiveness curves (illustrative assumptions)")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=curve_df["Facilities"], y=curve_df["Total logistics cost (Rs cr/mo)"],
                               mode="lines", name="Total logistics cost (Rs cr/mo)"))
    fig2.add_trace(go.Scatter(x=curve_df["Facilities"], y=curve_df["Response time (min)"],
                               mode="lines", name="Response time (min)", yaxis="y2", line=dict(dash="dot")))
    optimal_n = int(curve_df.loc[curve_df["Total logistics cost (Rs cr/mo)"].idxmin(), "Facilities"])
    fig2.update_layout(
        xaxis=dict(title="Number of facilities", type="log"),
        yaxis=dict(title="Total logistics cost (Rs cr/mo)"),
        yaxis2=dict(title="Response time (min)", overlaying="y", side="right"),
        title="Total logistics cost vs. response time - the strategic trade-off",
        height=450,
    )
    if show_company_lines:
        for count, label, color in [(2243, "Blinkit", "red"), (1089, "Zepto", "green"), (1038, "Instamart", "purple")]:
            fig2.add_vline(x=count, line_dash="dot", line_color=color, annotation_text=label)
    fig2.add_vline(x=optimal_n, line_dash="dash", line_color="black", annotation_text=f"Cost-minimizing ~ {optimal_n}")
    fig2.add_vline(x=n_current, line_color="gray", annotation_text="Your setting")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        f"On these illustrative assumptions, total logistics cost is minimized around "
        f"**{optimal_n} facilities** - yet real quick-commerce platforms operate at "
        f"1,000-2,200+ stores. That gap is the point: they are buying response time "
        f"(and the order volume/revenue it captures) at a logistics-cost premium, a "
        f"deliberate strategic choice, not an oversight."
    )

    with st.expander("Reference: six network design archetypes (with contemporary Indian examples)"):
        st.dataframe(NETWORK_ARCHETYPES, use_container_width=True)
        st.caption("Quick commerce (the rest of this simulator) sits in the "
                    "'distributor storage, last-mile delivery' row.")


# ----------------------------------------------------------------------------
# 6. STREAMLIT APP
# ----------------------------------------------------------------------------

st.title("Facility Location & Network Design: Quick-Commerce Edition")

st.sidebar.header("1. Page")
mode = st.sidebar.radio(
    "Choose a page",
    ["Single scenario", "Compare all three cities", "Core Concepts: Facility Count Trade-offs"],
)

if mode in ("Single scenario", "Compare all three cities"):

    if mode == "Single scenario":
        scenario_name = st.sidebar.selectbox("City scenario", list(CITY_PRESETS.keys()))
        scenarios_to_run = {scenario_name: CITY_PRESETS[scenario_name]}
    else:
        scenarios_to_run = CITY_PRESETS

    n_facilities = st.sidebar.slider("Number of dark stores", 1, 10, 3)
    n_points_slider = st.sidebar.slider(
        "Demand sample points (simulation resolution)", 50, 400, 200, 25,
        help="More points = smoother density pattern. Total order volume is fixed "
             "by population x penetration x orders/user; this only controls resolution."
    )

    st.sidebar.header("2. Quick-commerce delivery promise")
    service_radius = st.sidebar.slider(
        "Delivery service radius (km)", 1.0, 6.0, 2.0, 0.25,
        help="Demand outside this radius from every facility is not captured by "
             "the quick-commerce channel."
    )

    st.sidebar.header("3. Cost structure (Rs)")
    fixed_cost_month = st.sidebar.number_input("Fixed cost per store / month (Rs)", value=450_000, step=25_000,
        help="Rent, staff, equipment - cost of simply keeping a store open, independent of order volume.")
    contribution_per_order = st.sidebar.number_input("Contribution margin per order (Rs)", value=30, step=1,
        help="Revenue per order minus variable delivery/packing cost.")
    transport_cost_per_order_km = st.sidebar.number_input("Delivery cost per order per km (Rs)", value=5, step=1,
        help="Variable cost of physically delivering one order one km. This is what "
             "rises as stores get farther apart and falls as you add more, closer stores.")

    daily_fixed_cost = fixed_cost_month / 30
    breakeven_orders = daily_fixed_cost / contribution_per_order
    st.sidebar.metric("-> Breakeven orders / store / day", f"{breakeven_orders:,.0f}")

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
                "Viable?": "Yes" if store_orders >= breakeven_orders else "No",
                "Daily profit (Rs)": round(profit),
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
        network_profit = store_df["Daily profit (Rs)"].sum()
        c4.metric("Network daily profit (Rs)", f"{network_profit:,.0f}")

        st.dataframe(store_df, use_container_width=True)

        result = {
            "name": name,
            "total_orders": total_orders_day,
            "captured": captured_orders,
            "coverage_pct": coverage_pct,
            "avg_orders_per_store": store_df["Orders/day (within radius)"].mean(),
            "n_viable": (store_df["Viable?"] == "Yes").sum(),
            "n_stores": len(store_df),
            "network_profit": network_profit,
        }
        return result, x, y, weights

    def render_cost_tradeoff(name, x, y, weights, daily_fixed_cost, transport_cost_per_order_km, current_n):
        """
        Classic facility-location trade-off: as the number of facilities rises,
        total fixed cost rises linearly while average delivery distance - and
        therefore total variable/transport cost - falls. This sweep ignores the
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
                "Fixed cost (Rs/day)": total_fixed,
                "Transport cost (Rs/day)": total_transport,
                "Total cost (Rs/day)": total_fixed + total_transport,
            })
        df = pd.DataFrame(rows)
        optimal_k = int(df.loc[df["Total cost (Rs/day)"].idxmin(), "Facilities"])

        fig = px.line(
            df, x="Facilities",
            y=["Fixed cost (Rs/day)", "Transport cost (Rs/day)", "Total cost (Rs/day)"],
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

    if len(summary_rows) > 1:
        st.header("Cross-City Summary")
        summary_df = pd.DataFrame(summary_rows)[
            ["name", "avg_orders_per_store", "n_viable", "n_stores", "coverage_pct", "network_profit"]
        ]
        summary_df.columns = ["City", "Avg orders/store/day", "Viable stores", "Total stores",
                               "Demand coverage %", "Network daily profit (Rs)"]
        st.dataframe(summary_df, use_container_width=True)

        fig = px.bar(summary_df, x="City", y="Avg orders/store/day",
                     title="Average orders captured per store per day vs breakeven line")
        fig.add_hline(y=breakeven_orders, line_dash="dash", line_color="red",
                      annotation_text="Breakeven threshold")
        st.plotly_chart(fig, use_container_width=True)

else:  # Core Concepts page
    st.sidebar.header("2. Core concepts controls")
    n_core = st.sidebar.select_slider(
        "Number of facilities",
        options=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 2243, 2500],
        value=100,
        help="Move this to see where your setting falls relative to the cost-minimizing "
             "point and to real quick-commerce platforms' actual store counts."
    )
    show_company_lines = st.sidebar.checkbox("Show Blinkit / Zepto / Instamart reference lines", value=True)
    render_core_concepts(n_core, show_company_lines)
