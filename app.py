# app.py
import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime

# plotting
import plotly.graph_objects as go
import plotly.express as px

# your environment (simulation)
from env import RuralEnv

# ---------------------------
# Page config & CSS (simple)
# ---------------------------
st.set_page_config(page_title="Emergency Response Simulator", page_icon="🚑", layout="wide")

st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        font-weight: 700;
        color: #0B62A4;
        text-align: center;
        margin-bottom: 4px;
    }
    .subtitle {
        font-size: 14px;
        color: #666;
        text-align: center;
        margin-bottom: 16px;
    }
    .big-card {
        background: linear-gradient(90deg,#f8fbff,#eef6ff);
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .large-btn {
        font-size: 16px;
        padding: 8px 12px;
    }
    .eta-badge {
        font-weight:700;
        padding:6px 10px;
        border-radius:8px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="title">🚑 Emergency Response Simulator (Rural)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find nearby ambulances & hospitals by specialty — simple, fast, and visual.</div>', unsafe_allow_html=True)

# ---------------------------
# Constants & specialties
# ---------------------------
SPECIALTIES = [
    "General", "Cardiology", "Neurology", "Gynecology", "ICU", "Orthopedic", "Pediatrics"
]

# how many minutes per grid hop (manhattan distance). Tuneable.
ETA_PER_HOP_MIN = 5

# ---------------------------
# Session state init
# ---------------------------
if "simulation_history" not in st.session_state:
    st.session_state.simulation_history = []

if "total_runs" not in st.session_state:
    st.session_state.total_runs = 0

if "success_count" not in st.session_state:
    st.session_state.success_count = 0

if "show_analytics" not in st.session_state:
    st.session_state.show_analytics = False

# ---------------------------
# Sidebar - Controls
# ---------------------------
with st.sidebar:
    st.header("Configuration")
    st.markdown("**Simulation Settings**")
    n_episodes = st.slider("Number of emergencies", 1, 10, 2, help="How many simulated emergencies to run")
    n_hospitals = st.selectbox("Number of hospitals (max 2)", [1, 2], index=1)
    n_ambulances = st.selectbox("Number of ambulances (1-3)", [1, 2, 3], index=1)
    grid_size = st.selectbox("Grid size", ["4x4", "5x5", "6x6"], index=1)
    grid_rows, grid_cols = map(int, grid_size.split("x"))
    st.markdown("---")
    st.markdown("**Emergency Input**")
    severity = st.radio("Severity", options=["Low", "Medium", "High"], index=2, horizontal=True)
    required_specialty = st.selectbox("Required specialty", SPECIALTIES, index=0)
    st.markdown("---")
    st.markdown("**Display Options**")
    top_k = st.slider("Top ambulances to show", 1, min(3, n_ambulances), min(2, n_ambulances))
    show_grid_map = st.checkbox("Show grid map", value=True)
    show_analytics = st.checkbox("Show analytics tab", value=False)
    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.simulation_history = []
        st.session_state.total_runs = 0
        st.session_state.success_count = 0
        st.experimental_rerun()

# ---------------------------
# Load Q-table if available
# ---------------------------
Q = None
try:
    Q = np.load("results/q_table.npy")
    st.sidebar.success("Loaded trained Q-table")
except Exception:
    st.sidebar.info("No Q-table found — simulation will use a simple policy")

# ---------------------------
# Initialize environment
# ---------------------------
env = RuralEnv(grid=(grid_rows, grid_cols), n_hospitals=n_hospitals, n_ambulances=n_ambulances)

# ---------------------------
# Utility functions
# ---------------------------
def manhattan(a_zone, b_zone, cols):
    """Compute manhattan distance in grid given zone index and number of columns"""
    ar, ac = a_zone // cols, a_zone % cols
    br, bc = b_zone // cols, b_zone % cols
    return abs(ar - br) + abs(ac - bc)

def eta_minutes(distance_hops):
    return int(distance_hops * ETA_PER_HOP_MIN)

def get_top_ambulances(patient_zone, ambulances, top_k=3, cols=4):
    """Return ambulances sorted by ETA (ascending) with computed eta and distance."""
    items = []
    for a in ambulances:
        dist = manhattan(patient_zone, a["zone"], cols)
        eta = eta_minutes(dist)
        items.append({**a, "dist_hops": dist, "eta_min": eta})
    items_sorted = sorted(items, key=lambda x: x["eta_min"])
    return items_sorted[:top_k], items_sorted

def draw_grid(patient_zone, ambulance_list, hospital_list, selected_ambulance=None, selected_hospital=None, rows=4, cols=4):
    """Return a textual grid view with emoji markers for quick display."""
    grid = [["⬜" for _ in range(cols)] for _ in range(rows)]
    # patient
    pr, pc = patient_zone // cols, patient_zone % cols
    grid[pr][pc] = "🧍"
    # hospitals
    for idx, h in enumerate(hospital_list):
        hr, hc = h["zone"] // cols, h["zone"] % cols
        icon = "🏥"
        if required_specialty != "General" and required_specialty in [s.title() for s, v in h["specialties"].items() if v]:
            icon = "🏥✨"  # highlight hospitals matching specialty
        if selected_hospital is not None and idx == selected_hospital:
            icon = icon + "🔶"
        # append if already something there
        if grid[hr][hc] == "⬜":
            grid[hr][hc] = icon
        else:
            grid[hr][hc] = grid[hr][hc] + icon
    # ambulances
    for a in ambulance_list:
        ar, ac = a["zone"] // cols, a["zone"] % cols
        icon = "🚑"
        if selected_ambulance is not None and a["id"] == selected_ambulance:
            icon = icon + "⭐"
        if grid[ar][ac] == "⬜":
            grid[ar][ac] = icon
        else:
            grid[ar][ac] = grid[ar][ac] + icon
    return "\n".join([" ".join(r) for r in grid])

# ---------------------------
# Main content layout
# ---------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("## 🔎 Find Ambulance & Hospital")
    st.markdown("<div class='big-card'>Choose severity and specialty, then click **Find Help**. The simulator will pick the best ambulance(s).</div>", unsafe_allow_html=True)
    run = st.button("🚨 Find Help", help="Start simulation for the current emergency", use_container_width=True)

    if run:
        # Run single "emergency" episode
        for episode in range(n_episodes):
            # Reset environment and get readable info
            state = env.reset()
            info = env.render_state_readable()
            patient = info["patient"]
            ambulances = info["ambulances"]
            hospitals = info["hospitals"]

            # Decide action: use Q-table if available (simple argmax), else heuristic:
            if Q is not None:
                # clamp state index for safety
                try:
                    action_row = Q[state]
                    action = int(np.argmax(action_row))
                except Exception:
                    action = np.random.randint(0, env.n_ambulances * env.n_hospitals)
            else:
                # Simple heuristic: pick ambulance with min distance to patient, and hospital that supports specialty and has beds
                # We choose the ambul index and hospital index of shortest combined ETA
                candidates = []
                for a_idx, a in enumerate(ambulances):
                    for h_idx, h in enumerate(hospitals):
                        # distance: ambulance -> patient -> hospital (sum of hops)
                        d1 = manhattan(a["zone"], patient["zone"], grid_cols)
                        d2 = manhattan(patient["zone"], h["zone"], grid_cols)
                        eta_total = eta_minutes(d1 + d2)
                        # check specialty and bed availability (prefer matches)
                        specialty_ok = True
                        req = required_specialty.lower()
                        if req != "general":
                            # env stores specialties possibly in a dict; check keys case-insensitively
                            specialty_ok = any(req == s.lower() and v for s, v in h["specialties"].items())
                        bed_ok = h.get("available_beds", 0) > 0
                        priority = (0 if specialty_ok else 1) + (0 if bed_ok else 2)
                        candidates.append({
                            "a_idx": a_idx, "h_idx": h_idx, "eta": eta_total, "priority": priority
                        })
                # sort by priority then eta
                candidates_sorted = sorted(candidates, key=lambda x: (x["priority"], x["eta"]))
                if candidates_sorted:
                    top = candidates_sorted[0]
                    action = top["a_idx"] * env.n_hospitals + top["h_idx"]
                else:
                    action = np.random.randint(0, env.n_ambulances * env.n_hospitals)

            # Interpret action
            amb_idx = action // env.n_hospitals
            hosp_idx = action % env.n_hospitals

            # Compute selected ambulance ETAs
            patient_zone = patient["zone"]
            amb_list_sorted, all_amb_sorted = get_top_ambulances(patient_zone, ambulances, top_k=top_k, cols=grid_cols)

            # Display results
            st.markdown(f"### 🧾 Emergency #{st.session_state.total_runs + 1} — {severity} severity")
            st.markdown(f"**Required specialty:** {required_specialty}")
            st.markdown(f"**Patient location:** Zone {patient_zone} — {datetime.now().strftime('%H:%M:%S')}")
            st.divider()

            # Show top ambulances (big readable cards)
            st.markdown("#### 🚑 Nearby Ambulances")
            amb_cols = st.columns(len(amb_list_sorted))
            for idx, amb in enumerate(amb_list_sorted):
                with amb_cols[idx]:
                    # ETA color simple mapping
                    eta = amb["eta_min"]
                    if eta <= 10:
                        color = "#2ecc71"  # green
                        label = "Fast"
                    elif eta <= 25:
                        color = "#f39c12"  # orange
                        label = "Okay"
                    else:
                        color = "#e74c3c"  # red
                        label = "Slow"

                    st.markdown(f"<div style='padding:10px;border-radius:8px;background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,0.06)'>"
                                f"<h4 style='margin:0'>{amb['id']} {'' if idx>0 else ' (Selected)'}</h4>"
                                f"<p style='margin:4px 0'><strong>ETA:</strong> <span class='eta-badge' style='background:{color};color:#fff'>{amb['eta_min']} min</span></p>"
                                f"<p style='margin:4px 0'>Location: Zone {amb['zone']}</p>"
                                f"</div>", unsafe_allow_html=True)

            st.divider()

            # Show hospitals list
            st.markdown("#### 🏥 Nearby Hospitals")
            # annotate hospitals with distance and specialty match
            hosp_rows = []
            for i, h in enumerate(hospitals):
                d = manhattan(patient_zone, h["zone"], grid_cols)
                eta_h = eta_minutes(d)
                specialties_available = ", ".join([k.title() for k, v in h["specialties"].items() if v]) or "General"
                specialty_match = required_specialty.lower() == "general" or any(required_specialty.lower() == k.lower() and v for k, v in h["specialties"].items())
                hosp_rows.append({
                    "id": h["id"],
                    "zone": h["zone"],
                    "eta_min": eta_h,
                    "beds": f"{h.get('available_beds',0)}/{h.get('total_beds',0)}",
                    "icu": h.get("icu_available", False),
                    "specialties": specialties_available,
                    "match": specialty_match
                })
            hosp_sorted = sorted(hosp_rows, key=lambda x: x["eta_min"])
            for h in hosp_sorted:
                match_text = "✅" if h["match"] else ""
                highlight = "background:#e8f8f5;border-left:4px solid #0b9a78;padding:8px;border-radius:6px;margin-bottom:8px;" if h["match"] else "background:#fff;border:1px solid #eee;padding:8px;border-radius:6px;margin-bottom:8px;"
                st.markdown(f"<div style='{highlight}'>"
                            f"<strong>Hospital {h['id']} {match_text}</strong> — Zone {h['zone']}<br/>"
                            f"ETA: <strong>{h['eta_min']} min</strong> &nbsp; • &nbsp; Beds: {h['beds']} &nbsp; • &nbsp; ICU: {'Yes' if h['icu'] else 'No'}<br/>"
                            f"Specialties: {h['specialties']}"
                            f"</div>", unsafe_allow_html=True)

            # Optionally show grid visual for easy understanding
            if show_grid_map:
                st.markdown("#### 🗺️ Grid View (Patient / Ambulances / Hospitals)")
                grid_text = draw_grid(patient_zone, ambulances, hospitals, selected_ambulance=ambulances[amb_idx]["id"], selected_hospital=hosp_idx, rows=grid_rows, cols=grid_cols)
                st.code(grid_text)

            # Simulate dispatch step (call env.step)
            ns, reward, done, step_info = env.step(action)

            # Display outcome card
            if step_info.get("success", False):
                st.success(f"✅ Dispatch successful — Travel time: {step_info.get('travel_time_min', 'N/A')} min • Distance: {step_info.get('dist_hops', 'N/A')} hops")
                st.session_state.success_count += 1
            else:
                reason = step_info.get("reason", "Unknown")
                st.error(f"❌ Dispatch failed — Reason: {reason}")

            # Save to session history
            st.session_state.simulation_history.append({
                "episode": st.session_state.total_runs + 1,
                "reward": float(reward),
                "success": bool(step_info.get("success", False)),
                "travel_time": float(step_info.get("travel_time_min", 0)),
                "distance": int(step_info.get("dist_hops", 0)),
                "severity": severity,
                "specialty": required_specialty
            })
            st.session_state.total_runs += 1

            # short pause between episodes to simulate time (tuneable)
            time.sleep(0.6)

        st.success(f"Completed {n_episodes} simulation(s).")

with right_col:
    st.markdown("### 📊 Quick Stats")
    st.markdown("<div class='big-card'>", unsafe_allow_html=True)
    st.metric("Total Runs", st.session_state.total_runs)
    sr = (st.session_state.success_count / st.session_state.total_runs * 100) if st.session_state.total_runs > 0 else 0.0
    st.metric("Success Rate", f"{sr:.1f}%")
    if st.session_state.simulation_history:
        df_recent = pd.DataFrame(st.session_state.simulation_history[-10:])
        st.markdown("#### Recent Results")
        st.dataframe(df_recent[["episode", "success", "travel_time", "distance", "severity", "specialty"]].sort_values(by="episode", ascending=False).reset_index(drop=True))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Analytics tab (optional)
# ---------------------------
if show_analytics and st.session_state.simulation_history:
    st.markdown("---")
    st.markdown("## Analytics")
    df = pd.DataFrame(st.session_state.simulation_history)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Episodes", len(df))
    col2.metric("Avg Travel Time (min)", f"{df['travel_time'].mean():.1f}")
    col3.metric("Avg Distance (hops)", f"{df['distance'].mean():.1f}")

    # Success by specialty
    spec_stats = df.groupby("specialty")["success"].mean().reset_index().sort_values("success", ascending=False)
    fig = px.bar(spec_stats, x="specialty", y="success", labels={"success": "Success Rate"}, title="Success Rate by Specialty")
    st.plotly_chart(fig, use_container_width=True)

    # Travel time distribution
    fig2 = px.histogram(df, x="travel_time", nbins=20, title="Travel Time Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#888'>Emergency Response Simulator • Academic Demo • Built with Streamlit</div>", unsafe_allow_html=True)
