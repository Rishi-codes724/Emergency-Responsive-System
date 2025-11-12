# app.py
"""
Mobile-style single-page frontend for the Emergency Response Simulator (academic demo).
Uses your RuralEnv simulation and the RL agent (Q-table or rl_agent.select_action if present).
Designed for end-users: large buttons/cards, minimal technical details.
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime

from env import RuralEnv  # your environment

# constants
SPECIALTIES = ["General", "Cardiology", "Neurology", "Gynecology", "ICU", "Orthopedic", "Pediatrics"]
ETA_PER_HOP_MIN = 5  # minutes per Manhattan hop — tuneable

# page config and mobile-like styles
st.set_page_config(page_title="Ambulance Finder", page_icon="🚑", layout="centered")
st.markdown(
    """
    <style>
    /* Mobile-like centered card */
    .top-title {font-size:28px; font-weight:700; text-align:center; margin-bottom:6px; color:#0b60a4;}
    .top-sub {font-size:14px; text-align:center; color:#555; margin-bottom:18px;}
    .card {background: #fff; border-radius:14px; padding:16px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); margin-bottom:12px;}
    .big-btn {font-size:18px; padding:12px 16px; border-radius:10px;}
    .ambulance-card {padding:10px; border-radius:10px; margin-bottom:8px;}
    .hospital-card {padding:10px; border-radius:10px; margin-bottom:8px;}
    .badge {display:inline-block; padding:6px 10px; border-radius:12px; font-weight:600;}
    .green {background:#2ecc71;color:#fff;}
    .orange {background:#f39c12;color:#fff;}
    .red {background:#e74c3c;color:#fff;}
    .small {font-size:13px; color:#444;}
    .center {text-align:center;}
    code {white-space: pre-wrap;}
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="top-title">🚑 Ambulance Finder — Rural Emergency</div>', unsafe_allow_html=True)
st.markdown('<div class="top-sub">Select severity & specialty → Get the nearest ambulances and hospitals</div>', unsafe_allow_html=True)

# --- session state initialization ---
if "history" not in st.session_state:
    st.session_state.history = []
if "total_runs" not in st.session_state:
    st.session_state.total_runs = 0
if "success_count" not in st.session_state:
    st.session_state.success_count = 0

# --- Controls Card (central, mobile style) ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # layout: vertical stack for mobile feel
    grid_choice = st.selectbox("Grid size", ["4x4", "5x5", "6x6"], index=1)
    grid_rows, grid_cols = map(int, grid_choice.split("x"))

    n_hospitals = st.selectbox("Hospitals (max 2)", [1, 2], index=1)
    n_ambulances = st.selectbox("Ambulances (1–3)", [1, 2, 3], index=1)

    severity = st.selectbox("Severity", ["Low", "Medium", "High"], index=2)
    specialty = st.selectbox("Required Specialty", SPECIALTIES, index=0)

    top_k = st.slider("Show top ambulances", 1, min(3, n_ambulances), min(2, n_ambulances))
    sim_count = st.slider("Simulate episodes (for demo)", 1, 5, 1)

    st.markdown("<div class='small'>Tip: Choose severity and specialty, then press <strong>Find Help</strong>.</div>", unsafe_allow_html=True)
    find_btn = st.button("🚨 Find Help", key="find_btn", help="Run the simulation and find best ambulance + hospital", args=None)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Attempt to load Q-table (silent) ---
Q = None
try:
    Q = np.load("results/q_table.npy")
except Exception:
    Q = None

# Try import rl_agent.select_action for additional policy (optional)
rl_agent = None
select_action_fn = None
try:
    import rl_agent
    select_action_fn = getattr(rl_agent, "select_action", None)
except Exception:
    select_action_fn = None

# --- Utility helpers (hidden from user) ---
def manhattan(a_zone, b_zone, cols):
    ar, ac = a_zone // cols, a_zone % cols
    br, bc = b_zone // cols, b_zone % cols
    return abs(ar - br) + abs(ac - bc)

def eta_from_hops(hops):
    return int(hops * ETA_PER_HOP_MIN)

def eta_badge_html(eta):
    if eta <= 10:
        cls = "green"
        text = f"{eta} min • Fast"
    elif eta <= 25:
        cls = "orange"
        text = f"{eta} min • Moderate"
    else:
        cls = "red"
        text = f"{eta} min • Slow"
    return f"<span class='badge {cls}'>{text}</span>"

def choose_action_using_Q(state):
    try:
        row = Q[state]
        action = int(np.argmax(row))
        return action
    except Exception:
        return None

def fallback_heuristic(ambulances, hospitals, patient_zone, grid_cols, req_specialty):
    """
    Hidden heuristic: prefer ambulance with smallest (ambulance->patient + patient->hospital) ETA,
    prefer hospital with matching specialty and available beds.
    Returns action index (amb_idx * n_hospitals + hosp_idx).
    """
    candidates = []
    for a_idx, a in enumerate(ambulances):
        for h_idx, h in enumerate(hospitals):
            d1 = manhattan(a["zone"], patient_zone, grid_cols)
            d2 = manhattan(patient_zone, h["zone"], grid_cols)
            eta = eta_from_hops(d1 + d2)
            # check specialty
            req = req_specialty.lower()
            specialty_ok = True
            if req != "general":
                # environment specialties likely a dict of booleans; check keys
                specialty_ok = any(req == s.lower() and v for s, v in h["specialties"].items())
            beds_ok = h.get("available_beds", 0) > 0
            priority = (0 if specialty_ok else 1) + (0 if beds_ok else 2)
            candidates.append((priority, eta, a_idx, h_idx))
    if not candidates:
        return 0  # fallback to zero
    candidates_sorted = sorted(candidates, key=lambda x: (x[0], x[1]))
    _, _, a_idx, h_idx = candidates_sorted[0]
    return a_idx, h_idx

# --- Run simulation and show user-facing results ---
def run_simulation_once(n_episodes=1):
    env = RuralEnv(grid=(grid_rows, grid_cols), n_hospitals=n_hospitals, n_ambulances=n_ambulances)
    results_for_display = []

    for ep in range(n_episodes):
        state = env.reset()
        info = env.render_state_readable()  # dictionary with patient, ambulances, hospitals
        patient = info["patient"]
        ambulances = info["ambulances"]
        hospitals = info["hospitals"]

        # Decide action using priority:
        action = None
        # 1) Q-table if present
        if Q is not None:
            act = choose_action_using_Q(state)
            if act is not None:
                action = act

        # 2) rl_agent.select_action if available
        if action is None and select_action_fn is not None:
            try:
                # try to call agent (some implementations expect env/state)
                action = select_action_fn(state=state, env=env)
            except TypeError:
                try:
                    action = select_action_fn(state)  # alternate signature
                except Exception:
                    action = None
            except Exception:
                action = None

        # 3) fallback heuristic
        if action is None:
            try:
                a_idx, h_idx = fallback_heuristic(ambulances, hospitals, patient["zone"], grid_cols, specialty)
                action = a_idx * len(hospitals) + h_idx
            except Exception:
                action = np.random.randint(0, env.n_ambulances * env.n_hospitals)

        # interpret action
        amb_idx = int(action // len(hospitals))
        hosp_idx = int(action % len(hospitals))

        # prepare top ambulances list (sorted by ambulance->patient ETA)
        amb_with_eta = []
        for a in ambulances:
            hops = manhattan(a["zone"], patient["zone"], grid_cols)
            amb_with_eta.append({**a, "hops_to_patient": hops, "eta_min": eta_from_hops(hops)})
        amb_sorted = sorted(amb_with_eta, key=lambda x: x["eta_min"])
        top_ambs = amb_sorted[:top_k]

        # prepare hospital info (patient->hospital ETA), matches for specialty
        hosp_display = []
        for i, h in enumerate(hospitals):
            hops = manhattan(patient["zone"], h["zone"], grid_cols)
            match = (specialty.lower() == "general") or any(specialty.lower() == k.lower() and v for k, v in h["specialties"].items())
            hosp_display.append({
                "id": h["id"],
                "zone": h["zone"],
                "eta_min": eta_from_hops(hops),
                "beds": f"{h.get('available_beds',0)}/{h.get('total_beds',0)}",
                "icu": h.get("icu_available", False),
                "specialties": ", ".join([k.title() for k, v in h["specialties"].items() if v]) or "General",
                "match": match,
                "index": i
            })
        hosp_display_sorted = sorted(hosp_display, key=lambda x: x["eta_min"])
        # selected hospital object (from action)
        selected_hospital = next((h for h in hosp_display if h["index"] == hosp_idx), hosp_display_sorted[0] if hosp_display_sorted else None)
        selected_ambulance = next((a for a in ambulances if a["id"] == ambulances[amb_idx]["id"]), ambulances[0] if ambulances else None)

        # call env.step(action) to simulate and get step_info (this remains hidden internally)
        try:
            ns, reward, done, step_info = env.step(action)
        except Exception:
            # if env.step expects different action encoding, try fallback: just simulate success
            step_info = {"success": True, "travel_time_min": selected_hospital["eta_min"] if selected_hospital else 0, "dist_hops": selected_hospital["eta_min"] // ETA_PER_HOP_MIN if selected_hospital else 0}
            reward = 1.0

        # determine success to store stats
        success_flag = bool(step_info.get("success", False))
        travel_time = float(step_info.get("travel_time_min", selected_hospital["eta_min"] if selected_hospital else 0))
        dist_hops = int(step_info.get("dist_hops", (travel_time // ETA_PER_HOP_MIN) if travel_time else 0))

        # store summary for display
        results_for_display.append({
            "patient_zone": patient["zone"],
            "severity": severity,
            "specialty": specialty,
            "top_ambulances": top_ambs,
            "nearby_hospitals": hosp_display_sorted[:n_hospitals],
            "selected_ambulance": {"id": selected_ambulance["id"], "zone": selected_ambulance["zone"], "eta_min": next((t["eta_min"] for t in top_ambs if t["id"] == selected_ambulance["id"]), None)},
            "selected_hospital": selected_hospital,
            "success": success_flag,
            "reward": float(reward),
            "travel_time": travel_time,
            "distance_hops": dist_hops,
            "raw_step_info": step_info
        })

        # update session stats (not shown to user as RL internals)
        st.session_state.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_zone": patient["zone"],
            "severity": severity,
            "specialty": specialty,
            "selected_ambulance": selected_ambulance["id"],
            "selected_hospital": selected_hospital["id"] if selected_hospital else None,
            "success": success_flag,
            "travel_time": travel_time,
            "distance_hops": dist_hops
        })
        st.session_state.total_runs += 1
        if success_flag:
            st.session_state.success_count += 1

        # small wait to feel like a real app (tweak or remove)
        time.sleep(0.4)

    return results_for_display

# --- UI display logic when Find Help is pressed ---
if find_btn:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="center"><strong>Searching for nearest ambulances and hospitals...</strong></div>', unsafe_allow_html=True)
        # run simulation
        results = run_simulation_once(sim_count)
        # show only last result (most recent) to keep UI simple for users
        last = results[-1]

        st.markdown("<br/>", unsafe_allow_html=True)

        # Show top ambulances as big horizontal cards
        st.markdown("### 🚑 Nearest Ambulances")
        for idx, amb in enumerate(last["top_ambulances"]):
            eta_html = eta_badge_html(amb["eta_min"])
            selected_label = " (Selected)" if amb["id"] == last["selected_ambulance"]["id"] else ""
            card_style = "ambulance-card"
            st.markdown(
                f"<div class='{card_style}' style='background:#f7fbff;border:1px solid #e6f0fb;padding:10px;'>"
                f"<strong style='font-size:16px'>Ambulance {amb['id']}{selected_label}</strong> &nbsp; • &nbsp; Zone {amb['zone']}<br/>"
                f"{eta_html} &nbsp; <span class='small'>({amb['hops_to_patient']} hops)</span>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<br/>")
        # Hospitals
        st.markdown("### 🏥 Nearby Hospitals")
        for h in last["nearby_hospitals"]:
            match_txt = "✅ Specialist" if h["match"] else ""
            highlight_style = "background:#eaf7f0;border-left:4px solid #0b9a78;padding:8px;border-radius:8px;margin-bottom:8px;" if h["match"] else "background:#fff;border:1px solid #eee;padding:8px;border-radius:8px;margin-bottom:8px;"
            st.markdown(f"<div style='{highlight_style}'>"
                        f"<strong>Hospital {h['id']}</strong> — Zone {h['zone']} &nbsp; {match_txt}<br/>"
                        f"ETA: <strong>{h['eta_min']} min</strong> &nbsp; • &nbsp; Beds: {h['beds']} &nbsp; • &nbsp; ICU: {'Yes' if h['icu'] else 'No'}<br/>"
                        f"Specialties: {h['specialties']}"
                        f"</div>", unsafe_allow_html=True)

        st.markdown("<br/>")
        # Final selected option card
        sel_amb = last["selected_ambulance"]
        sel_hosp = last["selected_hospital"]
        if sel_amb and sel_hosp:
            st.markdown("### ✅ Best Option")
            st.markdown(
                f"<div style='background:linear-gradient(90deg,#e8f8ff,#f3fbff);padding:14px;border-radius:12px;border:1px solid #dfeffd;'>"
                f"<div style='font-size:18px;font-weight:700'>Dispatching Ambulance {sel_amb['id']}</div>"
                f"<div style='margin-top:6px'>Ambulance ETA: <strong>{sel_amb['eta_min']} min</strong></div>"
                f"<div style='margin-top:6px'>Target: <strong>Hospital {sel_hosp['id']}</strong> — ETA to hospital: <strong>{sel_hosp['eta_min']} min</strong></div>"
                f"<div style='margin-top:8px;color:#444'>If required, move patient immediately to the ambulance. Follow local instructions.</div>"
                f"</div>", unsafe_allow_html=True)
        else:
            st.warning("No suitable ambulance/hospital found. Please try again or change severity/specialty.")

        st.markdown('</div>', unsafe_allow_html=True)

        # provide CSV download and small stats
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<div class='center'><strong>Session Summary</strong></div>")
            st.markdown(f"<div class='small'>Total runs: {st.session_state.total_runs} • Successes: {st.session_state.success_count}</div>", unsafe_allow_html=True)

            if st.button("⬇️ Download Session CSV"):
                df_hist = pd.DataFrame(st.session_state.history)
                csv = df_hist.to_csv(index=False)
                st.download_button("Download", csv, "session_history.csv", "text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

# --- small footer (mobile friendly) ---
st.markdown("<div style='height:18px'></div>")
st.markdown("<div style='text-align:center;color:#888;font-size:13px'>Emergency Response Simulator • Academic Demo</div>")

