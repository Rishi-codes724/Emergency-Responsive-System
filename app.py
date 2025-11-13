import streamlit as st
import random
import time
from rl_agent import QLearningAgent
from env import RuralEnv

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(
    page_title="Emergency Ambulance Dispatch System",
    layout="centered",
    page_icon="🚑",
)

# --------------------------------------------------
# Global Styling
# --------------------------------------------------
st.markdown("""
    <style>
    /* Page background */
    body {
        background: linear-gradient(135deg, #f0f7f4 0%, #dff6ff 100%);
        font-family: 'Poppins', sans-serif;
        color: #1a1a1a;
    }
    /* Header title */
    h2 {
        color: #0077b6 !important;
        text-align: center;
        font-weight: 800;
        letter-spacing: 0.6px;
        margin-bottom: 5px;
    }
    h3 {
        color: #0077b6 !important;
        font-weight: 700;
    }
    /* Buttons */
    .stButton > button {
        background-color: #0077b6;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #0096c7;
        transform: scale(1.03);
    }
    /* Cards */
    .info-card {
        background: linear-gradient(145deg, #ffffff, #f0f9ff);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        transition: all 0.25s ease-in-out;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    /* Best option card */
    .best-option {
        background: linear-gradient(135deg, #caffbf, #d8f3dc);
        border-left: 6px solid #2d6a4f;
        padding: 15px 20px;
        border-radius: 12px;
        color: #1a1a1a;
        font-weight: 500;
    }
    /* Section dividers */
    hr {
        border: 1px solid #caf0f8;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    /* Info text */
    .subtext {
        text-align: center;
        color: #555;
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Initialize Environment and Agent
# --------------------------------------------------
env = RuralEnv(n_hospitals=2, n_ambulances=random.randint(1, 3))
agent = QLearningAgent(env.n_states, env.action_space)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("<h2>🚑 Emergency Ambulance Dispatch System</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>AI-Powered Rural Emergency Response System — Optimized using Q-Learning</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# Step 1 – User Input
# --------------------------------------------------
st.markdown("### 🆘 Describe the Emergency")

col1, col2 = st.columns(2)
with col1:
    severity = st.selectbox("Emergency Severity", ["Low", "Medium", "High"], index=1)
with col2:
    specialty = st.selectbox(
        "Required Specialty",
        ["General", "Cardiology🫀", "Neurology🧠", "Gynecology🤰🏻", "Orthopedic🦴🦵🏻", "Pediatrics👶🏻"],
    )

find_help = st.button("🚨 Find Help", use_container_width=True)

# --------------------------------------------------
# Step 2 – Simulation
# --------------------------------------------------
if find_help:
    with st.spinner("Analyzing available ambulances and hospitals..."):
        time.sleep(1.2)

        # Reset and simulate
        state = env.reset()
        env.patient["severity"] = ["Low", "Medium", "High"].index(severity)
        env.patient["required_specialty"] = specialty if specialty != "General" else None

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
            if steps > 5:
                break

        # Randomized example data for display
        ambulances = [{"name": f"Ambulance {i+1}", "eta": random.randint(3, 20)} for i in range(env.n_ambulances)]
        hospitals = [{
            "name": f"Hospital {i+1}",
            "specialty": random.choice(["General", "Cardiology", "Neurology", "ICU", "Orthopedic", "Pediatrics"]),
            "distance": random.randint(10, 60),
        } for i in range(env.n_hospitals)]

        ambulances.sort(key=lambda x: x["eta"])
        hospitals.sort(key=lambda x: x["distance"])

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🚑 Nearby Ambulances")

        for amb in ambulances:
            st.markdown(
                f"<div class='info-card'><b>{amb['name']}</b><br>"
                f"⏱️ ETA: <span style='color:#0077b6; font-weight:600;'>{amb['eta']} mins</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 🏥 Nearby Hospitals")

        for hosp in hospitals:
            highlight = "#e0f7ff" if hosp["specialty"] == specialty else "#ffffff"
            st.markdown(
                f"<div class='info-card' style='background:{highlight};'>"
                f"<b>{hosp['name']}</b><br>"
                f"🩺 Specialty: {hosp['specialty']}<br>"
                f"🚗 Distance: {hosp['distance']} mins</div>",
                unsafe_allow_html=True,
            )

        best_ambulance = ambulances[0]
        best_hospital = hospitals[0]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### ✅ Best Option Found")

        st.markdown(
            f"<div class='best-option'>"
            f"🚑 <b>{best_ambulance['name']}</b> will reach in <b>{best_ambulance['eta']} mins</b><br>"
            f"🏥 Destination: <b>{best_hospital['name']}</b><br>"
            f"Specialty: {best_hospital['specialty']} | Distance: {best_hospital['distance']} mins"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<p class='subtext'>Simulation complete — for academic demonstration only.</p>", unsafe_allow_html=True)

else:
    st.info("Select emergency details above and press **Find Help** to start the simulation.")
