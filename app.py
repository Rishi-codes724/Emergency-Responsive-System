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

# Custom CSS styling for a professional UI
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f9fafc 0%, #f0f4ff 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    h2 {
        color: #e63946 !important;
        text-align: center;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .stSelectbox label, .stButton button, .stMarkdown {
        font-size: 16px !important;
    }
    .ambulance-card, .hospital-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: 0.2s ease;
    }
    .ambulance-card:hover, .hospital-card:hover {
        transform: scale(1.01);
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
    }
    .best-option {
        background-color: #e6ffee;
        border-left: 6px solid #00b050;
        padding: 15px;
        border-radius: 10px;
        font-size: 17px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Initialize Environment and Agent
# --------------------------------------------------
env = RuralEnv(n_hospitals=2, n_ambulances=random.randint(1, 3))
agent = QLearningAgent(env.n_states, env.action_space)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("<h2>🚑 Emergency Ambulance Dispatch System</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>AI-powered dispatch optimization for rural healthcare emergencies</p>", unsafe_allow_html=True)
st.markdown("---")

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
# Step 2 – Run Simulation
# --------------------------------------------------
if find_help:
    with st.spinner("Analyzing and dispatching nearest resources..."):
        time.sleep(1.2)

        state = env.reset()
        env.patient["severity"] = ["Low", "Medium", "High"].index(severity)
        env.patient["required_specialty"] = specialty if specialty != "General" else None

        done = False
        steps = 0
        total_reward = 0
        chosen_info = None

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            chosen_info = info
            steps += 1
            if steps > 5:
                break

        ambulances = [{"name": f"Ambulance {i+1}", "eta": random.randint(3, 20)} for i in range(env.n_ambulances)]
        hospitals = [{
            "name": f"Hospital {i+1}",
            "specialty": random.choice(["General", "Cardiology", "Neurology", "ICU", "Orthopedic", "Pediatrics"]),
            "distance": random.randint(10, 60),
        } for i in range(env.n_hospitals)]

        ambulances.sort(key=lambda x: x["eta"])
        hospitals.sort(key=lambda x: x["distance"])

        # --------------------------------------------------
        # Step 4 – Display Results
        # --------------------------------------------------
        st.markdown("---")
        st.markdown("### 🚑 Nearby Ambulances")
        for amb in ambulances:
            st.markdown(
                f"<div class='ambulance-card'><b>{amb['name']}</b> — "
                f"<span style='color:#e63946;'>{amb['eta']} mins away</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 🏥 Nearby Hospitals")
        for hosp in hospitals:
            highlight = "#e6f7ff" if hosp["specialty"] == specialty else "#ffffff"
            st.markdown(
                f"<div class='hospital-card' style='background-color:{highlight};'>"
                f"<b>{hosp['name']}</b><br>"
                f"🩺 Specialty: {hosp['specialty']}<br>"
                f"⏱️ Distance: {hosp['distance']} mins</div>",
                unsafe_allow_html=True,
            )

        best_ambulance = ambulances[0]
        best_hospital = hospitals[0]

        st.markdown("---")
        st.markdown("### ✅ Best Option Found")
        st.markdown(
            f"<div class='best-option'>🚑 <b>{best_ambulance['name']}</b> will reach in "
            f"<b>{best_ambulance['eta']} mins</b><br>"
            f"🏥 Hospital: <b>{best_hospital['name']}</b> "
            f"({best_hospital['specialty']}, {best_hospital['distance']} mins away)</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='text-align:center; color:gray; font-size:14px;'>Simulation complete — for research demo only.</p>",
            unsafe_allow_html=True,
        )

else:
    st.info("Select emergency details above and press **Find Help** to start the simulation.")
