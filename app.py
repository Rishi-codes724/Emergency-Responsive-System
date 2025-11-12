# app.py
import streamlit as st
import random
import time
from rl_agent import QLearningAgent as RLAgent
from env import RuralEnv

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Emergency Ambulance Finder", layout="centered")

# --------------------------------------------------
# Initialize Environment and Agent
# --------------------------------------------------
env = RuralEnv()
agent = RLAgent(env)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h2 style='text-align:center; color:#ff4d4d;'>🚑 Emergency Ambulance Finder</h2>
    <p style='text-align:center; font-size:18px;'>Emergency Response System for Rural Areas</p>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Step 1 – User Input
# --------------------------------------------------
st.markdown("### 🆘 Step 1: Describe the Emergency")

col1, col2 = st.columns(2)
with col1:
    severity = st.selectbox(
        "Emergency Severity",
        ["Low", "Medium", "High"],
        index=1,
    )
with col2:
    specialty = st.selectbox(
        "Required Specialty",
        [
            "General",
            "Cardiology",
            "Neurology",
            "Gynecology",
            "ICU",
            "Orthopedic",
            "Pediatrics",
        ],
    )

find_help = st.button("🚨 Find Help", use_container_width=True)

# --------------------------------------------------
# Step 2 – Run Simulation
# --------------------------------------------------
if find_help:
    with st.spinner("Finding the nearest ambulances and hospitals..."):
        time.sleep(1.5)

        # Reset environment and simulate one RL episode
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
            if steps > 10:
                break

        # --------------------------------------------------
        # Generate Simulated Ambulance and Hospital Data
        # --------------------------------------------------
        ambulances = [
            {"name": "Ambulance 1", "eta": random.randint(3, 7)},
            {"name": "Ambulance 2", "eta": random.randint(8, 15)},
            {"name": "Ambulance 3", "eta": random.randint(15, 25)},
        ]
        hospitals = [
            {
                "name": "City Hospital",
                "specialty": random.choice(
                    ["General", "Cardiology", "Neurology", "ICU"]
                ),
                "distance": random.randint(10, 40),
            },
            {
                "name": "Rural Health Center",
                "specialty": random.choice(
                    ["Gynecology", "Orthopedic", "Pediatrics"]
                ),
                "distance": random.randint(15, 60),
            },
        ]

        ambulances = sorted(ambulances, key=lambda x: x["eta"])
        hospitals = sorted(hospitals, key=lambda x: x["distance"])

        # --------------------------------------------------
        # Step 3 – Display Results
        # --------------------------------------------------
        st.markdown("---")
        st.markdown("### 🚑 Step 2: Nearby Ambulances")

        for amb in ambulances:
            st.markdown(
                f"""
                <div style='background-color:#fff5f5; padding:10px; border-radius:10px; margin-bottom:8px;'>
                    <b>{amb['name']}</b> — 
                    <span style='color:#ff4d4d;'>{amb['eta']} mins away</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 🏥 Step 3: Nearby Hospitals")
        for hosp in hospitals:
            highlight = "#d4f8d4" if hosp["specialty"] == specialty else "#f9f9f9"
            st.markdown(
                f"""
                <div style='background-color:{highlight}; padding:10px; border-radius:10px; margin-bottom:8px;'>
                    <b>{hosp['name']}</b><br>
                    🩺 Specialty: {hosp['specialty']}<br>
                    ⏱️ Distance: {hosp['distance']} mins
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --------------------------------------------------
        # Step 4 – Best Option Summary
        # --------------------------------------------------
        best_ambulance = ambulances[0]
        best_hospital = hospitals[0]

        st.markdown("---")
        st.markdown("### ✅ Step 4: Best Option Found")
        st.success(
            f"""
            🚑 **{best_ambulance['name']}** will reach you in **{best_ambulance['eta']} mins**  
            🏥 Nearest hospital: **{best_hospital['name']}**  
            (Specialty: {best_hospital['specialty']}, {best_hospital['distance']} mins away)
            """
        )

        st.markdown(
            "<p style='text-align:center; color:gray;'>Simulation complete — academic demo only.</p>",
            unsafe_allow_html=True,
        )

else:
    st.info(
        "Select emergency details above and press **Find Help** to start the simulation."
    )

