# app.py
import streamlit as st
import random
import time
from rl_agent import QLearningAgent
from env import RuralEnv

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Emergency Ambulance Finder", layout="centered")

# --------------------------------------------------
# Initialize Environment and Agent
# --------------------------------------------------
env = RuralEnv(n_hospitals=2, n_ambulances=random.randint(1, 3))
agent = QLearningAgent(env.n_states, env.action_space)

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
    severity = st.selectbox("Emergency Severity", ["Low", "Medium", "High"], index=1)
with col2:
    specialty = st.selectbox(
        "Required Specialty",
        ["General", "Cardiology", "Neurology", "Gynecology", "ICU", "Orthopedic", "Pediatrics"],
    )

find_help = st.button("🚨 Find Help", use_container_width=True)

# --------------------------------------------------
# Step 2 – Run Simulation
# --------------------------------------------------
if find_help:
    with st.spinner("Finding the nearest ambulances and hospitals..."):
        time.sleep(1.2)

        # Reset environment and set patient manually
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

        # --------------------------------------------------
        # Step 3 – Simulated Display Data
        # --------------------------------------------------
        ambulances = []
        for i in range(env.n_ambulances):
            ambulances.append(
                {"name": f"Ambulance {i+1}", "eta": random.randint(3, 20)}
            )

        hospitals = []
        for i in range(env.n_hospitals):
            hosp = env.hospitals[i]
            hospitals.append(
                {
                    "name": f"Hospital {i+1}",
                    "specialty": random.choice(
                        ["General", "Cardiology", "Neurology", "ICU", "Orthopedic", "Pediatrics"]
                    ),
                    "distance": random.randint(10, 60),
                }
            )

        ambulances = sorted(ambulances, key=lambda x: x["eta"])
        hospitals = sorted(hospitals, key=lambda x: x["distance"])

        # --------------------------------------------------
        # Step 4 – Display Results
        # --------------------------------------------------
        st.markdown("---")
        st.markdown("### 🚑 Nearby Ambulances")

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

        st.markdown("### 🏥 Nearby Hospitals")
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
        # Step 5 – Best Option Summary
        # --------------------------------------------------
        best_ambulance = ambulances[0]
        best_hospital = hospitals[0]

        st.markdown("---")
        st.markdown("### ✅ Best Option Found")
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
    st.info("Select emergency details above and press **Find Help** to start the simulation.")
