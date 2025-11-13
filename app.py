# app.py
import streamlit as st
import random
import time
from rl_agent import QLearningAgent
from env import RuralEnv

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Emergency Ambulance Dispatch System", layout="centered")

# Add professional background and fonts
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #1a1a1a;
    }
    .stApp {
        background: linear-gradient(145deg, #fdfdfd, #f3f6f9);
        color: #333;
    }
    div[data-testid="stMarkdownContainer"] {
        font-size: 17px;
        line-height: 1.5;
    }
    .result-box {
        background-color: #ffffff;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        border-radius: 12px;
        padding: 12px 15px;
        margin-bottom: 10px;
    }
    .highlight {
        background-color: #d4f8d4;
    }
    .btn-primary {
        background-color: #ff4d4d;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        border: none;
        font-weight: 600;
    }
    .btn-primary:hover {
        background-color: #ff3333;
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
st.markdown(
    """
    <h2 style='text-align:center; color:#ff4d4d; font-weight:700;'>🚑 Emergency Ambulance Dispatch System</h2>
    <p style='text-align:center; color:#4a4a4a;'>AI-Powered Emergency Response for Rural Areas</p>
    """,
    unsafe_allow_html=True,
)

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
                <div class='result-box'>
                    <b>{amb['name']}</b> — 
                    <span style='color:#ff4d4d;'>{amb['eta']} mins away</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 🏥 Nearby Hospitals")
        for hosp in hospitals:
            highlight = "highlight" if hosp["specialty"] == specialty else ""
            st.markdown(
                f"""
                <div class='result-box {highlight}'>
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
            🚑 <b>{best_ambulance['name']}</b> will reach you in <b>{best_ambulance['eta']} mins</b><br>
            🏥 Nearest hospital: <b>{best_hospital['name']}</b><br>
            (Specialty: {best_hospital['specialty']}, {best_hospital['distance']} mins away)
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='text-align:center; color:gray; font-size:14px;'>Simulation complete — academic demo only.</p>",
            unsafe_allow_html=True,
        )

else:
    st.info("Select emergency details above and press **Find Help** to start the simulation.")
