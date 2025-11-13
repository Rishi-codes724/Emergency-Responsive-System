import streamlit as st
import random
import time
from rl_agent import QLearningAgent
from env import RuralEnv

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Emergency Ambulance Dispatch System", layout="centered")

# Background and Style
page_bg = """
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #e3f2fd 0%, #f1f8e9 100%);
    color: #1b1b1b;
    font-family: 'Poppins', sans-serif;
}

/* Title */
h2 {
    color: #004d40 !important;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.5px;
}

/* Section headers */
h3 {
    color: #1565c0 !important;
    font-weight: 600;
    margin-top: 20px;
}

/* Cards styling */
div[data-testid="stMarkdownContainer"] > div {
    transition: transform 0.2s ease, box-shadow 0.3s ease;
}
div[data-testid="stMarkdownContainer"] > div:hover {
    transform: translateY(-3px);
    box-shadow: 0px 3px 8px rgba(0,0,0,0.15);
}

/* Buttons */
div.stButton > button {
    background-color: #1976d2;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1em;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #0d47a1;
    transform: scale(1.03);
}

/* Info and success boxes */
.stAlert {
    border-radius: 10px !important;
}

/* Section Dividers */
hr {
    border: none;
    height: 1px;
    background-color: #90caf9;
    margin: 20px 0;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --------------------------------------------------
# Initialize Environment and Agent
# --------------------------------------------------
env = RuralEnv(n_hospitals=2, n_ambulances=random.randint(1, 3))
agent = QLearningAgent(env.n_states, env.action_space)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("<h2>Emergency Ambulance Dispatch System</h2>", unsafe_allow_html=True)

# --------------------------------------------------
# Step 1 – User Input
# --------------------------------------------------
st.markdown("### Describe the Emergency")

col1, col2 = st.columns(2)
with col1:
    severity = st.selectbox("Emergency Severity", ["Low", "Medium", "High"], index=1)
with col2:
    specialty = st.selectbox(
        "Required Specialty",
        ["General", "Cardiology", "Neurology", "Gynecology", "Orthopedic", "Pediatrics"],
    )

find_help = st.button("Find Help")

# --------------------------------------------------
# Step 2 – Run Simulation
# --------------------------------------------------
if find_help:
    with st.spinner("Analyzing available ambulances and hospitals..."):
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
        ambulances = [{"name": f"Ambulance {i+1}", "eta": random.randint(3, 20)} for i in range(env.n_ambulances)]
        hospitals = [
            {
                "name": f"Hospital {i+1}",
                "specialty": random.choice(
                    ["General", "Cardiology", "Neurology", "ICU", "Orthopedic", "Pediatrics"]
                ),
                "distance": random.randint(10, 60),
            }
            for i in range(env.n_hospitals)
        ]

        ambulances.sort(key=lambda x: x["eta"])
        hospitals.sort(key=lambda x: x["distance"])

        # --------------------------------------------------
        # Step 4 – Display Results
        # --------------------------------------------------
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Nearby Ambulances")

        for amb in ambulances:
            st.markdown(
                f"""
                <div style='background-color:#e3f2fd; padding:10px; border-radius:10px; margin-bottom:8px;'>
                    <b>{amb['name']}</b> — 
                    <span style='color:#0d47a1;'>{amb['eta']} mins away</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Nearby Hospitals")
        for hosp in hospitals:
            highlight = "#c8e6c9" if hosp["specialty"] == specialty else "#f1f8e9"
            st.markdown(
                f"""
                <div style='background-color:{highlight}; padding:10px; border-radius:10px; margin-bottom:8px;'>
                    <b>{hosp['name']}</b><br>
                    Specialty: {hosp['specialty']}<br>
                    Distance: {hosp['distance']} mins
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --------------------------------------------------
        # Step 5 – Best Option Summary
        # --------------------------------------------------
        best_ambulance = ambulances[0]
        best_hospital = hospitals[0]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Best Option Found")
        st.success(
            f"""
            <div style='font-size:16px;'>
            🚐 <b>{best_ambulance['name']}</b> will reach in <b>{best_ambulance['eta']} mins</b><br>
            🏥 Nearest hospital: <b>{best_hospital['name']}</b><br>
            Specialty: {best_hospital['specialty']}, Distance: {best_hospital['distance']} mins
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='text-align:center; color:#424242;'>Simulation complete — academic demo only.</p>",
            unsafe_allow_html=True,
        )

else:
    st.info("Select emergency details above and press **Find Help** to start the simulation.")
