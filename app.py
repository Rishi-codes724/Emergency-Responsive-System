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
# Custom Styling with Enhanced Readability
# --------------------------------------------------
st.markdown("""
    <style>
    /* Page background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 30% 20%, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2a33, #11252f);
        color: white;
    }
/* Title Styling */
h2 {
    color: #00fff0; /* bright cyan */
    text-align: center;
    font-weight: 1000;
    font-size: 3em; /* increased size */
    letter-spacing: 1px;
    margin-bottom: 8px;
    text-transform: uppercase;
    text-shadow: 
        0 0 20px rgba(0, 255, 240, 0.8),
        0 0 40px rgba(0, 255, 255, 0.6),
        0 0 60px rgba(0, 255, 255, 0.4);
    animation: pulseGlow 2.5s infinite alternate;
    position: relative;
}

/* Ambulance Emoji Before Title */
h2::before {
    content: "🚑 ";
    font-size: 1.5em;
    vertical-align: middle;
    margin-right: 6px;
    text-shadow: 0 0 20px rgba(0,255,255,0.8);
}

/* Soft pulsing glow effect */
@keyframes pulseGlow {
    0% {
        text-shadow: 
            0 0 15px rgba(0, 255, 240, 0.6),
            0 0 30px rgba(0, 255, 255, 0.4);
        transform: scale(1);
    }
    100% {
        text-shadow: 
            0 0 30px rgba(0, 255, 255, 1),
            0 0 60px rgba(0, 255, 255, 0.8);
        transform: scale(1.02);
    }
}

/* Subtext Styling */
.subtext {
    text-align: center;
    color: #b8f3ff;
    font-size: 18px;
    font-weight: 600;
    margin-top: -4px;
    text-shadow: 
        0 0 10px rgba(0, 255, 255, 0.4),
        0 0 20px rgba(0, 255, 255, 0.2);
}


    /* Select boxes */
    label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00e0ff, #0077b6);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        height: 3em;
        font-size: 16px;
        border: none;
        box-shadow: 0 0 8px #00e0ff60;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        transform: scale(1.05);
        box-shadow: 0 0 16px #00e0ff90;
    }

    /* Info Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 0 15px rgba(0, 224, 255, 0.2);
        transition: all 0.25s ease-in-out;
        color: #f1faff;
    }
    .info-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 0 20px rgba(0, 224, 255, 0.4);
    }

    /* Best Option Card */
    .best-option {
        background: linear-gradient(120deg, #1b4332, #2d6a4f);
        border-left: 6px solid #00ff9d;
        padding: 15px 20px;
        border-radius: 12px;
        color: #d9fdd3;
        font-weight: 500;
        box-shadow: 0 0 20px #00ff9d40;
    }

    hr {
        border: 1px solid rgba(255,255,255,0.1);
        margin-top: 20px;
        margin-bottom: 20px;
    }

    @keyframes pulse {
        0% {box-shadow: 0 0 8px #00e0ff80;}
        50% {box-shadow: 0 0 16px #00e0ff;}
        100% {box-shadow: 0 0 8px #00e0ff80;}
    }
    .pulse {
        animation: pulse 2s infinite;
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
st.markdown("<h2> Emergency Ambulance Dispatch System</h2>", unsafe_allow_html=True)
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
        ["General", "Cardiology", "Neurology", "Gynecology", "Orthopedic", "Pediatrics"],
    )

find_help = st.button("🚨 Find Help", use_container_width=True)

# --------------------------------------------------
# Step 2 – Simulation
# --------------------------------------------------
if find_help:
    with st.spinner("Analyzing available ambulances and hospitals..."):
        time.sleep(1.2)

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
                f"<div class='info-card pulse'><b>{amb['name']}</b><br>"
                f"⏱️ ETA: <span style='color:#00e0ff; font-weight:600;'>{amb['eta']} mins</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 🏥 Nearby Hospitals")

        for hosp in hospitals:
            highlight = "rgba(0,255,255,0.1)" if hosp["specialty"] == specialty else "rgba(255,255,255,0.05)"
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

else:
    st.info("Select emergency details above and press **Find Help** to start the simulation.")




