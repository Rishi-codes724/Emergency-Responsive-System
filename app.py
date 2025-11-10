# app.py
import streamlit as st
import numpy as np
import time
from env import RuralEnv
from rl_agent import QLearningAgent

st.set_page_config(page_title="Emergency Response System (RL)", layout="wide")

st.title("🚑 Emergency Response System using Reinforcement Learning")
st.markdown("""
This demo simulates ambulance–hospital dispatch optimization in a rural region using **Q-learning**.
""")

# Sidebar controls
st.sidebar.header("Simulation Controls")
episodes = st.sidebar.slider("Number of Training Episodes", 100, 5000, 1000, step=100)
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 1.0, 0.98)
epsilon = st.sidebar.slider("Exploration Rate (ε)", 0.01, 1.0, 0.2)

# Environment setup
env = RuralEnv()
agent = QLearningAgent(env.n_states, env.action_space, alpha=alpha, gamma=gamma, epsilon=epsilon)

if st.button("🧠 Train Agent"):
    st.write("Training agent... please wait ⏳")
    rewards = []
    progress = st.progress(0)
    for ep in range(episodes):
        s = env.reset()
        done = False
        total_r = 0
        while not done:
            a = agent.select_action(s)
            s_next, r, done, info = env.step(a)
            agent.learn(s, a, r, s_next)
            total_r += r
            s = s_next
        rewards.append(total_r)
        progress.progress((ep+1)/episodes)
    st.success("Training completed ✅")
    st.line_chart(rewards, x_label="Episode", y_label="Reward")

st.divider()
st.header("🚨 Test the Trained Agent")

if st.button("Run Simulation"):
    state = env.reset()
    s_readable = env.render_state_readable()
    st.subheader("Initial Scenario")
    st.json(s_readable["patient"])
    st.json({"Ambulances": s_readable["ambulances"], "Hospitals": s_readable["hospitals"]})

    a = agent.select_action(state)
    s_next, r, done, info = env.step(a)
    st.subheader("Result")
    st.write(f"**Action chosen:** Ambulance #{info['ambulance']} → Hospital #{info['hospital']}")
    st.write(f"**Reward:** {r:.2f}")
    st.write(f"**Success:** {info['success']}")
    st.write(f"**Reason:** {info['reason'] or 'N/A'}")
    st.write(f"**Travel time (min):** {info['travel_time_min']:.1f}")

    st.success("Simulation completed ✅")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Reinforcement Learning")
