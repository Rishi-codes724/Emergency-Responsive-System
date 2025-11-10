import streamlit as st
import numpy as np
from env import RuralEnv

st.title("Smart Ambulance Dispatch Simulator")
st.write("Simulation of RL-based ambulance dispatch considering hospital beds and specialties.")

# Slider to select number of demo episodes
n_episodes = st.slider("Number of Demo Episodes:", 1, 10, 3)

# Load trained Q-table if exists
try:
    Q = np.load("results/q_table.npy")
    st.success("Trained Q-table loaded!")
except:
    Q = None
    st.warning("No Q-table found. Using default actions.")

# Initialize environment
env = RuralEnv(grid=(4,4), n_hospitals=6, n_ambulances=3)

# Function to visualize 4x4 grid
def draw_grid(patient_zone, ambulance_zone=None, hospital_zones=[]):
    rows, cols = 4,4
    grid = [["⬜" for _ in range(cols)] for _ in range(rows)]
    pr, pc = patient_zone // cols, patient_zone % cols
    grid[pr][pc] = "🧍"  # Patient

    if ambulance_zone is not None:
        ar, ac = ambulance_zone // cols, ambulance_zone % cols
        grid[ar][ac] = "🚑"  # Ambulance

    for hz in hospital_zones:
        hr, hc = hz // cols, hz % cols
        if grid[hr][hc] == "⬜":
            grid[hr][hc] = "🏥"
        else:
            grid[hr][hc] += "🏥"
    return "\n".join([" ".join(row) for row in grid])

# Run demo episodes
for i in range(n_episodes):
    state = env.reset()
    info = env.render_state_readable()
    st.subheader(f"Episode {i+1}")

    # Display patient
    st.write("**Patient:**", info["patient"])
    # Display ambulances
    st.write("**Ambulances (id, zone):**", [(a['id'], a['zone']) for a in info['ambulances']])
    # Display hospitals
    st.write("**Hospitals summary:**")
    for h in info['hospitals']:
        st.write(f"H{h['id']} | Zone: {h['zone']} | Beds: {h['available_beds']}/{h['total_beds']} | ICU: {h['icu_available']} | Specialties: {h['specialties']}")

    # Agent selects action using Q-table
    if Q is not None:
        row = Q[state]
        action = int(np.argmax(row))
    else:
        action = 0

    amb_idx = action // env.n_hospitals
    hosp_idx = action % env.n_hospitals
    st.write("**Agent selected action:** Ambulance", amb_idx, "→ Hospital", hosp_idx)

    # Step environment
    ns, reward, done, step_info = env.step(action)
    st.write("**Result:**", step_info, "| **Reward:**", reward)

    # Draw 4x4 grid
    patient_zone = info["patient"]["zone"]
    ambulance_zone = info["ambulances"][amb_idx]["zone"]
    hospital_zones = [h["zone"] for h in info["hospitals"]]
    st.text(draw_grid(patient_zone, ambulance_zone, hospital_zones))
    st.write("---")

