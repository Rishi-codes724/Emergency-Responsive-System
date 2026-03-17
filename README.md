# 🚑 Emergency Ambulance Dispatch System using Reinforcement Learning

## 📌 Project Overview

Emergency response time is critical in saving lives. This project proposes an AI-based ambulance dispatch system that uses Reinforcement Learning (RL) to assign the most suitable ambulance and hospital to a patient in emergency situations.
The system simulates a rural healthcare network and learns optimal dispatch strategies through interaction with the environment.

## 🎯 Objective

- Assign the nearest available ambulance to emergency patients.
- Select a hospital based on availability and patient needs.
- Minimize response time and dispatch delays.
- Improve decision making using Reinforcement Learning.

## 🧠 Reinforcement Learning Approach

We use a Q-Learning agent to learn the best dispatch strategy.
- Agent:
The dispatch system that decides which ambulance and hospital should handle the emergency.
- Environment:
  - A 4×4 grid simulation representing rural zones where:
  - Ambulances are placed randomly
  - Patients appear with different severities
  - Hospitals have limited resources
- State
The agent observes:
  - Patient location
  - Patient severity
  - Ambulance locations
Hospital availability (beds, ICU, specialties)
- Actions
The agent selects:
  - Which ambulance should respond
  - Which hospital the patient should be taken to
- Reward System
| Scenario | Reward |
|---------|--------|
| Successful treatment | +50 |
| Travel delay | -1 |
| No hospital bed | -10 |
| No required specialty | -20 |
| No ICU available | -40 |

The agent learns to maximize reward by improving dispatch decisions.
[View Live Project](https://emergency-responsive-system-5t3ep6uefqvj8ebcgqerdz.streamlit.app/#emergency-ambulance-finder)
