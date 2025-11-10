# demo.py
import numpy as np
from env import RuralEnv

def demo(n=10):
    env = RuralEnv(grid=(4,4), n_hospitals=6, n_ambulances=3)
    try:
        Q = np.load("results/q_table.npy")
    except:
        Q = None
    for i in range(n):
        s = env.reset()
        readable = env.render_state_readable()
        print("\n--- Episode", i+1)
        print("Patient:", readable["patient"])
        print("Hospitals summary (id, zone, beds, icu, specs):")
        for h in readable["hospitals"]:
            print(f" H{h['id']} z{h['zone']} beds{h['available_beds']}/{h['total_beds']} icu{h['icu_available']} specs{h['specialties']}")
        print("Ambulances (id, zone):", [(a['id'], a['zone']) for a in readable['ambulances']])

        if Q is not None:
            # pick greedy action
            row = Q[s]
            a = int(np.argmax(row))
            print("Agent selected action (ambulance_idx, hospital_idx):", a//env.n_hospitals, a%env.n_hospitals)
        else:
            a = 0
            print("No Q-table found, using action 0")

        ns, r, done, info = env.step(a)
        print("Result:", info, "Reward:", r)

if __name__ == "__main__":
    demo(5)