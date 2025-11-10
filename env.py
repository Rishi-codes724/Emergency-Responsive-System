%%writefile env.py
# env.py
import random
import numpy as np
from data_generator import generate_zones, generate_hospitals, generate_ambulances, generate_patient, manhattan_zone_distance

class RuralEnv:
    """
    Simplified discrete environment.
    Episode = handle one patient event; agent picks (ambulance_idx, hospital_idx).
    State is represented as tuple: (patient_zone, severity, ambulance_zones..., hospital_beds..., hospital_specialties_flag)
    For tabular Q we compress to a small discrete state index by bucketing important features.
    """

    def __init__(self, grid=(4,4), n_hospitals=6, n_ambulances=3, max_distance_penalty=50):
        self.grid = grid
        self.cols = grid[1]
        self.zones = generate_zones(n_zones=grid[0]*grid[1], grid_size=grid, center=(17.55,78.40))
        self.n_hospitals = n_hospitals
        self.n_ambulances = n_ambulances
        self.hospitals = generate_hospitals(n_hospitals, self.zones)
        self.ambulances = generate_ambulances(n_ambulances, self.zones)
        self.max_distance_penalty = max_distance_penalty
        # Build action space size = ambulances * hospitals
        self.action_space = n_ambulances * n_hospitals
        # State buckets sizes for tabular encoding
        # severity in {0,1,2}, nearestAmb in {0..n_ambulances-1}, nearestHospHasSpec {0/1}
        self.n_states = 3 * n_ambulances * 2
        self.reset()

    def reset(self):
        # refresh dynamic hospitals & ambulances to start of episode
        self.hospitals = generate_hospitals(self.n_hospitals, self.zones)
        self.ambulances = generate_ambulances(self.n_ambulances, self.zones)
        self.patient = generate_patient(self.zones)
        self.done = False
        # compute shorthand state
        state_idx = self._encode_state(self.patient, self.ambulances, self.hospitals)
        return state_idx

    def _encode_state(self, patient, ambulances, hospitals):
        """
        Encodes to small state index for Q-table:
        -> severity (0..2)
        -> nearest ambulance index (0..n_ambulances-1)
        -> is there any nearby hospital with required specialty & bed (0/1)
        """
        sev = patient["severity"]
        # nearest ambulance
        dists = [manhattan_zone_distance(patient["zone"], a["zone"], cols=self.cols) for a in ambulances]
        nearest_amb = int(np.argmin(dists))
        # hospital match flag
        spec = patient["required_specialty"]
        hosp_has = 0
        if spec is None:
            hosp_has = 1
        else:
            for h in hospitals:
                if h["specialties"].get(spec, False) and h["available_beds"]>0:
                    hosp_has = 1
                    break
        idx = sev * (self.n_ambulances*2) + nearest_amb*2 + hosp_has
        return idx

    def step(self, action):
        """
        action is integer in [0, n_ambulances*n_hospitals)
        flow:
         - decode ambulance idx, hospital idx
         - if ambulance not idle -> heavy penalty (invalid)
         - compute travel distance (manhattan zones), travel time ~ distance * speed_factor
         - check hospital beds/specialty
         - compute reward
         - update hospital bed (occupy if successful), mark ambulance busy (for single-episode it's end)
         - return next_state (dummy), reward, done, info
        """
        amb_idx = action // self.n_hospitals
        hosp_idx = action % self.n_hospitals
        info = {}
        # validate indices
        if amb_idx >= self.n_ambulances or hosp_idx >= self.n_hospitals:
            return None, -50.0, True, {"reason":"invalid_action"}

        amb = self.ambulances[amb_idx]
        hosp = self.hospitals[hosp_idx]
        patient = self.patient

        # If ambulance busy (should not happen unless user manipulated), penalty
        if amb["status"] != "idle":
            return None, -40.0, True, {"reason":"ambulance_not_idle"}

        # Compute distance in zone hops
        dist = manhattan_zone_distance(patient["zone"], amb["zone"], cols=self.cols) + manhattan_zone_distance(amb["zone"], hosp["zone"], cols=self.cols)
        # Travel time minutes - assume 20 km/h ~ 20 zone units per hour? We'll use a scale
        travel_time = dist * 6.0  # each hop ~6 minutes for rural slow roads
        # Success criteria
        success = True
        reason = ""
        reward = 0.0

        # check bed availability and specialty
        spec = patient["required_specialty"]
        spec_ok = True
        if spec is not None and not hosp["specialties"].get(spec, False):
            spec_ok = False

        if patient["severity"] == 2:  # critical - need ICU if possible
            if hosp["icu_available"] <= 0:
                # heavy penalty if ICU needed but not available
                reward -= 40.0
                success = False
                reason += "no_icu;"

        if hosp["available_beds"] <= 0:
            reward -= 20.0
            success = False
            reason += "no_beds;"

        if not spec_ok:
            reward -= 10.0
            reason += "no_specialty;"

        # time penalty - larger for critical patients
        time_penalty = travel_time
        if patient["severity"] == 2:
            time_penalty *= 1.5
        reward -= time_penalty / 2.0  # normalize

        # success bonus
        if success:
            reward += 50.0  # successful delivery bonus
            # occupy bed
            if hosp["available_beds"]>0:
                hosp["available_beds"] -= 1
            if patient["severity"]==2 and hosp["icu_available"]>0:
                hosp["icu_available"] -= 1
        else:
            # partial small penalty already applied; mark as failed
            reward -= 5.0

        # finalize
        self.done = True
        next_state = self._encode_state(self.patient, self.ambulances, self.hospitals)
        info.update({
            "ambulance": amb_idx,
            "hospital": hosp_idx,
            "dist_hops": dist,
            "travel_time_min": travel_time,
            "success": success,
            "reason": reason
        })
        return next_state, reward, self.done, info

    def render_state_readable(self):
        s = {
            "patient": self.patient,
            "ambulances": self.ambulances,
            "hospitals": self.hospitals
        }
        return s