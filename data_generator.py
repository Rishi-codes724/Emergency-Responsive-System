# data_generator.py
import random
import math

def generate_zones(n_zones=16, grid_size=(4,4), center=(17.55,78.40), spread_km=30):
    """
    Create discrete zones labelled 0..n_zones-1 arranged in a grid.
    For simplicity we don't use lat/lon except for display later.
    """
    rows, cols = grid_size
    assert rows*cols == n_zones, "grid_size must match n_zones"
    zones = []
    for r in range(rows):
        for c in range(cols):
            zone_id = r*cols + c
            # pseudo lat/lon for visualization (not used by RL)
            lat = center[0] + (r - rows/2) * 0.02
            lon = center[1] + (c - cols/2) * 0.02
            zones.append({"id": zone_id, "row": r, "col": c, "lat": lat, "lon": lon})
    return zones

def manhattan_zone_distance(z1, z2, cols=4):
    r1, c1 = z1//cols, z1%cols
    r2, c2 = z2//cols, z2%cols
    return abs(r1-r2) + abs(c1-c2)

def generate_hospitals(n_hospitals=6, zones=None):
    """
    Each hospital is assigned to a zone.
    Simulates specialties and bed counts.
    """
    assert zones is not None
    zone_ids = [z["id"] for z in zones]
    hospitals = []
    for i in range(n_hospitals):
        z = random.choice(zone_ids)
        total = random.randint(10, 60)
        icu_total = random.randint(0, 8)
        icu_available = random.randint(0, icu_total) if icu_total>0 else 0
        available = random.randint(0, min(10, total))
        specialties = {
            "trauma": random.choice([True, False, False]),
            "cardiac": random.choice([True, False]),
            "maternity": random.choice([True, False]),
            "pediatrics": random.choice([True, False, False]),
            "general": True
        }
        hospitals.append({
            "id": i,
            "zone": z,
            "total_beds": total,
            "available_beds": available,
            "icu_total": icu_total,
            "icu_available": icu_available,
            "specialties": specialties
        })
    return hospitals

def generate_ambulances(n_ambulances=3, zones=None):
    """
    Place ambulances in random zones; status = 0 (free)
    """
    assert zones is not None
    zone_ids = [z["id"] for z in zones]
    ambulances = []
    for i in range(n_ambulances):
        z = random.choice(zone_ids)
        ambulances.append({
            "id": i,
            "zone": z,
            "status": "idle"  # or 'busy'
        })
    return ambulances

def generate_patient(zones=None):
    """
    Patient spawns in a random zone with severity and required specialty (maybe None)
    severity: 0=moderate,1=severe,2=critical
    """
    assert zones is not None
    z = random.choice(zones)["id"]
    severity = random.choices([0,1,2], weights=[0.5,0.35,0.15])[0]
    possible_specs = ["trauma","cardiac","maternity","pediatrics", None]
    required_specialty = random.choices(possible_specs, weights=[0.15,0.15,0.10,0.10,0.5])[0]
    return {"zone": z, "severity": severity, "required_specialty": required_specialty}



