import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

class Track:
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        self.s = df["s"].values
        self.curvature = df["curvature"].values
        self.grade = df.get("grade", 0.0).values

        self.ds = self.s[1] - self.s[0]
        self.n = len(self.s)
        self.length = self.s[-1]

class Vehicle:
    def __init__(self):
        self.m = 1500.0
        self.mu = 1.2
        self.g = 9.81

        self.max_power = 300e3      # W
        self.max_brake_force = 15000.0  # N

    def max_lat_accel(self):
        return self.mu * self.g

    def max_long_accel(self, v, ay):
        # friction limit
        a_tot = self.mu * self.g
        ax_fric = np.sqrt(max(0.0, a_tot**2 - ay**2))

        # power limit
        if v < 1.0:
            ax_power = a_tot
        else:
            ax_power = self.max_power / (self.m * v)

        return min(ax_fric, ax_power)

    def max_brake_accel(self, ay):
        a_tot = self.mu * self.g
        return np.sqrt(max(0.0, a_tot**2 - ay**2))

class ForwardBackwardLapSolver:
    def __init__(self, track, vehicle):
        self.track = track
        self.vehicle = vehicle
        self.g = vehicle.g

    def solve(self):
        s = self.track.s
        kappa = self.track.curvature
        grade = self.track.grade
        ds = self.track.ds
        n = self.track.n

        # --- 1) Corner speed limit ---
        v_lat = np.zeros(n)
        for i in range(n):
            if abs(kappa[i]) < 1e-6:
                v_lat[i] = 150.0  # straight cap
            else:
                v_lat[i] = np.sqrt(self.vehicle.max_lat_accel() / abs(kappa[i]))

        # --- 2) Forward pass (acceleration) ---
        v_fwd = np.zeros(n)
        v_fwd[0] = v_lat[0]

        for i in range(n - 1):
            v = v_fwd[i]
            ay = v**2 * abs(kappa[i])

            ax = self.vehicle.max_long_accel(v, ay)
            ax -= self.g * np.sin(grade[i])

            v_next = np.sqrt(max(0.0, v**2 + 2 * ax * ds))
            v_fwd[i + 1] = min(v_next, v_lat[i + 1])

        # --- 3) Backward pass (braking) ---
        v_bwd = np.zeros(n)
        v_bwd[-1] = v_lat[-1]

        for i in range(n - 2, -1, -1):
            v = v_bwd[i + 1]
            ay = v**2 * abs(kappa[i + 1])

            ax = self.vehicle.max_brake_accel(ay)
            ax += self.g * np.sin(grade[i])

            v_prev = np.sqrt(max(0.0, v**2 + 2 * ax * ds))
            v_bwd[i] = min(v_prev, v_lat[i])

        # --- 4) Envelope ---
        v = np.minimum(v_fwd, v_bwd)

        # --- 5) Lap time ---
        dt = ds / np.maximum(v, 0.1)
        lap_time = np.sum(dt)

        return {
            "s": s,
            "v": v,
            "v_lat": v_lat,
            "v_fwd": v_fwd,
            "v_bwd": v_bwd,
            "lap_time": lap_time
        }



# Example usage
if __name__ == "__main__":
    try:
        # --- Load track ---
        track = Track("imola_like_track.csv")

        # --- Create vehicle ---
        vehicle = Vehicle()

        # --- Create solver ---
        solver = ForwardBackwardLapSolver(track, vehicle)

        # --- Run simulation ---
        results = solver.solve()

        # --- Print summary ---
        print(f"Lap time: {results['lap_time']:.2f} s")
        print(f"Max speed: {results['v'].max() * 3.6:.1f} km/h")
        print(f"Min speed: {results['v'].min() * 3.6:.1f} km/h")

        # --- Plot speed profile ---
        plt.figure(figsize=(10, 4))
        plt.plot(results["s"], results["v"] * 3.6, label="Final speed")
        plt.plot(results["s"], results["v_lat"] * 3.6, "--", label="Corner limit")
        plt.plot(results["s"], results["v_fwd"] * 3.6, ":", label="Forward pass")
        plt.plot(results["s"], results["v_bwd"] * 3.6, ":", label="Backward pass")
        plt.xlabel("Distance [m]")
        plt.ylabel("Speed [km/h]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Error: Track file not found!")
        print("Make sure 'test_track.csv' exists in the current directory.")
