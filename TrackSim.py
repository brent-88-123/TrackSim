import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

class TrackModel:
    """Load and interpolate track data."""
    
    def __init__(self, csv_file):
        """Load track from CSV file."""
        df = pd.read_csv(csv_file)
        self.s_data = df['s'].values
        self.length = self.s_data[-1]
        
        # Create interpolators for track properties
        self.curvature_interp = interp1d(self.s_data, df['curvature'].values, 
                                        kind='cubic', fill_value='extrapolate')
        self.elevation_interp = interp1d(self.s_data, df['elevation'].values,
                                        kind='cubic', fill_value='extrapolate')
        self.grade_interp = interp1d(self.s_data, df['grade'].values,
                                        kind='cubic', fill_value='extrapolate')
        self.bank_interp = interp1d(self.s_data, df['bank_angle'].values,
                                        kind='cubic', fill_value='extrapolate')
        self.dz_ds_interp = interp1d(self.s_data, df['dz_ds'].values,
                                        kind='cubic', fill_value='extrapolate')
    
    def get_track_properties(self, s):
        """Get track properties at distance s."""
        # Handle wrap-around for closed tracks
        s_wrapped = s % self.length
        
        return {
            'curvature': float(self.curvature_interp(s_wrapped)),
            'elevation': float(self.elevation_interp(s_wrapped)),
            'grade': float(self.grade_interp(s_wrapped)),
            'bank': float(self.bank_interp(s_wrapped)),
            'dz_ds': float(self.dz_ds_interp(s_wrapped))
        }

class VehicleModel:
    """Simple 2DOF vehicle model with suspension."""
    
    def __init__(self):
        # Vehicle parameters
        self.m = 1500.0  # Mass (kg)
        self.Iz = 2000.0  # Yaw inertia (kg*m^2)
        self.lf = 1.2  # Distance CG to front axle (m)
        self.lr = 1.3  # Distance CG to rear axle (m)
        self.h = 0.5  # CG height (m)
        self.track = 1.5  # Track width (m)
        
        # Tire parameters (simple linear model)
        self.Cf = 80000.0  # Front cornering stiffness (N/rad)
        self.Cr = 85000.0  # Rear cornering stiffness (N/rad)
        self.mu = 1.2  # Friction coefficient
        
        # Suspension parameters
        self.k_susp = 50000.0  # Suspension stiffness (N/m)
        self.c_susp = 3000.0   # Suspension damping (N*s/m)
        self.m_susp = 100.0    # Sprung mass per corner (kg)
        
        # Aerodynamics
        self.Cd = 0.3  # Drag coefficient
        self.A = 2.0   # Frontal area (m^2)
        self.rho = 1.225  # Air density (kg/m^3)
        self.Cl = 2.0  # Downforce coefficient
        
        # Powertrain
        self.max_power = 300e3  # Max power (W)
        self.max_brake_force = 15000.0  # Max brake force (N)
        
    def max_lat_accel(self):
        # Maximum lateral acceleration capability (m/s^2)
        return self.mu * 9.81  # ADD THIS METHOD
    
    def tire_force_lateral(self, alpha, Fz):
        """
        Compute lateral tire force.
        
        Parameters:
        -----------
        alpha : float
            Slip angle (rad)
        Fz : float
            Normal load (N)
        """
        # Simple linear model with saturation
        Fy_max = self.mu * Fz
        Fy_linear = self.Cf * alpha
        
        # Saturate at friction limit
        Fy = np.clip(Fy_linear, -Fy_max, Fy_max)
        return Fy
    
    def aero_forces(self, v):
        """Compute aerodynamic forces."""
        if v < 0.1:
            return 0.0, 0.0
        
        q = 0.5 * self.rho * v**2  # Dynamic pressure
        Fdrag = self.Cd * self.A * q
        Fdown = self.Cl * self.A * q
        
        return Fdrag, Fdown

class Track:
    def __init__(self, s, curvature, grade, bank, elevation):
        self.s = s
        self.length = s[-1]

        self.curvature = curvature
        self.grade = grade
        self.bank = bank
        self.elevation = elevation

        self.ds = s[1] - s[0]
        self.dz_ds = np.gradient(elevation, self.ds)

    def get_track_properties(self, s_query):
        s_wrapped = s_query % self.length

        idx = int(s_wrapped / self.ds)

        return {
            'curvature': self.curvature[idx],
            'grade': self.grade[idx],
            'bank': self.bank[idx],
            'elevation': self.elevation[idx],
            'dz_ds': self.dz_ds[idx]
        }

class LapSimulator:    
    def __init__(self, vehicle, track):
        self.vehicle = vehicle
        self.track = track
        self.g = 9.81
        
        # Controller gains
        self.K_lat = 2.0    # Lateral error gain
        self.K_heading = 8.0  # Heading error gain
    
    def calculate_speed_limit(self, kappa):
        """
        Calculate maximum sustainable speed for given curvature.
        
        This is the key to following the track!
        """
        if abs(kappa) < 1e-6:
            return 100.0  # Straight: very high limit
        
        # v_max = sqrt(a_y_max / kappa)
        # where a_y_max is the max lateral acceleration
        a_y_max = self.vehicle.max_lat_accel()
        v_max = np.sqrt(a_y_max / abs(kappa))
        
        return v_max
    
    def state_derivatives(self, t, state):
        """
        Improved dynamics with proper path following.
        
        State: [s, v, n, psi, r]
        s = distance along track (m)
        v = speed (m/s)
        n = lateral offset from centerline (m)
        psi = heading error (rad)
        r = yaw rate (rad/s)
        """
        s, v, n, psi, r = state
        
        # Avoid numerical issues at zero speed
        if v < 0.1:
            return [0.1, 0.0, 0.0, 0.0, 0.0]
        
        # Get track properties
        track_props = self.track.get_track_properties(s)
        kappa = track_props['curvature']
        grade = track_props['grade']
        
        # Calculate speed limit for this corner
        v_limit = self.calculate_speed_limit(kappa)
        
        # Aero forces
        Fdrag, Fdown = self.vehicle.aero_forces(v)
        
        # Normal loads (static + aero)
        Fz_static = self.vehicle.m * self.g
        Fz_total = Fz_static + Fdown
        Fz_f = Fz_total * self.vehicle.lr / (self.vehicle.lf + self.vehicle.lr)
        Fz_r = Fz_total * self.vehicle.lf / (self.vehicle.lf + self.vehicle.lr)
        
        # ========== PATH FOLLOWING CONTROLLER ==========
        # Calculate desired lateral acceleration to follow track
        # Use feedback on lateral error and heading error
        
        # Desired yaw rate to follow track curvature
        r_desired = v * kappa
        
        # Add corrective terms for path errors
        r_desired += -self.K_heading * psi - self.K_lat * n / max(v, 1.0)
        
        # Desired lateral acceleration
        ay_desired = v * r_desired
        
        # Saturate at tire limits
        ay_max = self.vehicle.max_lat_accel()
        ay_desired = np.clip(ay_desired, -ay_max, ay_max)
        
        # Calculate required tire slip angles (bicycle model)
        # Using steady-state assumption
        beta = np.arctan2(self.vehicle.lr * r_desired, v)  # Sideslip angle
        alpha_f = beta + self.vehicle.lf * r_desired / v - psi
        alpha_r = beta - self.vehicle.lr * r_desired / v - psi
        
        # Tire lateral forces
        Fy_f = self.vehicle.Cf * alpha_f
        Fy_r = self.vehicle.Cr * alpha_r
        
        # Saturate at friction limits
        Fy_f = np.clip(Fy_f, -self.vehicle.mu * Fz_f, self.vehicle.mu * Fz_f)
        Fy_r = np.clip(Fy_r, -self.vehicle.mu * Fz_r, self.vehicle.mu * Fz_r)
        
        # Total lateral force available
        Fy_total = Fy_f + Fy_r
        
        # ========== LONGITUDINAL CONTROLLER ==========
        # Simple speed control: accelerate if below limit, brake if above
        
        if v < 0.9 * v_limit:
            # Accelerate
            Fx_max = min(self.vehicle.max_power / max(v, 1.0), 
                        self.vehicle.mu * Fz_total)
            
            # Traction circle: reduce Fx if using lateral grip
            Fx_available = np.sqrt(max(0, (self.vehicle.mu * Fz_total)**2 - Fy_total**2))
            Fx = min(Fx_max, Fx_available)
            
        elif v > v_limit:
            # Brake
            Fx = -min(self.vehicle.max_brake_force, self.vehicle.m * (v - v_limit) * 5.0)
        else:
            # Coast
            Fx = 0.0
        
        # ========== EQUATIONS OF MOTION ==========
        
        # Longitudinal acceleration
        ax = (Fx - Fdrag - self.vehicle.m * self.g * np.sin(grade)) / self.vehicle.m
        
        # Lateral acceleration (in vehicle frame)
        ay = Fy_total / self.vehicle.m
        
        # Yaw acceleration
        r_dot = (Fy_f * self.vehicle.lf - Fy_r * self.vehicle.lr) / self.vehicle.Iz
        
        # Path coordinates (Frenet frame)
        # These convert vehicle motion to track coordinates
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        
        # Speed along track
        ds_dt = v * cos_psi / (1 - n * kappa)
        
        # Lateral deviation rate
        dn_dt = v * sin_psi
        
        # Heading error rate
        dpsi_dt = r - kappa * ds_dt
        
        # Velocity magnitude
        dv_dt = ax
        
        return [ds_dt, dv_dt, dn_dt, dpsi_dt, r_dot]
    
    def simulate_lap(self, v0=10.0, t_max=100.0):
        """Simulate one lap starting at speed v0."""
        
        # Initial state: [s, v, n, psi, r]
        state0 = [0.0, v0, 0.0, 0.0, 0.0]
        
        # Event: lap completion
        def lap_complete(t, state):
            return state[0] - self.track.length
        lap_complete.terminal = True
        lap_complete.direction = 1
        
        # Event: went way off track
        def off_track(t, state):
            return 10.0 - abs(state[2])  # n > 10m means off track
        off_track.terminal = True
        
        # Solve
        sol = solve_ivp(
            self.state_derivatives,
            [0, t_max],
            state0,
            method='RK45',  # Runge-Kutta 4/5 (good for this problem)
            events=[lap_complete, off_track],
            dense_output=True,
            max_step=0.05,
            rtol=1e-6,
            atol=1e-8
        )
        
        return sol
    
    def plot_results(self, sol):
        """Plot simulation results."""
        t = sol.t
        s = sol.y[0]
        v = sol.y[1]
        n = sol.y[2]
        psi = sol.y[3]
        z_susp = sol.y[4]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Velocity
        axes[0].plot(s, v * 3.6, 'b-', linewidth=2)
        axes[0].set_ylabel('Speed (km/h)')
        axes[0].set_title('Lap Simulation Results')
        axes[0].grid(True, alpha=0.3)
        
        # Lateral deviation
        axes[1].plot(s, n * 1000, 'g-', linewidth=2)
        axes[1].set_ylabel('Lateral Deviation (mm)')
        axes[1].grid(True, alpha=0.3)
        
        # Suspension displacement
        axes[2].plot(s, z_susp * 1000, 'r-', linewidth=2)
        axes[2].set_ylabel('Suspension Travel (mm)')
        axes[2].set_xlabel('Distance (m)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print lap time
        if sol.t_events[0].size > 0:
            lap_time = sol.t_events[0][0]
            print(f"\nLap Time: {lap_time:.2f} seconds")
        else:
            print("\nLap not completed within time limit")

# Example usage
if __name__ == "__main__":
    # Load track (assumes track file exists from previous script)
    try:
        track = TrackModel('Smooth_R30.csv')
        vehicle = VehicleModel()
        sim = LapSimulator(vehicle, track)
        
        print("Running lap simulation...")
        sol = sim.simulate_lap(v0=15.0, t_max=50.0)
        
        sim.plot_results(sol)
        
    except FileNotFoundError:
        print("Track file not found. Run the track generator script first.")