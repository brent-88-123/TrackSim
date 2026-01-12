"""
vehicle_model.py

Separate vehicle model file containing:
- Vehicle parameters
- Tire model
- Suspension model

Import this into your lap simulator.
"""

import numpy as np

class TireModel:
    """Tire model with various formulations."""
    
    def __init__(self, model_type='linear'):
        """
        Initialize tire model.
        
        Parameters:
        -----------
        model_type : str
            'linear', 'pacejka', or 'fiala'
        """
        self.model_type = model_type
        
        # Linear model parameters
        self.Cy = 80000.0  # Lateral stiffness (N/rad)
        self.mu_peak = 1.2  # Peak friction coefficient
        self.mu_slide = 1.0  # Sliding friction coefficient
        
        # Pacejka Magic Formula parameters (simplified)
        self.B = 10.0   # Stiffness factor
        self.C = 1.3    # Shape factor
        self.D = 1.0    # Peak factor (multiplied by Fz)
        self.E = -2.0   # Curvature factor
    
    def lateral_force(self, alpha, Fz, model=None):
        """
        Calculate lateral tire force.
        
        Parameters:
        -----------
        alpha : float
            Slip angle (rad)
        Fz : float
            Normal load (N)
        model : str, optional
            Override default model type
        """
        if model is None:
            model = self.model_type
        
        if model == 'linear':
            return self._linear_model(alpha, Fz)
        elif model == 'pacejka':
            return self._pacejka_model(alpha, Fz)
        elif model == 'fiala':
            return self._fiala_model(alpha, Fz)
        else:
            raise ValueError(f"Unknown tire model: {model}")
    
    def _linear_model(self, alpha, Fz):
        """Simple linear tire model with saturation."""
        Fy_linear = self.Cy * alpha
        Fy_max = self.mu_peak * Fz
        return np.clip(Fy_linear, -Fy_max, Fy_max)
    
    def _pacejka_model(self, alpha, Fz):
        """
        Simplified Pacejka Magic Formula.
        Fy = D * sin(C * atan(B * alpha - E * (B * alpha - atan(B * alpha))))
        """
        D = self.D * self.mu_peak * Fz
        
        x = self.B * alpha
        y = x - np.arctan(x)
        Fy = D * np.sin(self.C * np.arctan(x - self.E * y))
        
        return Fy
    
    def _fiala_model(self, alpha, Fz):
        """Fiala brush tire model."""
        # Critical slip angle
        alpha_sl = np.arctan(3 * self.mu_peak * Fz / self.Cy)
        
        abs_alpha = np.abs(alpha)
        
        if abs_alpha < alpha_sl:
            # Adhesion region
            term1 = self.mu_peak * Fz
            term2 = (1 - self.Cy * abs_alpha / (3 * self.mu_peak * Fz))
            term3 = self.Cy * abs_alpha / (3 * self.mu_peak * Fz)
            Fy = term1 * (1 - term2**3) * np.sign(alpha)
        else:
            # Sliding region
            Fy = self.mu_slide * Fz * np.sign(alpha)
        
        return Fy
    
    def combined_slip(self, alpha, kappa, Fz):
        """
        Combined lateral and longitudinal slip (simplified).
        
        Parameters:
        -----------
        alpha : float
            Slip angle (rad)
        kappa : float
            Longitudinal slip ratio
        Fz : float
            Normal load (N)
        """
        # Total slip magnitude
        s_total = np.sqrt(alpha**2 + kappa**2)
        
        if s_total < 1e-6:
            return 0.0, 0.0
        
        # Forces in slip direction
        F_total = self.lateral_force(s_total, Fz)
        
        # Project onto x and y
        Fx = F_total * kappa / s_total
        Fy = F_total * alpha / s_total
        
        return Fx, Fy

class SuspensionModel:
    """Quarter-car or full vehicle suspension model."""
    
    def __init__(self, model_type='quarter_car'):
        """
        Initialize suspension model.
        
        Parameters:
        -----------
        model_type : str
            'quarter_car', 'half_car', or 'full_car'
        """
        self.model_type = model_type
        
        # Quarter car parameters (per corner)
        self.m_s = 375.0   # Sprung mass (kg) - 1/4 of vehicle
        self.m_u = 40.0    # Unsprung mass (kg) - wheel/brake/suspension
        self.k_s = 50000.0  # Spring stiffness (N/m)
        self.c_s = 3000.0   # Damper rate (N*s/m)
        self.k_t = 200000.0 # Tire stiffness (N/m)
        self.c_t = 100.0    # Tire damping (N*s/m)
        
        # Anti-roll bar
        self.k_arb = 10000.0  # ARB stiffness (N*m/rad)
        
    def quarter_car_forces(self, z_s, dz_s, z_u, dz_u, z_road):
        """
        Quarter car suspension forces.
        
        Parameters:
        -----------
        z_s : float
            Sprung mass displacement (m)
        dz_s : float
            Sprung mass velocity (m/s)
        z_u : float
            Unsprung mass displacement (m)
        dz_u : float
            Unsprung mass velocity (m/s)
        z_road : float
            Road height (m)
        
        Returns:
        --------
        F_s : float
            Force on sprung mass (N)
        F_u : float
            Force on unsprung mass (N)
        """
        # Spring and damper forces
        F_spring = -self.k_s * (z_s - z_u)
        F_damper = -self.c_s * (dz_s - dz_u)
        
        # Tire force
        F_tire = -self.k_t * (z_u - z_road) - self.c_t * (dz_u)
        
        # Forces on masses
        F_s = F_spring + F_damper
        F_u = -F_spring - F_damper + F_tire
        
        return F_s, F_u
    
    def ride_frequency(self):
        """Calculate natural frequency (Hz)."""
        omega_n = np.sqrt(self.k_s / self.m_s)
        return omega_n / (2 * np.pi)
    
    def damping_ratio(self):
        """Calculate damping ratio."""
        c_crit = 2 * np.sqrt(self.k_s * self.m_s)
        return self.c_s / c_crit

class VehicleParameters:
    """Container for all vehicle parameters."""
    
    def __init__(self):
        # Mass properties
        self.m = 1500.0      # Total mass (kg)
        self.Ixx = 500.0     # Roll inertia (kg*m^2)
        self.Iyy = 2500.0    # Pitch inertia (kg*m^2)
        self.Izz = 2000.0    # Yaw inertia (kg*m^2)
        
        # Geometry
        self.wheelbase = 2.5  # Wheelbase (m)
        self.lf = 1.2         # CG to front axle (m)
        self.lr = 1.3         # CG to rear axle (m)
        self.track_f = 1.5    # Front track width (m)
        self.track_r = 1.5    # Rear track width (m)
        self.h_cg = 0.5       # CG height (m)
        
        # Aerodynamics
        self.Cd = 0.30       # Drag coefficient
        self.Cl = 2.0        # Downforce coefficient
        self.A = 2.0         # Frontal area (m^2)
        self.rho = 1.225     # Air density (kg/m^3)
        self.cp_aero = 1.5   # Aero center of pressure from front axle (m)
        
        # Powertrain
        self.P_max = 300e3   # Max power (W)
        self.rpm_max = 8000  # Max engine RPM
        self.gear_ratios = [3.5, 2.5, 1.8, 1.4, 1.0, 0.8]
        self.final_drive = 4.0
        self.wheel_radius = 0.32  # Effective rolling radius (m)
        
        # Brakes
        self.brake_bias = 0.6  # Front brake bias (0-1)
        self.F_brake_max = 15000.0  # Max total brake force (N)
        
        # Initialize subsystems
        self.tire_fl = TireModel(model_type='linear')
        self.tire_fr = TireModel(model_type='linear')
        self.tire_rl = TireModel(model_type='linear')
        self.tire_rr = TireModel(model_type='linear')
        
        self.suspension = SuspensionModel(model_type='quarter_car')
    
    def aero_forces(self, v):
        """
        Calculate aerodynamic forces.
        
        Parameters:
        -----------
        v : float
            Velocity (m/s)
        
        Returns:
        --------
        F_drag : float
            Drag force (N)
        F_down : float
            Downforce (N)
        M_pitch : float
            Pitching moment (N*m)
        """
        if v < 0.1:
            return 0.0, 0.0, 0.0
        
        q = 0.5 * self.rho * v**2
        
        F_drag = self.Cd * self.A * q
        F_down = self.Cl * self.A * q
        
        # Pitching moment (downforce acts at cp_aero)
        M_pitch = F_down * (self.cp_aero - self.lf)
        
        return F_drag, F_down, M_pitch
    
    def load_transfer_longitudinal(self, ax):
        """
        Calculate longitudinal load transfer.
        
        Parameters:
        -----------
        ax : float
            Longitudinal acceleration (m/s^2)
        
        Returns:
        --------
        dFz_f : float
            Change in front axle load (N)
        dFz_r : float
            Change in rear axle load (N)
        """
        dFz = self.m * ax * self.h_cg / self.wheelbase
        return dFz, -dFz
    
    def load_transfer_lateral(self, ay, axle='front'):
        """
        Calculate lateral load transfer at an axle.
        
        Parameters:
        -----------
        ay : float
            Lateral acceleration (m/s^2)
        axle : str
            'front' or 'rear'
        
        Returns:
        --------
        dFz_left : float
            Change in left wheel load (N)
        dFz_right : float
            Change in right wheel load (N)
        """
        track = self.track_f if axle == 'front' else self.track_r
        
        # Roll gradient and load transfer
        # Simplified: assumes rigid chassis
        dFz = self.m * ay * self.h_cg / (2 * track)
        
        return dFz, -dFz
    
    def print_summary(self):
        """Print vehicle summary."""
        print("=" * 50)
        print("VEHICLE PARAMETERS SUMMARY")
        print("=" * 50)
        print(f"Mass: {self.m:.0f} kg")
        print(f"Wheelbase: {self.wheelbase:.2f} m")
        print(f"Weight Distribution: {self.lf/self.wheelbase*100:.1f}% rear")
        print(f"CG Height: {self.h_cg:.3f} m")
        print(f"Max Power: {self.P_max/1000:.0f} kW")
        print(f"Tire Model: {self.tire_fl.model_type}")
        print(f"Suspension Ride Freq: {self.suspension.ride_frequency():.2f} Hz")
        print(f"Damping Ratio: {self.suspension.damping_ratio():.3f}")
        print("=" * 50)

# Example usage
if __name__ == "__main__":
    # Create vehicle
    vehicle = VehicleParameters()
    vehicle.print_summary()
    
    # Test tire model
    print("\nTire Force Test:")
    alpha_test = np.linspace(-0.2, 0.2, 50)
    Fz_test = 5000.0
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    for model in ['linear', 'pacejka', 'fiala']:
        vehicle.tire_fl.model_type = model
        Fy = [vehicle.tire_fl.lateral_force(a, Fz_test) for a in alpha_test]
        plt.plot(np.degrees(alpha_test), np.array(Fy)/1000, label=model)
    
    plt.xlabel('Slip Angle (deg)')
    plt.ylabel('Lateral Force (kN)')
    plt.title('Tire Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()