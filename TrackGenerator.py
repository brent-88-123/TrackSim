import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def create_constant_radius_track_with_bump(radius=30, track_length=188.495, 
                                            roughness_rms = 0.0015, lambda_min = 1.0,
                                            lambda_max = 3.0, ds=0.02, seed=None):
    """
    Create a constant radius circular track with a bump.
    
    Parameters:
    -----------
    radius : Radius of curvature (m)
    track_length : Total track length (m)
    bump_height : Height of bump (m)
    bump_width : Width of bump (m)
    ds : Discretization step (m)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Distance array
    s = np.arange(0, track_length + ds, ds)
    
    # Constant curvature (positive = left turn)
    curvature = np.ones_like(s) / radius
    
    # Create bump profile
    elevation = np.zeros_like(s)
    N = len(elevation)
    
    # spatial frequencies (FFT)
    k = np.fft.fftfreq(N, d=ds)
    k_min = 1 / lambda_max
    k_max = 1 / lambda_min
    
    Z = np.zeros(N, dtype=complex)

    band = (np.abs(k) >= k_min) & (np.abs(k) <= k_max)

    # generate band-limited roughness
    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(np.sum(band)))
    Z[band] = random_phase
    
    # inverse FFT → spatial elevation
    elevation = np.real(np.fft.ifft(Z))
    
    # scale to desired RMS height
    elevation *= roughness_rms / np.std(elevation)
    
    # Calculate grade (derivative of elevation)
    grade = np.zeros_like(s)
    
    # Bank angle (could add banking in corners if desired)
    bank_angle = np.zeros_like(s)
    
    # Calculate dz_ds
    
    dz_ds = np.gradient(elevation, s, edge_order = 2)
    
    # Create track data dictionary
    track_data = {
        's': s,
        'curvature': curvature,
        'elevation': elevation,
        'grade': grade,
        'bank_angle': bank_angle,
        'dz_ds' : dz_ds
    }
    
    return track_data

def save_track_to_csv(track_data, filename='track.csv'):
    """Save track data to CSV file."""
    import pandas as pd
    df = pd.DataFrame(track_data)
    df.to_csv(filename, index=False)
    print(f"Track saved to {filename}")

def plot_track(track_data):
    """Visualize the track profile."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Curvature plot
    axes[0].plot(track_data['s'], track_data['curvature'], 'b-', linewidth=2)
    axes[0].set_ylabel('Curvature (1/m)')
    axes[0].set_title('Track Profile')
    axes[0].grid(True, alpha=0.3)
    
    # Elevation plot
    axes[1].plot(track_data['s'], track_data['elevation'] * 1000, 'g-', linewidth=2)
    axes[1].set_ylabel('Elevation (mm)')
    axes[1].grid(True, alpha=0.3)
    
    # Grade plot
    axes[2].plot(track_data['s'], track_data['grade'] * 100, 'r-', linewidth=2)
    axes[2].set_ylabel('Grade (%)')
    axes[2].set_xlabel('Distance (m)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_track_path_xy(track_data):
    """Convert s-curvature to x-y coordinates for visualization."""
    s = track_data['s']
    kappa = track_data['curvature']
    
    # Integrate heading angle
    theta = np.zeros_like(s)
    for i in range(1, len(s)):
        theta[i] = theta[i-1] + kappa[i] * (s[i] - s[i-1])
    
    # Integrate x, y positions
    x = np.zeros_like(s)
    y = np.zeros_like(s)
    for i in range(1, len(s)):
        ds_val = s[i] - s[i-1]
        x[i] = x[i-1] + ds_val * np.cos(theta[i-1])
        y[i] = y[i-1] + ds_val * np.sin(theta[i-1])
    
    return x, y

def plot_track_overhead(track_data):
    """Plot overhead view of track."""
    x, y = create_track_path_xy(track_data)
    
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
    
    # Mark bump location
    bump_idx = np.argmax(track_data['elevation'])
    plt.plot(x[bump_idx], y[bump_idx], 'ro', markersize=10, label='Bump')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Track Overhead View')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create track with R=30m and a 50mm bump at 100m
    track = create_constant_radius_track_with_bump(
        radius=30.0,
        track_length=188.495,
        roughness_rms = 0.0015,
        lambda_min = 1.0,
        lambda_max = 3.0,
        ds=0.02
    )
    
    # Save to file
    save_track_to_csv(track, 'Smooth_R30.csv')
    
    # Visualize
    plot_track(track)
    plot_track_overhead(track)
    
    print(f"\nTrack Statistics:")
    print(f"Length: {track['s'][-1]:.1f} m")
    print(f"Radius: {1/track['curvature'][0]:.1f} m")
    print(f"Max elevation: {np.max(track['elevation'])*1000:.1f} mm")
    print(f"Max grade: {np.max(np.abs(track['grade']))*100:.2f} %")