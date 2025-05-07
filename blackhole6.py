import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Constants
G = 6.67430e-11  # Gravitational constant
M = 1.989e30 * 1000  # Black hole mass (1000 solar masses)
c = 3e8  # Speed of light
Rs = (2 * G * M) / (c**2)  # Schwarzschild radius
dt = 0.001
num_particles = 1000

# Custom colormap for the accretion disk
def create_custom_cmap():
    colors = [(0, 0, 0), (0.3, 0, 0.5), (0.8, 0.2, 0), (1, 0.7, 0), (1, 1, 0.8)]
    return LinearSegmentedColormap.from_list("accretion_disk", colors)

accretion_cmap = create_custom_cmap()
jet_cmap = LinearSegmentedColormap.from_list("jet", [(0, 0, 0.5), (0, 0.5, 1), (1, 1, 1)])

# Initialize particles
radii = np.random.normal(loc=5*Rs, scale=Rs, size=num_particles)
angles = np.random.uniform(0, 2*np.pi, num_particles)
x = radii * np.cos(angles)
y = radii * np.sin(angles)

speed = np.sqrt(G * M / radii) * (1 + np.random.normal(0, 0.1, num_particles))
vx = -np.sin(angles) * speed
vy = np.cos(angles) * speed

# Add direct-infall particles
num_direct = num_particles // 10
x[:num_direct] = np.random.uniform(-10*Rs, 10*Rs, num_direct)
y[:num_direct] = np.random.uniform(10*Rs, 15*Rs, num_direct)
vx[:num_direct] = np.random.normal(0, 0.2*c, num_direct)
vy[:num_direct] = -np.abs(np.random.normal(0.1*c, 0.2*c, num_direct))

positions = np.array([x, y])
velocities = np.array([vx, vy])
alphas = np.ones(num_particles)
temperatures = 1e6 / (np.sqrt(x**2 + y**2)/Rs + 0.1)

# Plot setup
fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
ax.set_aspect('equal')
ax.set_facecolor('black')
ax.set_xlim(-15*Rs, 15*Rs)
ax.set_ylim(-15*Rs, 15*Rs)
ax.axis('off')

# Background stars
num_stars = 200
star_x = np.random.uniform(-15*Rs, 15*Rs, num_stars)
star_y = np.random.uniform(-15*Rs, 15*Rs, num_stars)
star_sizes = np.random.uniform(0.1, 2, num_stars)
star_alphas = np.random.uniform(0.3, 1, num_stars)
star_plot = ax.scatter(star_x, star_y, c='white', s=star_sizes, alpha=star_alphas, zorder=0)

# Add gravitational lensing effect to stars
def lens_stars():
    global star_x, star_y
    for i in range(num_stars):
        dist = np.sqrt(star_x[i]**2 + star_y[i]**2)
        if dist < 5*Rs:
            factor = 1 + (5*Rs - dist)/(5*Rs)
            star_x[i] *= factor
            star_y[i] *= factor

lens_stars()

# Add black hole visuals
event_horizon = plt.Circle((0, 0), Rs, color='black', zorder=20)
photon_sphere = plt.Circle((0, 0), 1.5*Rs, color='white', alpha=0.05, linestyle='--', fill=False, zorder=15)
lensing_effect = plt.Circle((0, 0), 3*Rs, color='white', alpha=0.02, zorder=5)
glow = plt.Circle((0, 0), 5*Rs, color='cyan', alpha=0.1, zorder=10)

ax.add_artist(lensing_effect)
ax.add_artist(glow)
ax.add_artist(photon_sphere)
ax.add_artist(event_horizon)

# Particle scatter plots
disk_scatter = ax.scatter([], [], c=[], cmap=accretion_cmap, s=2, alpha=0.8)
jet_scatter = ax.scatter([], [], c=[], cmap=jet_cmap, s=1, alpha=0.6)

# Update function
def update(frame):
    global positions, velocities, alphas, temperatures, star_x, star_y

    dx = -positions[0]
    dy = -positions[1]
    dist_sq = dx**2 + dy**2 + 1e-16
    dist = np.sqrt(dist_sq)
    gamma = 1 / np.sqrt(1 - (velocities[0]**2 + velocities[1]**2) / c**2)
    gamma = np.clip(gamma, 1, 10)
    force = G * M / (dist_sq * gamma)
    fx = force * dx / dist
    fy = force * dy / dist

    velocities[0] += fx * dt
    velocities[1] += fy * dt
    positions[0] += velocities[0] * dt
    positions[1] += velocities[1] * dt

    dist = np.sqrt(positions[0]**2 + positions[1]**2)
    temperatures = np.where(dist < 10*Rs, 1e6 + 1e7/(dist/Rs + 0.01), temperatures)

    inside_eh = dist < Rs
    near_eh = (dist < 3*Rs) & (dist >= Rs)
    in_jet = (np.abs(positions[0]) < 2*Rs) & (positions[1] < -Rs) & (positions[1] > -5*Rs)

    alphas[inside_eh] = 0
    alphas[near_eh] *= 0.95
    alphas[in_jet] = np.minimum(alphas[in_jet] + 0.05, 0.6)

    if frame % 10 == 0 and np.sum(inside_eh) > 0:
        new_particles = np.sum(inside_eh)
        new_radii = np.random.normal(loc=5*Rs, scale=Rs, size=new_particles)
        new_angles = np.random.uniform(0, 2*np.pi, new_particles)
        positions[0][inside_eh] = new_radii * np.cos(new_angles)
        positions[1][inside_eh] = new_radii * np.sin(new_angles)
        new_speed = np.sqrt(G * M / new_radii) * (1 + np.random.normal(0, 0.1, new_particles))
        velocities[0][inside_eh] = -np.sin(new_angles) * new_speed
        velocities[1][inside_eh] = np.cos(new_angles) * new_speed
        alphas[inside_eh] = 1
        temperatures[inside_eh] = 1e6 / (new_radii/Rs + 0.1)

    disk_mask = ~in_jet
    jet_mask = in_jet

    if np.any(disk_mask):
        disk_scatter.set_offsets(np.c_[positions[0][disk_mask], positions[1][disk_mask]])
        disk_scatter.set_array(temperatures[disk_mask])
        disk_scatter.set_alpha(alphas[disk_mask])
        disk_scatter.set_sizes(np.clip(2 * (1 + np.log10(temperatures[disk_mask]/1e6)), 0.5, 5))

    if np.any(jet_mask):
        jet_scatter.set_offsets(np.c_[positions[0][jet_mask], positions[1][jet_mask]])
        jet_scatter.set_array(np.clip(velocities[1][jet_mask]/c * 1e7, 0, 1e7))
        jet_scatter.set_alpha(alphas[jet_mask] * 0.7)
        jet_scatter.set_sizes(np.clip(-velocities[1][jet_mask]/c * 10, 0.5, 3))

    if frame % 5 == 0:
        angles = np.arctan2(star_y, star_x)
        angles += 0.01
        dists = np.sqrt(star_x**2 + star_y**2)
        star_x = dists * np.cos(angles)
        star_y = dists * np.sin(angles)
        lens_stars()
        star_plot.set_offsets(np.c_[star_x, star_y])

    return disk_scatter, jet_scatter

# Animation
ani = FuncAnimation(fig, update, frames=1000, interval=20, blit=True)

plt.tight_layout()
plt.show()