from __future__ import annotations
import numpy as np
from video_utils import create_video, open_video
from scipy.ndimage import gaussian_filter
from config import (
    TIME_STEPS, TIME_DELTA, SIZE, X, Y, center_x, center_y,
    POTENTIAL_STRENGTH, NUCLEAR_REPULSION_STRENGTH, NUCLEAR_CORE_RADIUS,
    COULOMB_STRENGTH, STRONG_FORCE_STRENGTH, STRONG_FORCE_RANGE
)
# Frame utilities
from frame_utils import (
    normalize_wavefunction, apply_absorbing_edge, apply_low_pass_filter
)
# Particle utilities
from particles import create_orbital_electron
# Physics utilities
from physics import (
    create_nucleus_potential, compute_force_from_density,
    compute_nuclear_force, propagate_wave_with_potential
)


# Hydrogen nuclei positions (pixels)
nucleus_offset_px = 60  # Distance between nuclei in pixels
nucleus1_x, nucleus1_y = center_x - nucleus_offset_px, center_y
nucleus2_x, nucleus2_y = center_x + nucleus_offset_px, center_y

# Helpers

def normalize_wavefunction(psi):
    """Normalize psi so that sum(|psi|^2)==1"""
    norm = np.sqrt(np.sum(np.abs(psi)**2))
    return psi / norm if norm > 0 else psi


def create_nucleus_potential(x, y, nx, ny, charge=1):
    r = np.sqrt((x - nx)**2 + (y - ny)**2)
    r = np.maximum(r, 0.1)
    coulomb = -POTENTIAL_STRENGTH * charge / r
    rep = NUCLEAR_REPULSION_STRENGTH * np.exp(-r / NUCLEAR_CORE_RADIUS) / (r + 0.1)
    return coulomb + rep


def add_mean_field_coulomb_repulsion(psi, strength=0.05, sigma=5):
    density = np.abs(psi)**2
    return strength * gaussian_filter(density, sigma=sigma)


def compute_force_from_density(density, pos):
    dx = X - pos[0]; dy = Y - pos[1]
    r = np.sqrt(dx**2 + dy**2); r = np.maximum(r, 1.0)
    fx = np.sum((dx / r**3) * density)
    fy = np.sum((dy / r**3) * density)
    return np.array([fx, fy])


def compute_nuclear_force(p1, p2):
    diff = p2 - p1; r = np.linalg.norm(diff); r = max(r, 0.5)
    col = COULOMB_STRENGTH * diff / r**3
    if r < STRONG_FORCE_RANGE:
        strg = -STRONG_FORCE_STRENGTH * np.exp(-r/STRONG_FORCE_RANGE) * diff / r
    else:
        strg = np.zeros(2)
    return col + strg

# Wavefunction constructor (pixel-based orbital radius)

def create_orbital_electron(x, y, cx, cy, radius_px, n, l, m):
    dx = x - cx; dy = y - cy
    r_px = np.sqrt(dx*dx + dy*dy)
    theta = np.arctan2(dy, dx)
    radial = np.exp(-r_px/(n*radius_px)) * (r_px/radius_px)**l
    angular = np.exp(1j*m*theta)
    envelope = np.exp(-r_px**2/(2*(radius_px*2)**2))
    return (radial * angular * envelope).astype(np.complex128)

# Low-pass filter

def apply_low_pass_filter(psi, cutoff):
    psi_hat = np.fft.fft2(psi)
    kx = np.fft.fftfreq(psi.shape[1])*2*np.pi
    ky = np.fft.fftfreq(psi.shape[0])*2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    mask = (KX**2 + KY**2) <= cutoff**2
    return np.fft.ifft2(psi_hat * mask)

# Absorbing edge

def apply_absorbing_edge(psi, strength=0.5):
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    absorb_r = min(SIZE * 0.35, center_x - 30)
    mask = np.ones_like(r)
    phase_damping = np.ones_like(r, dtype=np.complex128)
    outside = r > absorb_r
    mask[outside] = np.exp(-strength * (r[outside] - absorb_r))
    phase_damping[outside] = np.exp(-1j * strength * (r[outside] - absorb_r))
    return psi * mask * phase_damping

# Propagation via split-step

def propagate_wave_with_potential(psi, V, dt=TIME_DELTA):
    phase = np.exp(-1j*dt*V/2)
    psi *= phase
    psi_hat = np.fft.fft2(psi)
    k = np.fft.fftfreq(SIZE)*2*np.pi
    KX, KY = np.meshgrid(k, k)
    psi = np.fft.ifft2(psi_hat * np.exp(-1j*dt*(KX**2+KY**2)/2))
    return psi * phase

# Initialize electrons
orb_px = 30
psi1 = create_orbital_electron(X, Y, nucleus1_x, nucleus1_y, orb_px, 2,0,0)
psi2 = create_orbital_electron(X, Y, nucleus2_x, nucleus2_y, orb_px, 1,0,0)
psi1 = normalize_wavefunction(psi1)
psi2 = normalize_wavefunction(psi2)
psi1 = apply_absorbing_edge(psi1, strength=1)
psi2 = apply_absorbing_edge(psi2, strength=1)

# Simulation
frames_r, frames_i, frames_ph, frames_p = [],[],[],[]
pos1, v1 = np.array([nucleus1_x,nucleus1_y],float), np.zeros(2)
pos2, v2 = np.array([nucleus2_x,nucleus2_y],float), np.zeros(2)
print("Starting sim…")
for step in range(TIME_STEPS):
    d1, d2 = np.abs(psi1)**2, np.abs(psi2)**2
    f1 = compute_force_from_density(d1, pos1)
    f2 = compute_force_from_density(d2, pos2)
    fn = compute_nuclear_force(pos1,pos2)
    v1 += TIME_DELTA*(f1-fn)/1836.0; pos1 += TIME_DELTA*v1
    v2 += TIME_DELTA*(f2+fn)/1836.0; pos2 += TIME_DELTA*v2
    V1 = create_nucleus_potential(X,Y,*pos1) + add_mean_field_coulomb_repulsion(psi2)
    V2 = create_nucleus_potential(X,Y,*pos2) + add_mean_field_coulomb_repulsion(psi1)
    psi1 = propagate_wave_with_potential(psi1, V1)
    psi2 = propagate_wave_with_potential(psi2, V2)
    psi1 = apply_absorbing_edge(psi1)
    psi2 = apply_absorbing_edge(psi2)
    psi1 = apply_low_pass_filter(psi1, cutoff=0.999)
    psi2 = apply_low_pass_filter(psi2, cutoff=0.999)
    region = psi1+psi2
    frames_r.append(np.real(region))
    frames_i.append(np.imag(region))
    frames_ph.append(np.angle(gaussian_filter(region,sigma=1)))
    p = np.abs(region)**2; frames_p.append(p/np.max(p))
# Render video
video_file = "hydrogen_sim.mkv"
create_video(TIME_STEPS, frames_r, frames_i, frames_ph, frames_p, fps=30, output_file=video_file)
open_video(video_file)
