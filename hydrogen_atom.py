import numpy as np
from video_utils import create_video, open_video
from scipy.ndimage import gaussian_filter
from config import TIME_STEPS, X, Y, center_x, center_y
from particles import create_orbital_electron
from frame_utils import limit_frame
from physics import (
    create_nucleus_potential,
    compute_force_from_density,
    propagate_wave_with_potential
)



# Hydrogen nuclei positions (pixels)
nucleus1_x, nucleus1_y = center_x, center_y

# Initialize electrons
orb_px = 80
psi1 = create_orbital_electron(X, Y, nucleus1_x, nucleus1_y, orb_px, (1,0,0))
psi1 = limit_frame(psi1)

# Simulation
frames_r, frames_i, frames_ph, frames_p = [],[],[],[]
pos1, v1 = np.array([nucleus1_x,nucleus1_y],float), np.zeros(2)
print("Starting sim…")
for step in range(TIME_STEPS):
    if step % 10 == 0:
        print(f"Step {step}/{TIME_STEPS}…")
    d1 = np.abs(psi1)**2
    f1 = compute_force_from_density(d1, pos1)
    V1 = create_nucleus_potential(X,Y,*pos1)
    psi1 = propagate_wave_with_potential(psi1, V1)
    psi1 = limit_frame(psi1)
    region = psi1
    frames_r.append(np.real(region))
    frames_i.append(np.imag(region))
    frames_ph.append(np.angle(gaussian_filter(region,sigma=1)))
    p = np.abs(region)**2; frames_p.append(p/np.max(p))
# Render video
video_file = "hydrogen_atom_sim.mkv"
create_video(TIME_STEPS, frames_r, frames_i, frames_ph, frames_p, fps=30, output_file=video_file)
open_video(video_file)
