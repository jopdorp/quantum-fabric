import numpy as np
from config import SCALE, SIGMA_AMPLIFIER
from scipy.special import genlaguerre, factorial

def create_orbital_electron(x, y, cx, cy, radius_px, quantum_numbers):
    n, l, m = quantum_numbers
    dx = x - cx; dy = y - cy
    r_px = np.sqrt(dx*dx + dy*dy)
    theta = np.arctan2(dy, dx)
    radial = np.exp(-r_px/(n*radius_px)) * (r_px/radius_px)**l
    angular = np.exp(1j*m*theta)
    envelope = np.exp(-r_px**2/(2*(radius_px*2)**2))
    return (radial * angular * envelope).astype(np.complex128)

