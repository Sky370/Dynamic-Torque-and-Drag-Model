from scipy.integrate import cumtrapz
import numpy as np
from .Init_xls import *

def topdrive_interpolator(constants):
    # Create a dense time array for precomputation
    t_max = max(constants.a6, constants.b6)
    time_array = np.linspace(0, t_max, 10000)
    
    # Precompute velocity profiles
    axial_vel = np.zeros_like(time_array)
    rotational_vel = np.zeros_like(time_array)
    
    # Time intervals definition
    a1, a2, a3, a4, a5, a6, a7 = (
        constants.a1, constants.a2, constants.a3, 
        constants.a4, constants.a5, constants.a6, constants.a7
    )
    b1, b2, b3, b4, b5, b6 = (
        constants.b1, constants.b2, constants.b3, 
        constants.b4, constants.b5, constants.b6
    )

    # Axial velocity profile
    axial_vel = np.piecewise(time_array,
        [time_array < a1,
        (a1 <= time_array) & (time_array < a2),
        (a2 <= time_array) & (time_array < a3),
        (a3 <= time_array) & (time_array < a4),
        (a4 <= time_array) & (time_array < a5),
        (a5 <= time_array) & (time_array < a6),
        (a6 <= time_array)],
        [
        # Ramp up phase
        lambda t: constants.v1 * t/a1,
        # Constant drilling phase
        lambda t: constants.v1,
        # Ramp down to stop
        lambda t: constants.v1 * (a3 - t)/(a3 - a2),
        # Pause at zero velocity
        lambda t: 0,
        # Ramp down to negative (tripping out)
        lambda t: -constants.v2 * (t - a4)/(a5 - a4),
        # Constant tripping phase
        lambda t: -constants.v2,
        # Ramp up to stop
        lambda t: -constants.v2 * (a7 - t)/(a7 - a6)
        ]
    )

    # Rotational velocity profile
    rotational_vel = np.piecewise(time_array,
        [time_array < b1,
         (b1 <= time_array) & (time_array < b2),
         (b2 <= time_array) & (time_array < b3),
         (b3 <= time_array) & (time_array < b4),
         (b4 <= time_array) & (time_array < b5),
         (b5 <= time_array)],
        [
        # Ramp up phase
        lambda t: constants.rpm1 * t/b1,
        # Constant drilling phase
        lambda t: constants.rpm1,
        # Ramp down to stop
        lambda t: constants.rpm1 * (b3 - t)/(b3 - b2),
        # Ramp down to negative (tripping out)
        lambda t: -constants.rpm2 * (t - b3)/(b4 - b3),
        # Constant tripping phase
        lambda t: -constants.rpm2,
        # Ramp up to stop
        lambda t: -constants.rpm2 * (b6 - t)/(b6 - b5)
        ]
    )

    # Compute displacements using cumulative integration
    z_top_drive = cumtrapz(axial_vel, time_array, initial=0)
    theta_top_drive = cumtrapz(rotational_vel, time_array, initial=0)

    return {
        'ROP': interp1d(time_array, axial_vel, kind='linear', fill_value="extrapolate"),
        'RPM': interp1d(time_array, rotational_vel, kind='linear', fill_value="extrapolate"),
        'z': interp1d(time_array, z_top_drive, kind='linear', fill_value="extrapolate"),
        'theta': interp1d(time_array, theta_top_drive, kind='linear', fill_value="extrapolate")
    }

def topdrive(t, interps):
    # Get values at requested time
    return (
        float(interps['ROP'](t)),
        float(interps['RPM'](t)),
        float(interps['z'](t)),
        float(interps['theta'](t))
    )

#########################
