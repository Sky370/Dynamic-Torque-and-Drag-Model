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
    a1, a2, a3, a4, a5, a6 = (
        constants.a1, constants.a2, constants.a3, 
        constants.a4, constants.a5, constants.a6
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
         (a5 <= time_array)],
        [
        # Ramp up phase
        lambda t: constants.v1 * t/a1,
        # Constant drilling phase
        lambda t: constants.v1,
        # Ramp down to stop
        lambda t: constants.v1 * (a3 - t)/(a3 - a2),
        # Ramp down to negative (tripping out)
        lambda t: -constants.v2 * (t - a3)/(a4 - a3),
        # Constant tripping phase
        lambda t: -constants.v2,
        # Ramp up to stop
        lambda t: -constants.v2 * (a6 - t)/(a6 - a5)
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

def topdrive_new(t, constants, prev_ax_disp=0, prev_rot_disp=0):
    # Time intervals definition
    a1, a2, a3, a4, a5, a6 = (
        constants.a1, constants.a2, constants.a3, 
        constants.a4, constants.a5, constants.a6
    )
    b1, b2, b3, b4, b5, b6 = (
        constants.b1, constants.b2, constants.b3, 
        constants.b4, constants.b5, constants.b6
    )

    v1 = constants.v1
    v2 = constants.v2
    RPM1 = constants.rpm1
    RPM2 = constants.rpm2

    if np.isscalar(t):
        # Scalar case: Compute axial and rotational velocities at a single time point
        if 0 <= t < a1:
            ROP_topdrive = ((v1 / a1)) * t
            z_top_drive = ROP_topdrive * 0.5 * t
        elif a1 <= t < a2:
            ROP_topdrive = v1
            z_top_drive = ROP_topdrive * (t - 0.5 * a1)
        elif a2 <= t < a3:
            ROP_topdrive = ((v1) * (a3 - t)) / (a3 - a2)
            z_top_drive = v1 * (a2 - 0.5 * a1) + (0.5 * v1 * (t - a2) * (2 * a3 - a2 - t)) / (a3 - a2)
        elif a3 <= t < a4:
            ROP_topdrive = ((v2) * (a3 - t)) / (a4 - a3)
            z_top_drive = 0.5 * (a3 + a2 - a1) * v1 - (0.5 * (t - a3) ** 2) * (v2 / (a4 - a3))
        elif a4 <= t < a5:
            ROP_topdrive = -(v2)
            z_top_drive = 0.5 * (a3 + a2 - a1) * v1 - 0.5 * (a4 - a3) * v2 - (t - a4) * v2
        elif a5 <= t <= a6:
            ROP_topdrive = ((v2) * (t - a6)) / (a6 - a5)
            z_top_drive = 0.5 * (a3 + a2 - a1) * v1 - 0.5 * (a6 - a3 + a5 - a4) * v2

        if 0 <= t < b1:
            RPM_topdrive = ((RPM1 / b1)) * t
            theta_top_drive = RPM_topdrive * 0.5 * t
        elif b1 <= t < b2:
            RPM_topdrive = RPM1
            theta_top_drive = RPM_topdrive * (t - 0.5 * b1)
        elif b2 <= t < b3:
            RPM_topdrive = ((RPM1) * (b3 - t)) / (b3 - b2)
            theta_top_drive = RPM1 * (b2 - 0.5 * b1) + (0.5 * RPM1 * (t - b2) * (2 * b3 - b2 - t)) / (b3 - b2)
        elif b3 <= t < b4:
            RPM_topdrive = ((RPM2) * (b3 - t)) / (b4 - b3)
            theta_top_drive = 0.5 * (b3 + b2 - b1) * RPM1 - (0.5 * (t - b3) ** 2) * (RPM2 / (b4 - b3))
        elif b4 <= t < b5:
            RPM_topdrive = -(RPM2)
            theta_top_drive = 0.5 * (b3 + b2 - b1) * RPM1 - 0.5 * (b4 - b3) * RPM2 - (t - b4) * RPM2
        elif b5 <= t <= b6:
            RPM_topdrive = ((RPM2) * (t - b6)) / (b6 - b5)
            theta_top_drive = 0.5 * (b3 + b2 - b1) * RPM1 - 0.5 * (b6 - b3 + b5 - b4) * RPM2

        return ROP_topdrive, RPM_topdrive, z_top_drive, theta_top_drive

    else:
        # Array case (original implementation)
        axial_velocity = np.zeros_like(t)
        rotational_velocity = np.zeros_like(t)

        axial_velocity[(0 <= t) & (t < a1)] = v1 * (t[(0 <= t) & (t < a1)] / a1)
        axial_velocity[(a1 <= t) & (t < a2)] = v1
        axial_velocity[(a2 <= t) & (t < a3)] = v1 * (a3 - t[(a2 <= t) & (t < a3)]) / (a3 - a2)
        axial_velocity[(a3 <= t) & (t < a4)] = -v2 * (t[(a3 <= t) & (t < a4)] - a3) / (a4 - a3)
        axial_velocity[(a4 <= t) & (t < a5)] = -v2
        axial_velocity[(a5 <= t) & (t <= a6)] = -v2 * (a6 - t[(a5 <= t) & (t <= a6)]) / (a6 - a5)

        rotational_velocity[(0 <= t) & (t < b1)] = RPM1 * (t[(0 <= t) & (t < b1)] / b1)
        rotational_velocity[(b1 <= t) & (t < b2)] = RPM1
        rotational_velocity[(b2 <= t) & (t < b3)] = RPM1 * (b3 - t[(b2 <= t) & (t < b3)]) / (b3 - b2)
        rotational_velocity[(b3 <= t) & (t < b4)] = -RPM2 * (t[(b3 <= t) & (t < b4)] - b3) / (b4 - b3)
        rotational_velocity[(b4 <= t) & (t < b5)] = -RPM2
        rotational_velocity[(b5 <= t) & (t <= b6)] = -RPM2 * (b6 - t[(b5 <= t) & (t <= b6)]) / (b6 - b5)

        ax_disp = cumtrapz(axial_velocity, t, initial=0)  
        rot_disp = cumtrapz(rotational_velocity, t, initial=0)  

        return axial_velocity, rotational_velocity, ax_disp, rot_disp
