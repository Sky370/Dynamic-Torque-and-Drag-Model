from Libs.TD_velocity import topdrive
from Libs.Modules.Friction import Friction
from Libs.Modules.Bitrock import bit_rock as bit_rock
import numpy as np

def Main_Func(t, x, constants, p, fric_mod):

    """
    This function has the lumped spring mass damper dynamic equations in axial and torsional directions
    The second order ode is broken down to two first order ODE's to solve the problem
    This is the driver function to ODE solver

    Parameters
    ----------
    t : `float`,
        time in seconds

    x : `np.array`,
        global dof vector : [z,v,theta,omega]*noe

    p : `dict`,
        dictionary containing important constants

    Returns
    -------
    dx : `np.array`
        global dof_dot vector : [z_dot, v_dot, theta_dot, omega_dot]*noe

    """
    clc = constants
    
    dx = np.zeros(4 * clc.noe)
    Forcing_F = np.zeros(clc.noe)
    Forcing_T = np.zeros(clc.noe)
    ROP_top_drive, RPM_top_drive, z_top_drive, theta_top_drive = topdrive(t, constants.topdrive_interpolators)
    Forcing_F[0] = clc.ka[0] * z_top_drive
    Forcing_T[0] = clc.kt[0] * theta_top_drive

    # Checkpoint for debugging
    checkpoint = float(np.round(t, 2))
    
    mm_omega = np.zeros(clc.noe)
    # if motor_elem != "N":
    #     motor_speed = 2 * np.pi * p[MOTOR_RPG] * p[MOTOR_FLOW_RATE]*(1/60)
    #     mm_omega[motor_elem::] = motor_speed

    z = x[0::4]
    v = x[1::4]
    theta = x[2::4] - mm_omega * t
    omega = x[3::4] - mm_omega
    # theta = np.zeros(clc.noe)
    # omega = np.zeros(clc.noe)
    doc = bit_rock(z[-1], theta[-1], p, clc)

    # z[0] = z_top_drive
    v[0] = ROP_top_drive
    theta[0] = RPM_top_drive

    Forcing_F[-1] = -clc.K_WOB * doc  # - c_bit_axial * v[-1] * abs(np.sign(doc))
    Forcing_T[-1] = -clc.K_TQ * doc

    Friction_force, Friction_torque, p['STATIC_CHECK_PREV'], new_fric_force = Friction(
        z, v, theta, omega, Forcing_F, Forcing_T, p['STATIC_CHECK_PREV'], clc, fric_mod
    )

    # store weight , torque , depth of cut and solution time
    # TODO: Dense outputs needn't require storing solution time

    p['new_force'].append(new_fric_force[-1])
    p['DOWNHOLE_WEIGHT'].append(Forcing_F[-1])
    p['DOWNHOLE_TORQUE'].append(Forcing_T[-1])
    p['DOC'].append(doc)
    p['SOLUTION_TIME'].append(t)
    p['FRICTION_FORCE_STORE'].append(Friction_force)
    p['FRICTION_TORQUE_STORE'].append(Friction_torque)
    p['STATIC_CHECK_PREV_STORE'].append(p['STATIC_CHECK_PREV'])
    
    # Governing Equations
    alpha = 0.05  # Stiffness-proportional damping
    M_inv = clc.global_mass_inv_matrix
    J_inv = clc.global_inertia_inv_matrix
    Ka = clc.global_ka_matrix
    Kt = clc.global_kt_matrix
    Ca = clc.global_ca_matrix
    Ct = clc.global_ct_matrix
    # C = np.ones(clc.noe)*200
    R_EQ = 0.5*clc.DIA_EQ
    F_g = clc.bw_pipe*clc.global_length_array*np.cos(clc.inc_rad[1:])
    F_v = clc.F_visc1*20
    C_v = F_v*R_EQ

    dx[0::4] = np.array(v)
    dx[1::4] = M_inv.dot(Forcing_F + Friction_force + F_g + F_v - Ca.dot(v) - Ka.dot(z))
    dx[2::4] = np.array(omega + mm_omega)
    dx[3::4] = J_inv.dot(Forcing_T + Friction_torque - Ct.dot(omega) * R_EQ - Kt.dot(theta))

    return dx