from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import pandas as pd 
import numpy as np
import os

THIS_FOLDER = os.path.dirname(os.path.abspath("__file__"))
outputFolderPath = os.path.join(THIS_FOLDER, 'Output')
input_excel_path = os.path.join(THIS_FOLDER, 'Input/NewData_Zahra_3D.xlsx')

# Import the data
sheet_names = ["PUMP", "BHA", "ADVANCED", "SURVEY", "TOP_DRIVE", "Borehole_Properties", "steady_state_inputs"]
df_dict = pd.read_excel(input_excel_path, sheet_name=sheet_names)

df_PUMP = df_dict["PUMP"]
df_BHA = df_dict["BHA"]
df_ADV = df_dict["ADVANCED"]
df_SRV = df_dict["SURVEY"]
df_TOP = df_dict["TOP_DRIVE"]
df_WELL = df_dict["Borehole_Properties"]
df_SS = df_dict["steady_state_inputs"]

def nearestLength(a, b):
    """
    Takes two numbers a and b and tries to break a into 'n' integer parts with
    length close to b
    Parameters
    ----------
    a : `float`
        total length
    b : `float`
        element length

    Returns
    -------
    num , length : `tuple` of `float`

    """
    if a <= b:
        return 1, a

    ceil_parts = np.ceil(a / b)
    floor_parts = np.floor(a / b)
    
    ceil_length = a / ceil_parts
    floor_length = a / floor_parts
    
    # Compare which length is closer to b and return the appropriate result
    if abs(b - ceil_length) < abs(b - floor_length):
        return int(ceil_parts), ceil_length
    else:
        return int(floor_parts), floor_length

def survey_mod_SI(df, MD, bf, mass, g):
    x = ft2m(np.array(df["MD"].values))
    y = np.array(df["INC"].values)
    z = np.array(df["AZI"].values)
    dls = m2ft(np.array(df["DLS"].values))
    y_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")
    z_interp = interp1d(x, z, kind='linear', fill_value="extrapolate")
    k_interp = interp1d(x, dls, kind='linear', fill_value="extrapolate")
    theta_inclination = y_interp(MD)
    theta_azimuth = z_interp(MD)
    DLS = k_interp(MD) # deg/m
    # Inclination Angle in rad, not in 'deg'
    # Normal_force = bf * mass * g * np.sin(np.deg2rad((theta_inclination[:-1] + theta_inclination[1:])/2))
    return theta_inclination, theta_azimuth, DLS

def survey_mod_IMPERIAL(df, MD, bf, mass, g):
    x = (np.array(df["MD"].values))
    y = np.array(df["INC"].values)
    z = np.array(df["AZI"].values)
    dls = (np.array(df["DLS"].values))
    y_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")
    z_interp = interp1d(x, z, kind='linear', fill_value="extrapolate")
    k_interp = interp1d(x, dls, kind='linear', fill_value="extrapolate")
    theta_inclination = y_interp(MD)
    theta_azimuth = z_interp(MD)
    DLS = k_interp(MD) # deg/m
    # Inclination Angle in rad, not in 'deg'
    # Normal_force = bf * mass * g * np.sin(np.deg2rad((theta_inclination[:-1] + theta_inclination[1:])/2))
    return theta_inclination, theta_azimuth, DLS

# New Pressure Loss Calculation (more accurate)
def pres_calc(rho, m, K, tao, Q, D_o, D_i, D_w):
    # Conversion to SI
    D_i_new, D_o_new, D_w_new = D_i*0.0254, D_o*0.0254, D_w*0.0254  # in to m
    Q_new = Q / (264.172 * 60)                                      # GPM to m3/s
    rho_new = rho * 119.8264273167                                  # lbm/ft^3 to kg/m^3
    tao_yield = tao * 0.4788                                        # psi to Pa
    K_new = K / 1000                                          # cp to Pa.s

    # Calculation part
    D_hy = D_w_new - D_o_new                            # Hydraulic diameter
    Area_an = np.pi / 4 * (D_w_new**2 - D_o_new**2)     # Annular area
    Area_in = np.pi / 4 * D_i_new**2                    # Inner area
    v_in = Q_new / Area_in
    v_an = Q_new / Area_an

    def shear_stress_iteration(v, D, is_annular=True):
        # Iterative calculation for shear stress
        tao_initial = tao_yield + K_new * ((12 if is_annular else 8) * v / D)**m
        tao_new = np.copy(tao_initial)
        tolerance = np.ones_like(tao_initial)

        while np.any(tolerance > 1e-4):
            x = tao_yield / tao_initial
            if is_annular:
                C_c = (1-x)*((m*x/(1+m))+1)
                D_e = 3*m/(2*m+1)*C_c*D
            else:
                C_c = (1-x)*(2*(m*x)**2/((1+2*m)*(1+m))+2*m*x/(1+2*m)+1)
                D_e = 4*m/(3*m+1)*C_c*D

            shear_rate = (12 if is_annular else 8) * v / D_e
            tao_new = tao_yield + K_new * shear_rate**m
            tolerance = np.abs(tao_new - tao_initial)
            tao_initial = np.copy(tao_new)

        return tao_new

    # Iteration for shear stress in annulus and inner sections
    tao_new_an = shear_stress_iteration(v_an, D_hy, is_annular=True)
    tao_new_in = shear_stress_iteration(v_in, D_i_new, is_annular=False)

    # Reynolds number and friction factor calculations
    def calculate_friction_factor(N_RE, N, is_annular=True):
        N_RE_Crit = 3250 - 1150 * N
        laminar_mask = N_RE < N_RE_Crit
        fric_f = np.zeros_like(N_RE)
        fric_f[laminar_mask] = 24 / N_RE[laminar_mask] if is_annular else 16 / N_RE[laminar_mask]

        turbulent_mask = ~laminar_mask
        if np.any(turbulent_mask):
            def equation(f_f):
                return (1 / np.sqrt(f_f)) - (4 / (N[turbulent_mask] ** 0.75)) * \
                    np.log(N_RE[turbulent_mask] * (f_f ** (1 - N[turbulent_mask] / 2))) + (0.4 / (N[turbulent_mask] ** 1.2))

            initial_guess = np.full_like(N_RE[turbulent_mask], 0.01)
            fric_f[turbulent_mask] = fsolve(equation, initial_guess)

        return fric_f

    N_RE_an = 12 * rho_new * v_an**2 / tao_new_an
    N_RE_in = 8 * rho_new * v_in**2 / tao_new_in
    N_an = np.log(tao_new_an) / np.log(12 * v_an / D_hy)
    N_in = np.log(tao_new_in) / np.log(8 * v_in / D_i_new)

    fric_f_an = calculate_friction_factor(N_RE_an, N_an, is_annular=True)
    fric_f_in = calculate_friction_factor(N_RE_in, N_in, is_annular=False)

    # Pressure gradient calculation
    dPdL_an = 2 * fric_f_an * rho_new * v_an**2 / D_hy
    dPdL_in = 2 * fric_f_in * rho_new * v_in**2 / D_i_new

    return dPdL_an / 22620.40367, dPdL_in / 22620.40367, v_an * 3.28084, v_in * 3.28084 #, tao_new_an / 0.4788 , tao_new_in / 0.4788

def p_drop(rho, mu_p, tao, Q, D_o, D_i, D_w):
    """Corrected pressure drop calculation with proper units"""
    # Constants
    GPM_TO_FT3_PER_SEC = 0.002228
    IN2_TO_FT2 = 1/144

    # Annular calculations
    area_an = (np.pi/4 * (D_w**2 - D_o**2)) * IN2_TO_FT2  # ft² (corrected from 2.448)
    v_an = Q * GPM_TO_FT3_PER_SEC / area_an                # ft/s
    
    # Pipe calculations
    area_pipe = (np.pi/4 * D_i**2) * IN2_TO_FT2           # ft² (corrected from 2.448)
    v_pipe = Q * GPM_TO_FT3_PER_SEC / area_pipe
    
    # Reynolds numbers (corrected factors)
    N_re_an = 928 * rho * v_an * (D_w - D_o)/mu_p         # Annular (corrected from 757)
    N_re_pipe = 928 * rho * v_pipe * D_i/mu_p
    
    # Pressure drops (psi/ft)
    P_f_an = []
    P_f_in = []
    
    for i in range(len(D_o)):
        # Annular (corrected turbulent formula)
        if N_re_an[i] > 2100:
            P_f_an.append(rho**0.8 * v_an[i]**1.8 * mu_p**0.2 / (1413 * (D_w[i]-D_o[i])**1.2))
        else:
            P_f_an.append( (mu_p*v_an[i])/(1000*(D_w[i]-D_o[i])**2) + tao/(200*(D_w[i]-D_o[i])) )

        # Pipe (corrected to use v_pipe)
        if N_re_pipe[i] > 2100:
            P_f_in.append(rho**0.8 * v_pipe[i]**1.8 * mu_p**0.2 / (1815*D_i[i]**1.2))
        else:
            P_f_in.append( (mu_p*v_pipe[i])/(1500*D_i[i]**2) + tao/(225*D_i[i]) )
    
    return P_f_in, P_f_an, v_pipe, v_an

# Old Pressure Loss Calculation (less accurate)
# def p_drop(rho, mu_p, tao, Q, D_o, D_i, D_w):
    
#     # Annular section
#     P_f_an = []
#     Area_an = (np.pi/4 * (D_w**2 - D_o**2)) * 1/144  # Correct annular area (ft²)
#     v_an = Q/Area_an
#     mu_a_an = mu_p + 5*tao*(D_w-D_o)/v_an
#     N_re_an = 928*rho*v_an*(D_w-D_o)/mu_a_an
    
#     for i in range(len(D_o)):
#         if N_re_an[i] > 2100:
#             P_f_an.append((rho**0.75*v_an[i]**1.75*mu_p**0.25)/(1396*(D_w[i]-D_o[i])**1.25)) # Turbulent flow case
#         else:
#             P_f_an.append(((mu_p*v_an[i])/(1000*(D_w[i]-D_o[i])**2) + tao/(200*(D_w[i]-D_o[i])))) # Laminar flow case

#     # Inner section
#     P_f_in = []
#     Area_in = (np.pi/4 * D_o**2) * 1/144  # Correct annular area (ft²)
#     v_in = Q/Area_in
#     mu_a_in = mu_p + 6.66*tao*D_i/v_in
#     N_re_in = 928*rho*v_in*D_i/mu_a_in
#     for i in range(len(D_i)):
#         if N_re_in[i] > 2100:
#             P_f_in.append((rho**0.8*v_in[i]**1.8*mu_p**0.2)/(1815*D_i[i]**1.2)) # Turbulent flow case
#         else:
#             P_f_in.append(((mu_p*v_an[i])/(1500*(D_i[i])**2) + tao/(225*D_i[i]))) # Laminar flow case     
    
#     return P_f_in, P_f_an, v_in, v_an

# def p_gpt(rho, mu_p, tao, Q, D_o, D_i, D_w):
#     # 1. Flow Areas
#     annular_area = np.pi/4 * (D_i**2 - D_o**2)  # 95.6 in²
#     pipe_area = np.pi/4 * D_i**2                        # 7.79 in²

#     # 2. Velocities (ft/s)
#     v_annular = Q / (0.3208 * annular_area) / 60       # 2.56 ft/s (matches your value)
#     v_pipe = Q / (0.3208 * pipe_area) / 60             # 24.7 ft/s (matches your value)

#     # 3. Pressure Losses (Bingham Plastic Model)
#     # Pipe dP/dL (psi/ft)
#     dP_pipe = (tao/(300*D_i)) + (mu_p*v_pipe*60/(1500*D_i**2))
#             # = 10/(300*3.15) + (20*1482)/(1500*3.15²) 
#             # = 0.0106 + 1.99 = 2.0 psi/ft  # Your 5.28 psi/ft suggests code error

#     # Annular dP/dL (psi/ft)
#     dP_annular = (tao/(225*(D_w-D_o))) + (mu_p*v_annular*60/(1000*(D_w-D_o)**2))
#                 # = 10/(225*4.87) + (20*153.8)/(1000*4.87²)
#                 # = 0.0091 + 0.129 = 0.138 psi/ft

# Unit conversion [imperial-metric]
ft2m = lambda ft: ft * 0.3048
in2m = lambda inch: inch * 0.0254
ft_min2ms = lambda f: f * 0.3048 / 60
ft_hr2ms = lambda f: f * 0.3048 / 3600
ppg2kgm = lambda rho: rho * 1.1983e+2
ppg2lbft3 = lambda ppg: ppg*7.4805e+0
psi2Pa = lambda psi: psi * 0.4788
psf2Pa = lambda psf: psf * 47.880208
cp2Pas = lambda viscosity: viscosity/1000
cp2lbfft2 = lambda cp: cp*2.0885e-5
lbfft2Pas = lambda viscosity: 47.88*viscosity  # (lbf/ft^2).s to Pa.s
GPM2ms = lambda flowrate: flowrate * 6.309e-5
lbs2kg = lambda lbs: lbs * 4.536e-1
lbf2N = lambda lbf: lbf * 4.4482e+0
rpm2rad_s = lambda rpm: rpm * 2*np.pi/60

# Unit conversion [metric-imperial]
m2ft = lambda m: m / 0.3048
m2in = lambda m: m / 0.0254
N2lbf = lambda lbf: lbf / 4.44822