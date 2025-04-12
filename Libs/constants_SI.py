from .Init_xls import *
from .TD_velocity import topdrive_interpolator
import scipy.sparse as sps
from scipy.optimize import newton
# from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')

class Calculations:
    def __init__(self, depth):
        # Field units to SI units
        self.g = ft2m(32.1740)  # ft/s² to m/s²
        self.bit_depth = ft2m(depth) # ft to m
        self.elem_length = ft2m(float(df_ADV[df_ADV["Parameter"] == "Element Length"]["Value"]))        # ft to m
        self.mud_density_ppg = ppg2kgm(float(df_WELL[df_WELL["Parameter"] == "Mud Density"]["Value"]))  # ppg to kg/m3
        self.pipe_density = ppg2kgm(float(df_WELL[df_WELL["Parameter"] == "Steel Density"]["Value"]))   # ppg to kg/m3
        self.visc_p = cp2Pas(float(df_WELL[df_WELL["Parameter"] == "Plastic Viscosity"]["Value"]))   # (lbf/ft^2).s to Pa.s 
        self.tao_y = psi2Pa(float(df_WELL[df_WELL["Parameter"] == "Yield Point"]["Value"]))             # lbf/100ft2 to Pa
        self.bf = float((1-self.mud_density_ppg/self.pipe_density))
        self.mu_s = float(df_WELL[df_WELL["Parameter"] == "Static Friction Factor"]["Value"])           # Static Fric. 
        self.mu_d = float(df_WELL[df_WELL["Parameter"] == "Dynamic Friction Factor"]["Value"])          # Dynamic Fric.
        self.E = psf2Pa(float(df_ADV[df_ADV["Parameter"] == "Young Modulus"]["Value"]))                 # Young Modulus, lbf/ft2 to Pa
        self.G = psf2Pa(float(df_ADV[df_ADV["Parameter"] == "Shear Modulus"]["Value"]))                 # Shear Modulus, lbf/ft2 to Pa
        self.ccs = float(df_ADV[df_ADV["Parameter"] == "CCS"]["Value"])                                 # ksi
        self.v_cs = float(df_WELL[df_WELL["Parameter"] == "Stribeck Critical Velocity"]["Value"])       # m/s
        # self.CT_BOREHOLE = float(df_WELL[df_WELL["Parameter"] == "Torsional Drag Coefficient"]["Value"])    # N sec/m
        self.Q = float(df_PUMP[df_PUMP["Parameter"] == "Flow Rate"]["Value"])  # GPM

        # Time intervals definition
        self.a1 = float(df_TOP[df_TOP["Parameter"] == "a1"]["Value"])
        self.a2 = float(df_TOP[df_TOP["Parameter"] == "a2"]["Value"])
        self.a3 = float(df_TOP[df_TOP["Parameter"] == "a3"]["Value"])
        self.a4 = float(df_TOP[df_TOP["Parameter"] == "a4"]["Value"])
        self.a5 = float(df_TOP[df_TOP["Parameter"] == "a5"]["Value"])
        self.a6 = float(df_TOP[df_TOP["Parameter"] == "a6"]["Value"])
        self.b1 = float(df_TOP[df_TOP["Parameter"] == "b1"]["Value"])
        self.b2 = float(df_TOP[df_TOP["Parameter"] == "b2"]["Value"])
        self.b3 = float(df_TOP[df_TOP["Parameter"] == "b3"]["Value"])
        self.b4 = float(df_TOP[df_TOP["Parameter"] == "b4"]["Value"])
        self.b5 = float(df_TOP[df_TOP["Parameter"] == "b5"]["Value"])
        self.b6 = float(df_TOP[df_TOP["Parameter"] == "b6"]["Value"])
        
        # Velocity definition
        self.v1 = ft_min2ms(float(df_TOP[df_TOP["Parameter"] == "Top Drive Axial Velocity Magnitude 1 (ft/min)"]["Value"]))     # ft/min to m/s
        self.v2 = ft_min2ms(float(df_TOP[df_TOP["Parameter"] == "Top Drive Axial Velocity Magnitude 2 (ft/min)"]["Value"]))     # ft/min to m/s
        self.rpm1 = rpm2rad_s(float(df_TOP[df_TOP["Parameter"] == "Top Drive RPM Magnitude 1 (RPM)"]["Value"]))                 # rev/min to rad/s
        self.rpm2 = rpm2rad_s(float(df_TOP[df_TOP["Parameter"] == "Top Drive RPM Magnitude 2 (RPM)"]["Value"]))                 # rev/min to rad/s

        # Time array
        self.time_val = float(df_ADV[df_ADV["Parameter"] == "Run Time"]["Value"])   # seconds

        # Extra
        # --------- Load steady state inputs -------- #
        WOB_SS = float(df_SS[df_SS["Parameter"] == "WOB initial"]["Value"])                 # lbs
        ROP_SS = float(df_SS[df_SS["Parameter"] == "ROP steady state"]["Value"]) / 3600     # m/hr to m/s
        RPM_SS = float(df_SS[df_SS["Parameter"] == "RPM steady state"]["Value"]) / 60       # rev/min to rev/s  

        # Drill pipes:
        self.dp_length = self.bit_depth - ft2m(df_BHA["Total Length (ft)"].iloc[0])
        self.N_DP, self.L_DP = nearestLength(self.dp_length, self.elem_length)
        self.N_DP = round(self.N_DP)        
        self.DP_OD = np.ones(self.N_DP) * df_BHA["OD (in)"].iloc[0]  
        self.DP_ID = np.ones(self.N_DP) * df_BHA["ID (in)"].iloc[0]  
        self.DP_MASS_ARRAY = np.ones(self.N_DP) * df_BHA["Mass (lbs)"].iloc[0]  
        self.L_DP_ARRAY = np.ones(self.N_DP) * self.L_DP
        self.DP_TYPES = np.array(["DP"]*self.N_DP)
        self.TJ_OD = df_BHA["OD Tool Joint (in)"].iloc[0]   # Tool Joint OD, in

        # BHA:
        self.BHA_TYPES = np.array(np.repeat(df_BHA["BHA Type"], df_BHA["Number of Items"]))
        self.BHA_LEN = np.array(np.repeat(df_BHA["Length (ft)"]/df_BHA["Number of Items"] , df_BHA["Number of Items"]))
        self.BHA_MASS = np.array(np.repeat(df_BHA["Mass (lbs)"]/df_BHA["Number of Items"] , df_BHA["Number of Items"]))
        self.BHA_OD = np.array(np.repeat(df_BHA["OD (in)"], df_BHA["Number of Items"]))
        self.BHA_ID = np.array(np.repeat(df_BHA["ID (in)"], df_BHA["Number of Items"])) 

        # Bottom hole
        self.HOLE_OD = in2m(df_ADV[df_ADV["Parameter"] == "Hole Diameter"]["Value"].iloc[0])    # inch to m
        self.HOLE_DEPTH = ft2m(df_ADV[df_ADV["Parameter"] == "Hole Depth"]["Value"].iloc[0])    # ft to m
        self.noe = self.N_DP + len(self.BHA_OD)
        self.HOLE_ARRAY = np.ones(self.noe)*self.HOLE_OD                                        # inch to m
        self.HOLE_LENGTH = self.HOLE_DEPTH - self.bit_depth
        self.N_HOLE, self.L_HOLE = nearestLength(self.HOLE_LENGTH, self.elem_length)
        self.N_HOLE = round(self.N_HOLE)
        
        if self.HOLE_LENGTH > 0:
            self.OP_HOLE_ARRAY = np.ones(self.N_HOLE) * self.HOLE_OD
            self.global_hole_array = np.concatenate([self.HOLE_ARRAY, self.OP_HOLE_ARRAY])
        elif self.HOLE_DEPTH < self.bit_depth:
            raise ValueError("Hole depth cannot be smaller than bit depth.")
        else:
            self.global_hole_array = self.HOLE_ARRAY
        
        # Concatenation
        self.global_mass_array = lbs2kg(np.concatenate([self.DP_MASS_ARRAY, self.BHA_MASS]))     # Final conversion from lbs to kg
        self.global_length_array = np.concatenate([self.L_DP_ARRAY, ft2m(self.BHA_LEN)])         # ft to m
        self.global_od_array = in2m(np.concatenate([self.DP_OD, self.BHA_OD]))                   # inch to m
        self.global_id_array = in2m(np.concatenate([self.DP_ID, self.BHA_ID]))                   # inch to m
        self.global_types = np.concatenate([self.DP_TYPES, self.BHA_TYPES])
        self.global_eps = (self.HOLE_ARRAY/self.global_od_array) - 1
        
        # Area, Volume and Density calculation
        self.D_h = self.global_hole_array[:self.noe] - self.global_od_array             # Hydraulic diameter
        self.A_i = np.pi/4*(self.global_id_array)**2                                    # Inner area of the pipe
        self.A_o = np.pi/4*(self.global_od_array)**2                                    # Outer area of the pipe
        self.A_cross = np.pi/4*(self.global_od_array**2-self.global_id_array**2)        # Cross-sectional area of the pipe
        self.A_h = np.pi/4*(self.HOLE_ARRAY**2-self.global_od_array**2)                 # Annular flow area between the wellbore and the pipe
        self.Vol = self.A_cross*self.global_length_array                                # Volume of the pipe
        self.rho = self.global_mass_array/self.Vol                                      # Density of the pipe/bha

        # Prelim calculations:
        self.J_polar = np.pi / 32 * (self.global_od_array**4 - self.global_id_array** 4)                        # Polar moment of Inertia
        self.J_m = self.rho * self.J_polar * self.global_length_array                                            # Mass moment of Inertia
        self.ka = self.E * self.A_cross / self.global_length_array                                              # Axial stiffness 
        self.kt = self.G * self.J_polar / self.global_length_array                                              # Torsional stiffness

        # # Parallel stifness technique
        # hwdp_mask = self.global_types == "HWDP"            # Mask for HWDP components
        # collar_mask = self.global_types == "Collar"        # Mask for Collar components

        # if np.any(hwdp_mask):
        #     hwdp_ka = 1 / np.sum(1 / self.ka[hwdp_mask])        # Combined axial stiffness
        #     hwdp_kt = 1 / np.sum(1 / self.kt[hwdp_mask])        # Combined torsional stiffness
        #     self.ka[hwdp_mask] = hwdp_ka * len(self.BHA_TYPES[self.BHA_TYPES == "HWDP"]) 
        #     self.kt[hwdp_mask] = hwdp_kt * len(self.BHA_TYPES[self.BHA_TYPES == "HWDP"]) 

        # if np.any(collar_mask):
        #     collar_ka = 1 / np.sum(1 / self.ka[collar_mask])    # Combined axial stiffness
        #     collar_kt = 1 / np.sum(1 / self.kt[collar_mask])    # Combined torsional stiffness
        #     self.ka[collar_mask] = collar_ka * len(self.BHA_TYPES[self.BHA_TYPES == "Collar"]) 
        #     self.kt[collar_mask] = collar_kt * len(self.BHA_TYPES[self.BHA_TYPES == "Collar"]) 

        # Build and Turn rates calculation
        self.bw_pipe = self.bf*self.global_mass_array/self.global_length_array * self.g
        self.MD = np.insert(np.cumsum(self.global_length_array), 0, 0)
        # self.MD = np.cumsum(self.global_length_array)
        self.inc, self.azi, self.K = survey_mod_SI(df_SRV, self.MD, self.bf, self.global_mass_array, self.g)
        self.inc_rad = np.round(np.deg2rad(self.inc), 9)
        self.azi_rad = np.round(np.deg2rad(self.azi), 9)

        # Minimum Curvature Method optimized further with np.diff
        cos_delta_azi = np.cos(np.diff(self.azi_rad))
        self.md_diff = np.diff(self.MD)

        self.betta = np.arccos(np.sin(self.inc_rad[:-1])*np.sin(self.inc_rad[1:])*cos_delta_azi \
                               + np.cos(self.inc_rad[:-1])*np.cos(self.inc_rad[1:]))                    # radians
        self.betta = np.nan_to_num(self.betta, nan=0.0)
        self.RF = np.where(self.betta == 0, self.md_diff/2, self.md_diff/self.betta*np.tan(self.betta/2))
        self.DLS = self.betta/self.md_diff

        # Displacement calculations optimized with numpy
        self.del_x = self.RF*(np.sin(self.inc_rad[:-1])*np.cos(self.azi_rad[:-1]) + np.sin(self.inc_rad[1:])*np.cos(self.azi_rad[1:]))
        self.del_y = self.RF*(np.sin(self.inc_rad[:-1])*np.sin(self.azi_rad[:-1]) + np.sin(self.inc_rad[1:])*np.sin(self.azi_rad[1:]))
        self.del_z = self.RF*(np.cos(self.inc_rad[:-1]) + np.cos(self.inc_rad[1:]))
        self.MD_s = (self.MD[:-1] + self.MD[1:])/2

        # B, T, and inc_ave calculations
        self.B = np.diff(self.inc_rad)/self.md_diff
        self.T = np.diff(self.azi_rad)/self.md_diff
        self.inc_ave = (self.inc_rad[:-1] + self.inc_rad[1:]) / 2
        # self.inc_ave = np.insert(self.inc_ave, -1, self.inc_ave[-1])    # Temp Fix

        # Temp. Fix
        self.B_new = np.insert(self.B, 0, 0)
        self.T_new = np.insert(self.T, 0, 0)
        self.DLS_new = np.insert(self.DLS, 0, 0)

        # Tension, Normal force, and Side force calculations optimized
        # self.t_z = np.cos(self.inc_rad)
        # # self.n_z = -1 / np.degrees(self.DLS_new) * np.sin(self.inc_rad) * self.B_new
        # # self.b_z = 1 / np.degrees(self.DLS_new) * np.sin(self.inc_rad)**2 * self.T_new

        # self.n_z = -1 / self.K * np.sin(self.inc_rad) * self.B_new
        # self.b_z = 1 / self.K * np.sin(self.inc_rad)**2 * self.T_new
        
        self.t_z = abs(np.cos(self.inc_rad))
        self.n_z = -1 / self.DLS * np.sin(self.inc_rad[1:]) * self.B
        self.b_z = 1 / self.DLS * np.sin(self.inc_rad[1:])**2 * self.T

        self.tz = self.t_z[1:]
        self.nz = np.nan_to_num(self.n_z)
        self.bz = np.nan_to_num(self.b_z)
        
        # # Tension, Normal force, and Side force calculations optimized
        
        # self.t_z_min_val = np.cos(self.inc_rad[:-1])*np.cos(self.DLS*(self.MD_s - self.MD[:-1])) + \
        #     ((np.cos(self.inc_rad[1:]) - np.cos(self.inc_rad[:-1])*np.cos(self.betta))/np.sin(self.betta))*np.sin(self.DLS*(self.MD_s - self.MD[:-1]))
        # self.t_z = np.where(np.isnan(self.t_z_min_val), np.cos((self.inc_rad[1:] + self.inc_rad[:-1]) / 2), self.t_z_min_val)
        # self.n_z = -np.cos(self.inc_rad[:-1])*np.sin(self.DLS*(self.MD_s - self.MD[:-1])) + \
        #     ((np.cos(self.inc_rad[1:]) - np.cos(self.inc_rad[:-1])*np.cos(self.betta))/np.sin(self.betta))*np.cos(self.DLS*(self.MD_s - self.MD[:-1]))
        # self.n_z = np.nan_to_num(self.n_z)
        # self.b_z = np.sin(self.inc_rad[:-1])*np.sin(self.inc_rad[1:])*np.sin(self.azi_rad[1:] - self.azi_rad[:-1])/np.sin(self.betta)
        # self.b_z = np.nan_to_num(self.b_z)

        self.DIA_EQ = (27*self.global_od_array + 3*in2m(self.TJ_OD)) / 30     # in to m
        AXIAL_VEL_MULTIPLIER = self.global_od_array**2 / (self.HOLE_OD**2 - self.global_od_array**2)    # Accounting for mud velocity drag effects along axial direction
        DOC_SS = ROP_SS / RPM_SS   # m/rev
        # units of CCS of formation in ksi
        # units of k_CCS are in '1/ksi'
        MU_ROCK = -0.349 * np.log(self.ccs) + 2.0436
        # mu_rock = -0.0201 * CCS + 1.5333
        # coefficient of friction for different rock-strength
        self.K_WOB = 0.8 * (self.ccs*0.5) * lbf2N(WOB_SS) / DOC_SS * (m2in(self.HOLE_OD) / 12.25)    # units of k_WOB are in (N-rev)/(m)
        self.K_TQ = MU_ROCK / 3 * (self.K_WOB / 0.8) * self.HOLE_OD                            # units of k_TQ are in (N-rev)

        # self.CA_BOREHOLE = self.CT_BOREHOLE * (
        # (ROP_SS * AXIAL_VEL_MULTIPLIER) / (RPM_SS) / (self.DIA_EQ * np.pi)
        # )
        # self.global_ct_array = np.where(self.global_length_array == 0, 0, self.CT_BOREHOLE / self.global_length_array)
        # self.global_ca_array = np.where(self.global_length_array == 0, 0, self.CA_BOREHOLE / self.global_length_array)
        
        # Sparse Matrices       
        self.global_ka_matrix = sps.diags(
            [-self.ka[1:], self.ka + np.append(self.ka[1:], 0), -self.ka[1:]],
            offsets=[-1, 0, 1],
            format='csr'
        )
        self.global_kt_matrix = sps.diags(
            [-self.kt[1:], self.kt + np.append(self.kt[1:], 0), -self.kt[1:]],
            offsets=[-1, 0, 1],
            format='csr'
        )

        # ratio_p = 0.03
        ratio_p = 0.05
        # self.K_inv = sps.diags(1 / self.global_ka_matrix.diags(), format='csr')
        self.alpha_ax = 2 * ratio_p /np.sqrt(self.ka/ self.global_mass_array)                                       # Natural frequency for axial motion
        self.alpha_tor = 2 * ratio_p /np.sqrt(self.kt / self.global_mass_array)                                      # Natural frequency for torsional motion
        self.alpha_ax = 0.15
        # # Calculate for first and third modes
        self.global_ca_rayleigh = self.global_ka_matrix.dot(self.alpha_ax)  # + betta*sps.diags(self.global_mass_array)  # Now includes mass-proportional term
        self.ca_visc = self.visc_p* 2*np.pi*self.global_length_array*(self.global_od_array/self.D_h)
        self.global_ca_visc = sps.diags(self.ca_visc, format='csr')
        self.global_ca_visc = 0
        self.global_ca_matrix = self.global_ca_rayleigh     # Rayleigh axial viscous damping coefficient (assuming Betta is zero)
        self.global_ct_matrix = self.global_kt_matrix.dot(self.alpha_ax)      # Rayleigh torsional viscous damping coefficient (assuming Betta is zero)
        self.global_mass_inv_matrix = sps.diags(1 / self.global_mass_array, format='csr')
        self.global_inertia_inv_matrix = sps.diags(1 / self.J_m, format='csr')
        self.global_mass_inv_ka_matrix = self.global_mass_inv_matrix @ self.global_ka_matrix        # Sparse Matrix
        self.global_inertia_inv_kt_matrix = self.global_inertia_inv_matrix @ self.global_kt_matrix  # Sparse Matrix
        self.global_mass_inv_ca_matrix = self.global_mass_inv_matrix @ self.global_ca_matrix        # Sparse Matrix
        self.global_inertia_inv_ct_matrix = self.global_inertia_inv_matrix @ self.global_ct_matrix  # Sparse Matrix 

        # Extra
        # self.Normal_force = lambda z: np.sqrt(
        #     (self.global_ka_matrix.dot(z) * self.DLS_new)**2 + 
        #     (self.bf * self.global_mass_array * self.g * np.sin(self.inc_rad))**2
        # )
        # self.nz = -self.B_new/np.where(self.DLS_new==0, 1e-10, self.DLS_new) * np.sin(self.inc_rad)
        # self.bz = self.T_new/np.where(self.DLS_new==0, 1e-10, self.DLS_new) * np.sin(self.inc_rad)**2
        
        self.Normal_force = lambda z: np.sqrt(
            (self.global_ka_matrix.dot(z) * self.DLS + 
             self.bw_pipe * self.nz)**2 + 
            (self.bw_pipe * self.bz)**2
        )*self.global_length_array

        self._compute_initial_static_forces()
        self._compute_viscous_forces()
        self.topdrive_interpolators = topdrive_interpolator(self)
        
    def _compute_initial_static_forces(self):
        """Newton-Raphson based quasi-static force initialization"""
        self.f = np.zeros(self.noe)
        w = self.bw_pipe
        # self.f[-1] = w[-1]*self.global_length_array[-1]
        
        for n in range(self.noe-2, 0, -1):  # From second last node upwards
            # Get precomputed values for this node
            
            def func(x):
                if self.DLS[n] < 1e-10:
                    return -x + self.f[n+1] + w[n] * self.tz[n] * self.global_length_array[n]
                return -x + self.f[n+1] + self.global_length_array[n] \
                    * (w[n] * self.tz[n] - self.mu_d * np.sqrt((x * self.DLS[n] + w[n] * self.nz[n])**2 + (w[n] * self.bz[n])**2))
            
            x = newton(func, self.f[n+1], maxiter=100)
            self.f[n] = x if x > 0 else 0.00001
        
        # Handle first node explicitly
        self.f[0] = self.f[1] + w[0]*self.global_length_array[0]
        
        # Compute initial displacements using stiffness matrix
        new_u = np.cumsum(self.f[:-1]/self.ka[:-1])
        self.new_us = np.cumsum(self.f/self.ka)
        self.initial_displacement = np.insert(new_u, 0, 0)

    def _compute_viscous_forces(self):
        dP_dL_Shear_o , dP_dL_Shear_i, v_o, v_i = pres_calc(
        Q=self.Q,
        rho=ppg2lbft3(self.mud_density_ppg), 
        m = 1,
        K=self.visc_p, 
        tao=self.tao_y,
        D_o=self.global_od_array*12, 
        D_i=self.global_id_array*12,
        D_w=self.HOLE_ARRAY*12,
        )
        # dP_dL_Shear_i , dP_dL_Shear_o, v_i, v_o = p_drop(
        # Q=self.Q,
        # rho=self.mud_density_ppg, 
        # mu_p=self.visc_p,
        # tao=self.tao_y,
        # D_o=self.global_od_array*12, 
        # D_i=self.global_id_array*12,
        # D_w=self.HOLE_ARRAY*12,
        # )
        dP_dL_o = ppg2lbft3(self.mud_density_ppg)*self.g*np.cos(self.inc_rad[1:]) + dP_dL_Shear_o
        dP_dL_i = ppg2lbft3(self.mud_density_ppg)*self.g*np.cos(self.inc_rad[1:]) - dP_dL_Shear_i
        # dP_dL_i[:-4] = 0
        self.F_visc = self.global_length_array*(-dP_dL_o*self.A_h + dP_dL_i*self.A_i)
        self.F_visc1 = self.global_length_array*(dP_dL_Shear_o*self.A_h - dP_dL_Shear_i*self.A_i)
        return self.F_visc1