from Libs.constants_SI import *
from Libs.solver import Main_Func
from scipy.integrate import solve_ivp
from .plots import vis_plots

def simulation(depth, fric_mod):
    clc = Calculations(depth)
    initial_state = np.zeros(4*clc.noe)
    # initial_state[0::4] = clc.initial_displacement
    # initial_state[0::2] = clc.new_us
    
    # # Preallocate arrays (replace dictionary appends)
    # P = {
    #     'DOWNHOLE_WEIGHT': np.zeros_like(clc.runtime),
    #     'DOWNHOLE_TORQUE': np.zeros_like(clc.runtime),
    #     'DOC': np.zeros_like(clc.runtime),
    #     'FRICTION_FORCE': np.zeros((len(clc.runtime), n)),
    #     'FRICTION_TORQUE': np.zeros((len(clc.runtime), n)),
    #     'STATIC_CHECK': np.zeros((len(clc.runtime), n)),
    # }
       # Normal Force Calculation

    P = {}

    P['STATIC_CHECK_PREV'] = np.zeros(clc.noe)
    P['DOWNHOLE_WEIGHT'] = []
    P['SURFACE_WEIGHT'] = []
    P['DOWNHOLE_TORQUE'] = []
    P['SURFACE_TORQUE'] = []
    P['DOC'] = []
    P['SOLUTION_TIME'] = []
    P['THETA_PREV'] = [0.0]
    P['HOLE_DEPTH_PREV'] = [clc.HOLE_DEPTH]
    P['FRICTION_FORCE_STORE'] = []
    P['FRICTION_TORQUE_STORE'] = []
    P['STATIC_CHECK_PREV_STORE'] = []
    P['FRICTION_FORCE_TORSIONAL_STORE'] = []

    sol = solve_ivp(
    lambda t, x: Main_Func(t, x, clc, P, fric_mod),
    [0, clc.time_val],
    initial_state,
    method="BDF",  # Changed to implicit method
    atol=1e-5,     # Tighter tolerances
    rtol=1e-4,
    # t_eval=np.arange(0, 33, 0.005)
    max_step=0.01   # Limit maximum step size
    )

    X_rk45 = sol.y[0::4] - np.reshape(initial_state[0::4], (clc.noe,1)) # Displacement over time
    hook_displ = sol.y[0::4]
    V_rk45 = sol.y[1::4]  # Velocity over time

    # Compute net displacement: dynamic - static
    net_displ_init = hook_displ[0] - hook_displ[1]
    net_displacement = X_rk45[0] - X_rk45[1]

    # Calculate forces for all nodes over time
    print(sum(clc.bw_pipe * clc.global_length_array * np.cos(clc.inc_rad[1:])))
    fff = -clc.ka[0] * net_displ_init \
      + sum(clc.bw_pipe * clc.global_length_array * np.cos(clc.inc_rad[1:])) \
      + clc.TP_w
    
    P['TIME_ARRAY'] = sol.t
    P['SOLUTION_MAIN_ARRAY'] = sol["y"]
    P['z'] = sol.y[0::4]
    P['v'] = sol.y[1::4]
    P['theta'] = sol.y[2::4]
    P['omega'] = sol.y[3::4]
    P['Force'] = fff

    pd.DataFrame(
        {'Time':np.array(sol.t).T, 
         'Displ_surface':np.array(-X_rk45[0]).T,
         'Displ_bit':np.array(-X_rk45[-1]).T,
         'Vel_surface':np.array(-V_rk45[0]).T,
         'Vel_bit':np.array(-V_rk45[-1]).T,
         'Force_surface':np.array(fff).T
        }
    ).to_excel('BDF_data.xlsx', index=False)

    vis_plots(P)
    return None