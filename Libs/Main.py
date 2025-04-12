from Libs.constants_SI import *
from Libs.solver import Main_Func
# from Libs.TD_velocity import topdrive_new
# from Libs.visualize import visualize_vel
# from Libs.Init_xls import outputFolderPath
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def simulation(depth, fric_mod):
    clc = Calculations(depth)
    initial_state = np.zeros(4*clc.noe)
    initial_state[0::4] = clc.initial_displacement
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
    P['DOWNHOLE_TORQUE'] = []
    P['DOC'] = []
    P['SOLUTION_TIME'] = []
    P['THETA_PREV'] = [0.0]
    P['HOLE_DEPTH_PREV'] = [clc.HOLE_DEPTH]
    P['FRICTION_FORCE_STORE'] = []
    P['FRICTION_TORQUE_STORE'] = []
    P['STATIC_CHECK_PREV_STORE'] = []
    P['new_force'] = []

    sol = solve_ivp(
    lambda t, x: Main_Func(t, x, clc, P, fric_mod),
    [0, clc.time_val],
    initial_state,
    method="BDF",  # Changed to implicit method
    atol=1e-5,     # Tighter tolerances
    rtol=1e-4,
    # t_eval=np.arange(0, 33, 0.005)
    max_step=0.005   # Limit maximum step size
    )

    time_arr = sol.t
    sol_arr = sol["y"]
    P['TIME_ARRAY'] = time_arr
    P['SOLUTION_MAIN_ARRAY'] = sol_arr

    X_rk45 = sol.y[0::4] - np.reshape(initial_state[0::4], (clc.noe,1)) # Displacement over time
    hook_displ = sol.y[0::4]
    V_rk45 = sol.y[1::4]  # Velocity over time

    # Compute net displacement: dynamic - static
    # _, _, z_topdr, _ = topdrive_new(time_arr, clc)
    net_displ_init = hook_displ[0] - hook_displ[1]
    net_displacement = X_rk45[0] - X_rk45[1]

    # Calculate forces for all nodes over time
    print(sum(clc.bw_pipe * clc.global_length_array * np.cos(clc.inc_rad[1:])))
    fff = -clc.ka[0] * net_displ_init #+ sum(clc.bw_pipe * clc.global_length_array * np.cos(clc.inc_rad))      
      # + np.cumsum(np.array(P['new_force']))[-1]
      #  - clc.global_ca_matrix[0, 0] * V_rk45[0]          

    pd.DataFrame(
        {'Time':np.array(sol.t).T, 
         'Displ_surface':np.array(-X_rk45[0]).T,
         'Displ_bit':np.array(-X_rk45[-1]).T,
         'Vel_surface':np.array(-V_rk45[0]).T,
         'Vel_bit':np.array(-V_rk45[-1]).T,
         'Force_surface':np.array(fff).T
        }
    ).to_excel('BDF_data.xlsx', index=False)

    # pd.DataFrame([sol.t, fff]).to_csv('rasimchik.csv', index=False)
    plt.figure()
    plt.plot(sol.t, net_displ_init, label="u1-u2")
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (ft)')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, -V_rk45[-1]/0.3048, label='Bit Axial Velocity')
    plt.plot(sol.t, -V_rk45[0]/0.3048, '--', label='Topdrive Axial Velocity')
    # plt.plot(sol_rk45.t, sol_rk45.y[0, :], label='RK45 (Explicit)')
    # plt.plot(sol_bdf.t, sol_bdf.y[0, :], '--', label='BDF (Implicit)')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (ft/s)')
    plt.title('Comparison of surface and downhole axial velocities')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, -X_rk45[-1], label='Bit Axial Displacement')
    plt.plot(sol.t, -X_rk45[0], '--', label='Topdrive Axial Displacement')
    # plt.plot(sol_rk45.t, sol_rk45.y[0, :], label='RK45 (Explicit)')
    # plt.plot(sol_bdf.t, sol_bdf.y[0, :], '--', label='BDF (Implicit)')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (ft)')
    plt.title('Comparison of surface and downhole axial displacements')
    plt.legend()
    plt.grid()
    plt.show()

    # For RK45 results
    # ForceS_rk45 = clc.global_ka_matrix @ np.diff(X_rk45)
    # ForceS_hook = ForceS_rk45[0]  # First node force

    # For BDF results
    # ForceS_bdf = clc.global_ka_matrix @ sol_bdf.y[:N, :]
    # ForceS_bdf = ForceS_bdf[-1, :]

    # Unit conversions
    N2lbf = lambda N: N / 4.44822

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot in klbf (original unit)
    ax1.plot(sol.t, -clc.ka[0] * net_displ_init/4.44822, 'b', label='Hookload (lbf)')
    # ax1.plot(sol_bdf.t, -ForceS_bdf / 1000, 'g--', label='BDF (klbf)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Surface Load (lbf)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    # Add kN axis
    ax2 = ax1.twinx()
    ax2.plot(sol.t, -clc.ka[0] * net_displ_init*4.44822, 'r', linestyle=':', label='Hookload (N)')
    # ax2.plot(sol_bdf.t, lbf2N(-ForceS_bdf / 1000), 'm--', label='BDF (kN)')
    ax2.set_ylabel('Surface Load (N)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Surface Load (Hookload)')
    plt.grid()
    plt.show()

    # fig, ax1 = plt.subplots(figsize=(10, 6))

    # # Plot in klbf (original unit)
    # ax1.plot(sol.t, -clc.ka[0] * net_displacement, 'b', label='Hookload (lbf)')
    # # ax1.plot(sol_bdf.t, -ForceS_bdf / 1000, 'g--', label='BDF (klbf)')
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Surface Load (lbf)', color='k')
    # ax1.tick_params(axis='y', labelcolor='k')

    # # Add kN axis
    # ax2 = ax1.twinx()
    # ax2.plot(sol.t, -clc.ka[0] * net_displacement*4.44822, 'r', linestyle=':', label='Hookload (N)')
    # # ax2.plot(sol_bdf.t, lbf2N(-ForceS_bdf / 1000), 'm--', label='BDF (kN)')
    # ax2.set_ylabel('Surface Load (N)', color='k')
    # ax2.tick_params(axis='y', labelcolor='k')

    # # Combine legends
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # plt.title('Surface Load (Hookload)')
    # plt.grid()
    # plt.show()

    # visualize_vel(time_arr, sol_arr, clc.noe, P, clc, outputFolderPath, fric_mod)
    return None