import matplotlib.pyplot as plt
from .Init_xls import N2lbf, m2ft

def vis_plots(Data):

    plt.figure(figsize=(10, 6))
    plt.plot(Data['TIME_ARRAY'], m2ft(Data['v'][-1]), label='Bit Axial Velocity')
    plt.plot(Data['TIME_ARRAY'], m2ft(Data['v'][0]), '--', label='Topdrive Axial Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (ft/s)')
    plt.title('Comparison of surface and downhole axial velocities')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(Data['TIME_ARRAY'], m2ft(Data['z'][-1]), label='Bit Axial Displacement')
    plt.plot(Data['TIME_ARRAY'], m2ft(Data['z'][0]), '--', label='Topdrive Axial Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (ft)')
    plt.title('Comparison of surface and downhole axial displacements')
    plt.legend()
    plt.grid()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot in klbf (original unit)
    ax1.plot(Data['TIME_ARRAY'], N2lbf(Data['Force']), 'b', label='Hookload (lbf)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Surface Load (lbf)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    # Add kN axis
    ax2 = ax1.twinx()
    ax2.plot(Data['TIME_ARRAY'], Data['Force'], 'r', linestyle=':', label='Hookload (N)')
    ax2.set_ylabel('Surface Load (N)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Surface Load (Hookload)')
    plt.grid()
    plt.show()

    # # Export if requested
    # if html_out:
    #     fig.write_html(html_out)
    # if image_out:
    #     fig.write_image(image_out, scale=image_scale)

    # if show:
    #     fig.show()

    return
