import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import os
from .Init_xls import Nm2lbfft, N2lbf, m2ft, rad_s2rpm

def vis_plots(Data, save_path="drilling_dashboard.html"):
    time_array = np.array(Data['TIME_ARRAY'])
    time_array_2 = np.array(Data['SOLUTION_TIME'])

    # Prepare signals
    WEIGHT_S_lbf = N2lbf(np.array(Data['SURFACE_WEIGHT']))
    WEIGHT_B_lbf = N2lbf(np.array(Data['DOWNHOLE_WEIGHT']))
    TORQUE_S = Nm2lbfft(np.array(Data['SURFACE_TORQUE']))
    TORQUE_B = Nm2lbfft(np.array(Data['DOWNHOLE_TORQUE']))

    # Create a unified interactive dashboard
    fig = sp.make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=(
            "Axial Velocities (Topdrive vs. Bit)",
            "Torsional Velocities (Topdrive vs. Bit)",
            "Hookload Comparison (Surface vs. Bit)",
            "Torque Comparison (Topdrive vs. Bit)"
        )
    )

    # Row 1 - Axial Velocity
    fig.add_trace(go.Scatter(
        x=time_array, y=m2ft(Data['v'][0]), mode='lines',
        name='Topdrive Axial Velocity', line=dict(dash='dash', color='blue')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=time_array, y=m2ft(Data['v'][-1]), mode='lines',
        name='Bit Axial Velocity', line=dict(color='firebrick')
    ), row=1, col=1)
    fig.update_yaxes(title_text="Velocity (ft/s)", row=1, col=1)

    # Row 2 - Torsional Velocity
    fig.add_trace(go.Scatter(
        x=time_array, y=rad_s2rpm(Data['omega'][0]), mode='lines',
        name='Topdrive RPM', line=dict(dash='dash', color='teal')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=time_array, y=rad_s2rpm(Data['omega'][-1]), mode='lines',
        name='Bit RPM', line=dict(color='mediumpurple')
    ), row=2, col=1)
    fig.update_yaxes(title_text="RPM", row=2, col=1)

    # Row 3 - Hookload
    fig.add_trace(go.Scatter(
        x=time_array_2, y=WEIGHT_S_lbf,
        name="Hookload (lbf)", line=dict(color='royalblue')
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=time_array_2, y=WEIGHT_B_lbf,
        name="WOB (lbf)", line=dict(color='darkorange')
    ), row=3, col=1)
    fig.update_yaxes(title_text="Load (lbf)", row=3, col=1)

    # Row 4 - Torque
    fig.add_trace(go.Scatter(
        x=time_array_2, y=TORQUE_S,
        name="Topdrive Torque (lbf-ft)", line=dict(color='crimson')
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=time_array_2, y=TORQUE_B,
        name="Bit Torque (lbf-ft)", line=dict(color='seagreen')
    ), row=4, col=1)
    fig.update_yaxes(title_text="Torque (lbf-ft)", row=4, col=1)

    # Shared X-axis
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)

    # Layout Styling
    fig.update_layout(
        title_text="\U0001F4C8 Drilling Dynamics Dashboard",
        height=1000,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
        template="plotly_white"
    )

    fig.show()

    # Export to HTML file
    if save_path:
        fig.write_html(save_path)
        print(f"Dashboard saved to: {os.path.abspath(save_path)}")
    return
