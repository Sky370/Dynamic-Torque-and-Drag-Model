from Libs.constants_SI import Calculations
from Libs.Init_xls import *
import numpy as np
import os
import plotly.graph_objects as go

# Data for three-dimensional line and points
clc = Calculations(depth=13500)
x = np.cumsum(clc.del_x)
y = np.cumsum(clc.del_y)
z = np.cumsum(clc.del_z)*(-1)

print("The North displacement is: {}".format(round(x[-1], 2)))
print("The East displacement is: {}".format(round(y[-1], 2)))
print("TVD is: {}".format(round(abs(z[-1]), 2)))


fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', line=dict(color='grey'), marker=dict(size=4, color=z, colorscale='Magma')))
fig.update_layout(scene=dict(
                    xaxis_title='The North displacement',
                    yaxis_title='The East displacement',
                    zaxis_title='TVD'),
                  autosize=True, 
                  margin=dict(r=20, l=10, b=10, t=10))

fig.show()
fig.write_html(os.path.join(outputFolderPath, 'Wellbore_Trajectory.html'))