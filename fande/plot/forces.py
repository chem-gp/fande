
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"
except ImportError:
    print("plotly not installed...")


import numpy as np

try:
    import wandb
except ImportError:
    print("wandb not installed...")




def atom_forces_3d(mol_traj, forces_np, atoms_ind, show=False):

    xmin = forces_np[:, :, 0].min()
    ymin = forces_np[:, :, 0].min()
    zmin = forces_np[:, :, 0].min()

    xmax = forces_np[:, :, 0].max()
    ymax = forces_np[:, :, 0].max()
    zmax = forces_np[:, :, 0].max()

    data = []
    for ai in atoms_ind:
        x = forces_np[:, ai, 0]
        y = forces_np[:, ai, 1]
        z = forces_np[:, ai, 2]

        data.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=1, line=dict(width=1)),
                name="atom " + str(ai) + " : " + mol_traj[0][ai].symbol,
                visible="legendonly",
            )
        )

    fig = go.Figure(data=data)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=8,
                range=[xmin, xmax],
                showgrid=True,
                gridwidth=1,
                gridcolor="LightPink",
            ),
            yaxis=dict(
                nticks=8,
                range=[ymin, ymax],
                showgrid=True,
                gridwidth=1,
                gridcolor="LightPink",
            ),
            zaxis=dict(
                nticks=8,
                range=[zmin, zmax],
                showgrid=True,
                gridwidth=1,
                gridcolor="LightPink",
            ),
        ),
        width=1000,
        margin=dict(r=0, l=0, b=0, t=15),
    )

    wandb.log({"atom forces 3d": fig})

    if show:
        fig.show()

    return
