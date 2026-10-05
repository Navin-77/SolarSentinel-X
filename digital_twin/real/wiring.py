"""Clean physical-looking wiring paths for V3."""

import plotly.graph_objects as go


def add_wire(fig,points,color="#111827",width=5,name="Wire",dash=None):
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in points],y=[p[1] for p in points],z=[p[2] for p in points],
        mode="lines",line=dict(color=color,width=width,dash=dash),
        showlegend=False,hovertemplate=f"<b>{name}</b><extra></extra>"
    ))


def add_cable(fig,start,end,color="#111827",name="Cable",width=6):
    mid_z=max(start[2],end[2])+.12
    p1=(start[0],start[1],mid_z); p2=(end[0],start[1],mid_z)
    add_wire(fig,[start,p1,p2,end],color,width,name)


def add_signal(fig,start,end,color="#64748B",name="Signal cable"):
    mid=((start[0]+end[0])/2,(start[1]+end[1])/2,max(start[2],end[2])+.10)
    add_wire(fig,[start,mid,end],color,3,name)
