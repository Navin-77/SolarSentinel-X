"""SolarSentinel-X V3 physical component geometry with dynamic visual states."""

import math
import plotly.graph_objects as go


def _box_vertices(x0,x1,y0,y1,z0,z1):
    return [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
            [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]


def _box_faces():
    return [(0,1,2),(0,2,3),(4,6,5),(4,7,6),
            (0,4,5),(0,5,1),(1,5,6),(1,6,2),
            (2,6,7),(2,7,3),(4,0,3),(4,3,7)]


def add_box(fig,bounds,color,name,opacity=1.0,hover=None):
    v=_box_vertices(*bounds); f=_box_faces()
    fig.add_trace(go.Mesh3d(
        x=[p[0] for p in v],y=[p[1] for p in v],z=[p[2] for p in v],
        i=[p[0] for p in f],j=[p[1] for p in f],k=[p[2] for p in f],
        color=color,opacity=opacity,flatshading=True,name=name,showlegend=False,
        hovertemplate=(hover or name)+"<extra></extra>"
    ))


def add_cylinder(fig,center,radius,height,color,name,axis="z",segments=20,opacity=1.0):
    cx,cy,cz=center
    theta=[2*math.pi*i/segments for i in range(segments)]
    if axis=="z":
        a=[(cx+radius*math.cos(t),cy+radius*math.sin(t),cz-height/2) for t in theta]
        b=[(cx+radius*math.cos(t),cy+radius*math.sin(t),cz+height/2) for t in theta]
    elif axis=="y":
        a=[(cx+radius*math.cos(t),cy-height/2,cz+radius*math.sin(t)) for t in theta]
        b=[(cx+radius*math.cos(t),cy+height/2,cz+radius*math.sin(t)) for t in theta]
    else:
        a=[(cx-height/2,cy+radius*math.cos(t),cz+radius*math.sin(t)) for t in theta]
        b=[(cx+height/2,cy+radius*math.cos(t),cz+radius*math.sin(t)) for t in theta]
    vertices=a+b; faces=[]
    for i in range(segments):
        j=(i+1)%segments
        faces.extend([(i,j,segments+j),(i,segments+j,segments+i)])
    fig.add_trace(go.Mesh3d(
        x=[p[0] for p in vertices],y=[p[1] for p in vertices],z=[p[2] for p in vertices],
        i=[f[0] for f in faces],j=[f[1] for f in faces],k=[f[2] for f in faces],
        color=color,opacity=opacity,flatshading=True,name=name,showlegend=False,
        hovertemplate=f"<b>{name}</b><extra></extra>"
    ))


def add_pv_panel(fig,origin,width,height,z,power,state_color="#0B3B78"):
    x0,y0=origin; x1,y1=x0+width,y0+height
    add_box(fig,(x0-.14,x1+.14,y0-.14,y1+.14,z-.16,z),
            "#475569","PV aluminium frame")
    cols,rows=6,4; cw,ch=width/cols,height/rows
    for r in range(rows):
        for c in range(cols):
            xa=x0+c*cw+.045; xb=x0+(c+1)*cw-.045
            ya=y0+r*ch+.045; yb=y0+(r+1)*ch-.045
            fig.add_trace(go.Mesh3d(
                x=[xa,xb,xb,xa],y=[ya,ya,yb,yb],z=[z+.018]*4,
                i=[0,0],j=[1,2],k=[2,3],
                color=state_color,opacity=.97,showlegend=False,
                hovertemplate=f"PV cell {r+1}-{c+1}<extra></extra>"
            ))
    # Cell grid
    for c in range(1,cols):
        x=x0+c*cw
        fig.add_trace(go.Scatter3d(x=[x,x],y=[y0,y1],z=[z+.026,z+.026],
            mode="lines",line=dict(color="#CBD5E1",width=1),showlegend=False,hoverinfo="skip"))
    for r in range(1,rows):
        y=y0+r*ch
        fig.add_trace(go.Scatter3d(x=[x0,x1],y=[y,y],z=[z+.026,z+.026],
            mode="lines",line=dict(color="#CBD5E1",width=1),showlegend=False,hoverinfo="skip"))
    add_box(fig,(x0+2.35,x0+3.65,y0+.12,y0+.48,z+.025,z+.16),
            "#1F2937","PV junction box")
    add_cylinder(fig,(x0+3.0,y0+.07,z+.10),.07,.16,"#111827",
                 "PV cable gland",axis="y",segments=12)
    for x in (x0+.35,x1-.35):
        for y in (y0+.35,y1-.35):
            add_box(fig,(x-.08,x+.08,y-.08,y+.08,.05,z-.10),"#64748B","PV support leg")
    for y in (y0+.65,y1-.65):
        add_box(fig,(x0+.25,x1-.25,y-.05,y+.05,.20,z-.10),"#64748B","PV cross rail")


def add_sensor_mount(fig,x,y,z_base,z_top,color="#475569"):
    add_cylinder(fig,(x,y,(z_base+z_top)/2),.035,z_top-z_base,color,
                 "Sensor mounting post",segments=10)
    add_cylinder(fig,(x,y,z_top),.07,.06,color,"Sensor mounting bracket",segments=12)


def add_battery(fig,center,voltage,power,state_color="#22C55E"):
    cx,cy,cz=center
    add_box(fig,(cx-1.25,cx+1.25,cy-.78,cy+.78,cz-.62,cz+.62),"#20252B","Battery casing")
    add_box(fig,(cx-1.08,cx+1.08,cy-.69,cy+.69,cz-.48,cz+.48),"#111827","Battery front")
    add_box(fig,(cx-.70,cx+.70,cy-.715,cy-.68,cz-.28,cz+.28),"#374151","Battery label plate")
    add_box(fig,(cx-.46,cx+.46,cy-.735,cy-.71,cz-.08,cz+.10),"#64748B","Battery printed label")
    add_cylinder(fig,(cx-.60,cy,cz+.72),.14,.18,"#DC2626","Battery positive terminal")
    add_cylinder(fig,(cx+.60,cy,cz+.72),.14,.18,"#111111","Battery negative terminal")
    for i in range(6):
        add_box(fig,(cx-.85+i*.34,cx-.77+i*.34,cy+.56,cy+.66,cz-.24,cz-.08),"#4B5563","Battery vent")
    level=max(0.0,min(1.0,float(voltage)/12.0))
    for i in range(8):
        c=state_color if (i+1)/8<=level else "#334155"
        add_box(fig,(cx-.78+i*.20,cx-.64+i*.20,cy-.70,cy-.66,cz-.45,cz-.34),c,"Battery visual state")


def add_controller(fig,center,status):
    cx,cy,cz=center
    add_box(fig,(cx-1.15,cx+1.15,cy-.22,cy+.22,cz-.75,cz+.75),"#3F4650","MPPT charge controller")
    add_box(fig,(cx-.82,cx+.82,cy-.25,cy-.19,cz-.43,cz+.33),"#0B1220","Controller front")
    add_box(fig,(cx-.50,cx+.50,cy-.28,cy-.24,cz-.17,cz+.22),"#172033","Controller LCD")
    s=str(status).upper()
    led="#22C55E" if s=="NORMAL" else "#F59E0B" if s=="WARNING" else "#EF4444"
    for i in range(3):
        add_cylinder(fig,(cx-.42+i*.42,cy-.31,cz-.39),.055,.07,led if i==0 else "#475569",
                     "Controller LED",axis="y",segments=12)
    for i in range(4):
        add_cylinder(fig,(cx-.63+i*.42,cy+.25,cz-.52),.055,.12,"#111827","Controller terminal",axis="y",segments=12)


def add_esp32(fig,center):
    cx,cy,cz=center
    add_box(fig,(cx-.72,cx+.72,cy-.38,cy+.38,cz-.06,cz+.06),"#166534","ESP32 PCB")
    add_box(fig,(cx-.24,cx+.24,cy-.13,cy+.13,cz+.06,cz+.13),"#111827","ESP32 chip")
    add_box(fig,(cx-.13,cx+.13,cy-.50,cy-.38,cz-.02,cz+.04),"#CBD5E1","ESP32 USB connector")
    for side in (-1,1):
        for i in range(9):
            y=cy-.30+i*.075
            add_cylinder(fig,(cx+side*.77,y,cz+.075),.018,.13,"#D1D5DB","ESP32 pin",segments=8)
    add_box(fig,(cx-.52,cx+.52,cy+.20,cy+.28,cz+.06,cz+.075),"#A3E635","ESP32 antenna trace")


def add_ina219(fig,center,label):
    cx,cy,cz=center
    add_box(fig,(cx-.48,cx+.48,cy-.30,cy+.30,cz-.055,cz+.055),"#15803D",label+" PCB")
    add_box(fig,(cx-.12,cx+.12,cy-.09,cy+.09,cz+.055,cz+.13),"#111827",label+" IC")
    for x in (cx-.28,cx+.28):
        add_cylinder(fig,(x,cy-.34,cz),.065,.12,"#111827",label+" terminal",axis="y",segments=12)
    for x in (cx-.23,cx+.23):
        add_box(fig,(x-.07,x+.07,cy-.07,cy+.07,cz+.055,cz+.10),"#D1D5DB",label+" resistor")


def add_dht22(fig,center):
    cx,cy,cz=center
    add_box(fig,(cx-.30,cx+.30,cy-.20,cy+.20,cz-.34,cz+.34),"#E5E7EB","DHT22 housing")
    for i in range(6):
        z=cz-.23+i*.09
        add_box(fig,(cx-.20,cx+.20,cy-.215,cy-.17,z,z+.035),"#9CA3AF","DHT22 vent")


def add_ds18b20(fig,center):
    cx,cy,cz=center
    add_cylinder(fig,(cx,cy,cz),.13,.46,"#A8A29E","DS18B20 stainless probe",segments=20)
    add_cylinder(fig,(cx,cy,cz+.27),.14,.08,"#78716C","DS18B20 probe tip",segments=20)


def add_ldr(fig,center):
    cx,cy,cz=center
    add_cylinder(fig,(cx,cy,cz),.16,.12,"#111827","LDR body",segments=20)
    add_cylinder(fig,(cx,cy,cz+.08),.11,.035,"#FACC15","LDR photosensitive surface",segments=20)


def add_load(fig,center):
    cx,cy,cz=center
    add_box(fig,(cx-.65,cx+.65,cy-.48,cy+.48,cz-.48,cz+.38),"#4B5563","DC load")
    add_cylinder(fig,(cx,cy,cz+.57),.28,.45,"#D1D5DB","DC load lamp")
