"""
SolarSentinel-X — Dynamic Physical 3D Digital Twin V5 FINAL

IMPORTANT
---------
This file controls ONLY the 3D Digital Twin visualization.

It keeps:
    - Real ESP32 sensor values
    - Dataset-driven LOW / NORMAL / HIGH states
    - Real RF / LSTM / Autoencoder status
    - Physical PV panel
    - Physical sensors
    - Battery
    - MPPT controller
    - ESP32
    - INA219
    - DC load
    - Physical wiring

The information boards are now:
    - physically oriented toward the camera
    - black/dark blue
    - live-value driven
    - text placed on the FRONT face
"""


import math
import plotly.graph_objects as go


# ==============================================================
# PROJECT COMPONENTS
# ==============================================================

from .physical_components import (
    add_box,
    add_cylinder,
    add_pv_panel,
    add_sensor_mount,
    add_battery,
    add_controller,
    add_esp32,
    add_ina219,
    add_dht22,
    add_ds18b20,
    add_ldr,
    add_load
)


from .sensor_models import (
    sensor_snapshot,
    load_visual_profile,
    build_visual_states,
    state_color
)


from .wiring import (
    add_wire,
    add_cable,
    add_signal
)


# ==============================================================
# SAFE NUMBER
# ==============================================================

def _safe(value):

    try:

        value = float(value)

        if math.isfinite(value):
            return value

        return 0.0

    except (TypeError, ValueError):

        return 0.0


# ==============================================================
# AI STATUS COLOR
# ==============================================================

def _ai_color(status):

    status = str(status).upper()

    if status == "CRITICAL":
        return "#EF4444"

    if status == "WARNING":
        return "#F59E0B"

    if status in ("NORMAL", "HEALTHY"):
        return "#22C55E"

    return "#94A3B8"


# ==============================================================
# CAMERA CONFIGURATION
# ==============================================================
#
# IMPORTANT
#
# These values match the camera used at the bottom of the file.
#
# The information boards are rotated so their FRONT faces this
# camera instead of remaining aligned with the world X/Y axes.
# ==============================================================

CAMERA_X = 1.38
CAMERA_Y = 1.52
CAMERA_Z = 1.08


# ==============================================================
# CAMERA FACING CARD
# ==============================================================


def _add_3d_text(
    fig,
    text,
    center,
    horizontal=None,
    vertical=None,
    board_width=1.0,
    board_height=1.0,
    color="#F8FAFC",
    size_ratio=0.62,
    line_width=3
):
    """
    Clean fixed-size text for a physical information board.

    The complete card is rendered as ONE text object so the title,
    measurements and status stay together on the board.
    """

    fig.add_trace(
        go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode="text",
            text=[str(text)],
            textfont=dict(
                family="Arial",
                size=9,
                color=color
            ),
            textposition="middle center",
            showlegend=False,
            hoverinfo="skip"
        )
    )

def _add_oriented_board(
    fig,
    center,
    horizontal,
    vertical,
    normal,
    width,
    height,
    depth,
    accent,
    title
):
    """
    Create a camera-facing physical board.

    The board is a real 3D cuboid whose local axes are:
        horizontal -> left/right on the board
        vertical   -> up/down on the board
        normal     -> front/back of the board

    Text uses these exact same axes.
    """

    cx, cy, cz = center

    # Front and back centers
    front = (
        cx + normal[0] * depth / 2.0,
        cy + normal[1] * depth / 2.0,
        cz + normal[2] * depth / 2.0,
    )

    back = (
        cx - normal[0] * depth / 2.0,
        cy - normal[1] * depth / 2.0,
        cz - normal[2] * depth / 2.0,
    )

    hw = width / 2.0
    hh = height / 2.0

    def corner(base, su, sv):
        return (
            base[0] + horizontal[0] * su * hw + vertical[0] * sv * hh,
            base[1] + horizontal[1] * su * hw + vertical[1] * sv * hh,
            base[2] + horizontal[2] * su * hw + vertical[2] * sv * hh,
        )

    # Front: bottom-left, bottom-right, top-right, top-left
    f0 = corner(front, -1, -1)
    f1 = corner(front,  1, -1)
    f2 = corner(front,  1,  1)
    f3 = corner(front, -1,  1)

    # Back
    b0 = corner(back, -1, -1)
    b1 = corner(back,  1, -1)
    b2 = corner(back,  1,  1)
    b3 = corner(back, -1,  1)

    verts = [f0, f1, f2, f3, b0, b1, b2, b3]

    # Six faces, outward-facing winding.
    faces = [
        (0, 1, 2), (0, 2, 3),   # front
        (4, 6, 5), (4, 7, 6),   # back
        (0, 4, 5), (0, 5, 1),   # bottom
        (3, 2, 6), (3, 6, 7),   # top
        (0, 3, 7), (0, 7, 4),   # left
        (1, 5, 6), (1, 6, 2),   # right
    ]

    fig.add_trace(
        go.Mesh3d(
            x=[v[0] for v in verts],
            y=[v[1] for v in verts],
            z=[v[2] for v in verts],
            i=[f[0] for f in faces],
            j=[f[1] for f in faces],
            k=[f[2] for f in faces],
            color="#07111F",
            opacity=0.98,
            flatshading=True,
            name=title,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Front border
    border = [f0, f1, f2, f3, f0]
    fig.add_trace(
        go.Scatter3d(
            x=[p[0] for p in border],
            y=[p[1] for p in border],
            z=[p[2] for p in border],
            mode="lines",
            line=dict(color=accent, width=5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Small left status strip, using the SAME board axes.
    strip_w = 0.055
    strip_center_u = -hw + strip_w / 2.0

    strip_pts = [
        corner(front, -1, -1),
        corner(front, -1 + 2 * strip_w / width, -1),
        corner(front, -1 + 2 * strip_w / width, 1),
        corner(front, -1, 1),
    ]

    fig.add_trace(
        go.Mesh3d(
            x=[p[0] + normal[0] * 0.004 for p in strip_pts],
            y=[p[1] + normal[1] * 0.004 for p in strip_pts],
            z=[p[2] + normal[2] * 0.004 for p in strip_pts],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=accent,
            opacity=1.0,
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _card(
    fig,
    x,
    y,
    z,
    title,
    lines,
    accent,
    width=1.0
):
    """
    Clean floating live-data labels for the Digital Twin.

    There is intentionally NO black board.
    The text is positioned in 3D using the same camera-facing
    coordinate system used previously, while the actual values
    come directly from the live sensor_data passed into this file.

    Fixed font sizes are used for a clean instrument-style display.
    """

    length_xy = math.sqrt(CAMERA_X ** 2 + CAMERA_Y ** 2)

    front_x = CAMERA_X / length_xy
    front_y = CAMERA_Y / length_xy

    horizontal = (
        -front_y,
        front_x,
        0.0
    )

    # Keep the labels slightly in front of the physical scene.
    offset = 0.18

    center_x = x + front_x * offset
    center_y = y + front_y * offset

    # Scale only the POSITION of a card.
    # Font size never changes.
    half_width = 0.95 * width
    left_x = center_x - horizontal[0] * half_width
    left_y = center_y - horizontal[1] * half_width

    # ----------------------------------------------------------
    # TITLE
    # ----------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(
            x=[left_x],
            y=[left_y],
            z=[z + 0.42],
            mode="text",
            text=[str(title)],
            textfont=dict(
                family="Arial",
                size=12,
                color=accent
            ),
            textposition="middle left",
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # ----------------------------------------------------------
    # LIVE VALUES
    # ----------------------------------------------------------

    line_spacing = 0.22

    for index, line in enumerate(lines):

        # Final line is normally NORMAL / HIGH / LOW or ON / OFF.
        # Give it the hardware/status colour.
        if index == len(lines) - 1:
            line_color = accent
        else:
            line_color = "#F1F5F9"

        fig.add_trace(
            go.Scatter3d(
                x=[left_x],
                y=[left_y],
                z=[
                    z + 0.18 - index * line_spacing
                ],
                mode="text",
                text=[str(line)],
                textfont=dict(
                    family="Arial",
                    size=10,
                    color=line_color
                ),
                textposition="middle left",
                showlegend=False,
                hoverinfo="skip"
            )
        )


# ==============================================================
# SMALL LABEL
# ==============================================================

def _small_label(
    fig,
    x,
    y,
    z,
    text,
    color="#E5E7EB",
    size=9
):

    fig.add_trace(
        go.Scatter3d(

            x=[x],
            y=[y],
            z=[z],

            mode="text",

            text=[text],

            textfont=dict(
                size=size,
                color=color
            ),

            showlegend=False,

            hoverinfo="skip"
        )
    )


# ==============================================================
# LIVE POWER FLOW
# ==============================================================

def _flow(
    fig,
    start,
    end,
    power,
    name,
    color
):

    power = max(
        0.0,
        _safe(power)
    )

    # ----------------------------------------------------------
    # NO POWER
    # ----------------------------------------------------------

    if power <= 0:

        add_wire(
            fig,
            [start, end],
            "#475569",
            3,
            name,
            "dot"
        )

        return

    # ----------------------------------------------------------
    # FLOW SEGMENTS
    # ----------------------------------------------------------

    count = max(
        3,
        min(
            10,
            int(abs(power)) + 3
        )
    )

    points = []

    for i in range(count):

        t = i / (count - 1)

        points.append(
            (
                start[0] +
                (end[0] - start[0]) * t,

                start[1] +
                (end[1] - start[1]) * t,

                start[2] +
                (end[2] - start[2]) * t
            )
        )

    add_wire(
        fig,
        points,
        color,
        8,
        name
    )

    # ----------------------------------------------------------
    # FLOW MARKERS
    # ----------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(

            x=[
                p[0]
                for p in points
            ],

            y=[
                p[1]
                for p in points
            ],

            z=[
                p[2]
                for p in points
            ],

            mode="markers",

            marker=dict(
                size=5,
                color=color,
                symbol="diamond"
            ),

            showlegend=False,

            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Measured power: {power:.2f} W"
                "<extra></extra>"
            )
        )
    )


# ==============================================================
# SENSOR MARKER
# ==============================================================

def _dynamic_sensor_marker(
    fig,
    x,
    y,
    z,
    label,
    value,
    state
):

    color = state_color(
        state
    )

    fig.add_trace(
        go.Scatter3d(

            x=[x],
            y=[y],
            z=[z],

            mode="markers+text",

            marker=dict(
                size=11,
                color=color,
                line=dict(
                    color="#FFFFFF",
                    width=2
                )
            ),

            text=[label],

            textposition="top center",

            textfont=dict(
                size=9,
                color="#F8FAFC"
            ),

            showlegend=False,

            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Live value: {value:.2f}<br>"
                f"Visual state: {state}"
                "<extra></extra>"
            )
        )
    )


# ==============================================================
# MAIN DIGITAL TWIN
# ==============================================================

def create_visual_twin(
    sensor_data,
    overall_status="NORMAL"
):

    # ==========================================================
    # REAL SENSOR DATA
    # ==========================================================

    data = sensor_snapshot(
        sensor_data
    )

    # ==========================================================
    # DATASET PROFILE
    # ==========================================================

    profile = load_visual_profile()

    states = build_visual_states(
        data,
        profile
    )

    # ==========================================================
    # AI STATUS
    # ==========================================================

    ai_status = str(
        overall_status
    ).upper()

    ai_color = _ai_color(
        ai_status
    )

    # ==========================================================
    # VISUAL COLORS
    # ==========================================================

    solar_color = state_color(
        states["Solar_Power"]
    )

    battery_color = state_color(
        states["Battery_Voltage"]
    )

    environment_color = state_color(
        states["DHT22_Temperature"]
    )

    # ==========================================================
    # FIGURE
    # ==========================================================

    fig = go.Figure()

    # ==========================================================
    # BASE PLATFORM
    # ==========================================================

    add_box(
        fig,
        (
            -1.0,
            11.0,
            -1.15,
            7.7,
            -0.25,
            0
        ),
        "#25364B",
        "Digital Twin platform",
        opacity=0.96
    )

    # ==========================================================
    # SOLAR PANEL
    # ==========================================================

    add_pv_panel(
        fig,
        origin=(0.35, 0.55),
        width=6.15,
        height=3.85,
        z=4.05,
        power=data["Solar_Power"],
        state_color=solar_color
    )

    # ==========================================================
    # SENSOR MOUNTS
    # ==========================================================

    add_sensor_mount(
        fig,
        1.05,
        4.65,
        0,
        3.65
    )

    add_sensor_mount(
        fig,
        2.70,
        4.65,
        0,
        3.65
    )

    add_sensor_mount(
        fig,
        4.35,
        4.65,
        0,
        3.65
    )

    # ==========================================================
    # PHYSICAL SENSORS
    # ==========================================================

    add_ldr(
        fig,
        (1.05, 4.65, 3.82)
    )

    add_dht22(
        fig,
        (2.70, 4.65, 3.82)
    )

    add_ds18b20(
        fig,
        (4.35, 4.65, 3.82)
    )

    # ==========================================================
    # LIVE SENSOR MARKERS
    # ==========================================================

    _dynamic_sensor_marker(
        fig,
        1.05,
        4.95,
        4.28,
        "LDR",
        data["LDR"],
        states["LDR"]
    )

    _dynamic_sensor_marker(
        fig,
        2.70,
        4.95,
        4.28,
        "DHT22",
        data["DHT22_Temperature"],
        states["DHT22_Temperature"]
    )

    _dynamic_sensor_marker(
        fig,
        4.35,
        4.95,
        4.28,
        "DS18B20",
        data["DS18B20_Temperature"],
        states["DS18B20_Temperature"]
    )

    # ==========================================================
    # SOLAR INA219
    # ==========================================================

    add_ina219(
        fig,
        (6.15, 3.25, 1.28),
        "INA219 SOLAR"
    )

    # ==========================================================
    # MPPT CONTROLLER
    # ==========================================================

    add_controller(
        fig,
        (7.65, 4.25, 1.35),
        ai_status
    )

    # ==========================================================
    # BATTERY INA219
    # ==========================================================

    add_ina219(
        fig,
        (7.65, 2.25, 0.76),
        "INA219 BATTERY"
    )

    # ==========================================================
    # BATTERY
    # ==========================================================

    add_battery(
        fig,
        (8.55, 1.20, 1.15),
        data["Battery_Voltage"],
        data["Battery_Power"],
        battery_color
    )

    # ==========================================================
    # ESP32
    # ==========================================================

    add_esp32(
        fig,
        (4.05, 6.25, 0.48)
    )

    # ==========================================================
    # DC LOAD
    # ==========================================================

    add_load(
        fig,
        (10.0, 4.15, 0.72)
    )

    # ==========================================================
    # POWER CABLES
    # ==========================================================

    add_cable(
        fig,
        (5.55, 1.25, 4.08),
        (6.15, 3.00, 1.30),
        "#DC2626",
        "PV positive cable",
        7
    )

    add_cable(
        fig,
        (5.30, 1.05, 4.02),
        (5.85, 3.00, 1.25),
        "#111111",
        "PV negative cable",
        7
    )

    add_cable(
        fig,
        (6.15, 3.25, 1.32),
        (7.05, 4.05, 1.35),
        "#DC2626",
        "Solar positive",
        6
    )

    add_cable(
        fig,
        (6.15, 3.00, 1.25),
        (7.00, 4.00, 1.25),
        "#111111",
        "Solar negative",
        6
    )

    add_cable(
        fig,
        (7.65, 4.00, 1.40),
        (8.00, 2.10, 1.72),
        "#DC2626",
        "Controller positive",
        7
    )

    add_cable(
        fig,
        (8.05, 4.00, 1.25),
        (8.95, 2.10, 1.72),
        "#111111",
        "Controller negative",
        7
    )

    add_cable(
        fig,
        (8.00, 1.10, 1.82),
        (9.55, 3.80, 1.00),
        "#DC2626",
        "Battery positive",
        6
    )

    add_cable(
        fig,
        (9.10, 1.10, 1.75),
        (9.80, 3.80, 0.95),
        "#111111",
        "Battery negative",
        6
    )

    # ==========================================================
    # SENSOR SIGNAL WIRES
    # ==========================================================

    add_signal(
        fig,
        (1.05, 4.65, 3.82),
        (3.55, 6.25, 0.55),
        "#FACC15",
        "LDR signal"
    )

    add_signal(
        fig,
        (2.70, 4.65, 3.82),
        (3.82, 6.25, 0.55),
        "#F8FAFC",
        "DHT22 signal"
    )

    add_signal(
        fig,
        (4.35, 4.65, 3.82),
        (4.08, 6.25, 0.55),
        "#EF4444",
        "DS18B20 signal"
    )

    # ==========================================================
    # INA219 SIGNALS
    # ==========================================================

    add_signal(
        fig,
        (6.15, 3.25, 1.35),
        (4.55, 6.25, 0.55),
        "#A855F7",
        "Solar INA219 signal"
    )

    add_signal(
        fig,
        (7.65, 2.25, 0.82),
        (4.30, 6.25, 0.55),
        "#A855F7",
        "Battery INA219 signal"
    )

    # ==========================================================
    # SOLAR POWER FLOW
    # ==============================================================

    _flow(
        fig,
        (5.65, 2.0, 4.15),
        (6.05, 3.15, 1.45),
        data["Solar_Power"],
        "LIVE SOLAR POWER",
        solar_color
    )

    # ==========================================================
    # BATTERY POWER FLOW
    # ==============================================================

    _flow(
        fig,
        (8.00, 3.90, 1.45),
        (8.30, 2.05, 1.70),
        abs(data["Battery_Power"]),
        "LIVE BATTERY POWER",
        battery_color
    )

    # ==========================================================
    # CAMERA-FACING INFORMATION BOARDS
    # ==============================================================

    # ----------------------------------------------------------
    # PV
    # ----------------------------------------------------------

    _card(
        fig,
        0.45,
        6.05,
        5.35,
        "☀ PV PANEL",
        [
            f"V  {data['Solar_Voltage']:.2f} V",
            f"I  {data['Solar_Current']:.2f} A",
            f"P  {data['Solar_Power']:.2f} W",
            states["Solar_Power"]
        ],
        solar_color,
        width=0.90
    )

    # ----------------------------------------------------------
    # ENVIRONMENT
    # ----------------------------------------------------------

    _card(
        fig,
        9.25,
        5.90,
        5.15,
        "🌡 ENVIRONMENT",
        [
            f"T  {data['DHT22_Temperature']:.1f} C  {states['DHT22_Temperature']}",
            f"H  {data['DHT22_Humidity']:.1f}%  {states['DHT22_Humidity']}",
            f"S  {data['DS18B20_Temperature']:.1f} C  {states['DS18B20_Temperature']}"
        ],
        environment_color,
        width=1.00
    )

    # ----------------------------------------------------------
    # BATTERY
    # ----------------------------------------------------------

    _card(
        fig,
        0.55,
        1.35,
        3.05,
        "🔋 BATTERY",
        [
            f"V  {data['Battery_Voltage']:.2f} V",
            f"I  {data['Battery_Current']:.2f} A",
            f"P  {data['Battery_Power']:.2f} W",
            states["Battery_Voltage"]
        ],
        battery_color,
        width=0.90
    )

    # ----------------------------------------------------------
    # MPPT
    # ----------------------------------------------------------

    _card(
        fig,
        5.65,
        5.85,
        2.10,
        "⚙ MPPT",
        [
            "Mode  MPPT",
            f"In    {data['Solar_Power']:.2f} W",
            f"Out   {data['Battery_Power']:.2f} W",
            ai_status
        ],
        ai_color,
        width=0.95
    )

    # ----------------------------------------------------------
    # AI STATUS
    # ----------------------------------------------------------

    _card(
        fig,
        7.35,
        4.85,
        3.45,
        "🧠 AI STATUS",
        [
            "RF / LSTM / AE",
            "Decision Engine",
            ai_status
        ],
        ai_color,
        width=0.90
    )

    # ----------------------------------------------------------
    # ESP32
    # ----------------------------------------------------------

    _card(
        fig,
        9.65,
        5.05,
        2.55,
        "📡 ESP32",
        [
            "COM11 • 115200",
            "Connected",
            "LIVE"
        ],
        "#22C55E",
        width=0.85
    )

    # ----------------------------------------------------------
    # DC LOAD
    # ----------------------------------------------------------

    load_power = abs(
        data["Battery_Power"]
    )

    load_status = (
        "ON"
        if load_power > 0.05
        else "OFF"
    )

    _card(
        fig,
        10.10,
        1.25,
        1.35,
        "💡 DC LOAD",
        [
            f"P  {load_power:.2f} W",
            load_status
        ],
        "#94A3B8",
        width=0.80
    )

    # ==========================================================
    # VISUAL STATE LEGEND
    # ==============================================================

    legend_x = 8.75
    legend_y = 0.90
    legend_z = 4.15

    _small_label(
        fig,
        legend_x,
        legend_y,
        legend_z,
        "VISUAL STATE",
        "#CBD5E1",
        9
    )

    legend_items = [
        ("NORMAL", "#22C55E"),
        ("LOW", "#38BDF8"),
        ("HIGH", "#F59E0B")
    ]

    for index, (name, color) in enumerate(
        legend_items
    ):

        z_position = (
            legend_z
            - 0.25
            - index * 0.22
        )

        add_cylinder(
            fig,
            (
                legend_x - 0.55,
                legend_y,
                z_position
            ),
            0.035,
            0.08,
            color,
            name,
            segments=12
        )

        _small_label(
            fig,
            legend_x - 0.40,
            legend_y,
            z_position,
            name,
            "#E5E7EB",
            8
        )

    # ==========================================================
    # HARDWARE LABELS
    # ==============================================================

    _small_label(
        fig,
        3.45,
        4.62,
        4.52,
        "PV ARRAY",
        "#F8FAFC",
        12
    )

    _small_label(
        fig,
        6.15,
        3.00,
        1.75,
        "INA219",
        "#86EFAC",
        8
    )

    _small_label(
        fig,
        7.65,
        4.05,
        2.25,
        "MPPT",
        "#E5E7EB",
        9
    )

    _small_label(
        fig,
        8.55,
        1.05,
        2.00,
        "12V BATTERY",
        "#86EFAC",
        9
    )

    _small_label(
        fig,
        10.0,
        4.10,
        1.35,
        "DC LOAD",
        "#E5E7EB",
        9
    )

    _small_label(
        fig,
        4.05,
        6.20,
        0.92,
        "ESP32",
        "#A7F3D0",
        9
    )

    # ==========================================================
    # FINAL SCENE
    # ==============================================================

    fig.update_layout(

        title=dict(
            text="☀️ SolarSentinel-X — Digital Twin V4",
            x=0.5,
            xanchor="center",
            font=dict(
                size=24,
                color="#F8FAFC"
            )
        ),

        scene=dict(

            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                title=""
            ),

            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                title=""
            ),

            zaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                title=""
            ),

            bgcolor="#06101D",

            camera=dict(
                eye=dict(
                    x=CAMERA_X,
                    y=CAMERA_Y,
                    z=CAMERA_Z
                )
            ),

            aspectmode="manual",

            aspectratio=dict(
                x=1.58,
                y=1.05,
                z=0.82
            )
        ),

        height=820,

        margin=dict(
            l=0,
            r=0,
            t=65,
            b=0
        ),

        paper_bgcolor="rgba(0,0,0,0)",

        showlegend=False,

        font=dict(
            color="#E5E7EB"
        )
    )

    return fig