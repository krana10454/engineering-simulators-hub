import streamlit as st
import numpy as np
import matplotlib.pyplot as plt # Keeping for rod diagram
from io import BytesIO

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration for Streamlit Page ---
st.set_page_config(page_title="1D Heat Conduction Simulator", layout="centered")

st.title("🔥 1D Heat Conduction Simulator (Steady + Transient)")

# --- Enforce Dark Theme Only ---
st.markdown("""
    <style>
        .stApp {
            background-color: #1e1e1e; /* Dark background */
            color: white; /* Default text color for the app */
        }
        /* Targeted elements for dark theme */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown,
        .stText,
        div[data-testid="stRadio"] label > div > p, /* Radio button text */
        div[data-testid="stRadio"] label span, /* Radio button text fallback */
        div[data-testid="stSelectbox"] label, /* Selectbox label */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] div[data-testid="stTextInput"] input, /* Selectbox current value */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] span, /* Selectbox dropdown text */
        div[data-testid="stNumberInput"] label, /* Number input label */
        div[data-testid="stNumberInput"] div[data-baseweb="input"] input, /* Number input value */
        div[data-testid="stCheckbox"] label, /* Checkbox label */
        /* Streamlit internal text classes that might need override */
        .st-emotion-cache-16sx4w7, /* Common label/text class */
        .st-emotion-cache-1n1030j { /* Another generic text/label class */
            color: white !important;
        }
        /* Radio button dot color */
        .stRadio > label > div > div {
            border-color: white !important;
        }
        .stRadio > label > div > div > div {
            background-color: white !important;
        }
        /* Style for Play button in Plotly */
        .modebar-container .modebar-btn {
            color: white !important; /* Button icon color */
            background-color: #333333 !important; /* Button background */
            border: 1px solid #555555 !important;
        }
        .modebar-container .modebar-btn:hover {
            background-color: #555555 !important;
        }
        /* Style for Plotly Slider components for dark theme */
        .modebar-container .slider-container .slider-label {
            color: white !important;
        }
        .modebar-container .slider-container .slider-value {
            color: white !important;
        }
        .modebar-container .slider-container .slider-track {
            background-color: #555555 !important;
        }
        .modebar-container .slider-container .slider-handle {
            background-color: #cccccc !important;
            border: 1px solid #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Rod Diagram Visualization (New Feature) ---
st.subheader("Rod Geometry & Boundary Conditions")

def plot_rod_diagram(L, bc_left_type, T1, h_left, T_inf_left,
                     bc_right_type, T2, h_right, T_inf_right):
    fig, ax = plt.subplots(figsize=(8, 2), facecolor='#1e1e1e')
    
    # Rod
    rod_y = 0.5
    ax.plot([0, L], [rod_y, rod_y], 'k-', linewidth=5, solid_capstyle='butt', color='#8B4513') # Brown rod

    # Length label
    ax.text(L/2, rod_y + 0.15, f'L = {L} m', ha='center', va='bottom', color='white', fontsize=10)

    # Left Boundary
    if bc_left_type == "Fixed Temperature (Dirichlet)":
        ax.plot([0, 0], [rod_y - 0.1, rod_y + 0.1], 'b-', linewidth=2)
        ax.text(0, rod_y + 0.3, f'$T_1$ = {T1}°C', ha='center', va='bottom', color='cyan', fontsize=10)
        ax.annotate('', xy=(-0.05, rod_y), xytext=(0, rod_y + 0.2),
                    arrowprops=dict(facecolor='cyan', shrink=0.05, width=1, headwidth=5),
                    horizontalalignment='center', verticalalignment='bottom')
    elif bc_left_type == "Insulated (Neumann)":
        ax.plot([0, 0], [rod_y - 0.1, rod_y + 0.1], 'r--', linewidth=2)
        ax.text(0, rod_y + 0.3, r'$\frac{\partial T}{\partial x}=0$', ha='center', va='bottom', color='red', fontsize=12)
        ax.annotate('', xy=(-0.05, rod_y + 0.1), xytext=(0, rod_y + 0.2),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                    horizontalalignment='center', verticalalignment='bottom')
    elif bc_left_type == "Convective (Robin)":
        ax.plot([0, 0], [rod_y - 0.1, rod_y + 0.1], 'g:', linewidth=2)
        ax.text(0, rod_y + 0.3, f'h={h_left}, $T_\\infty$={T_inf_left}°C', ha='center', va='bottom', color='lightgreen', fontsize=10)
        ax.annotate('', xy=(-0.05, rod_y + 0.1), xytext=(0, rod_y + 0.2),
                    arrowprops=dict(facecolor='lightgreen', shrink=0.05, width=1, headwidth=5),
                    horizontalalignment='center', verticalalignment='bottom')

    # Right Boundary
    if bc_right_type == "Fixed Temperature (Dirichlet)":
        ax.plot([L, L], [rod_y - 0.1, rod_y + 0.1], 'b-', linewidth=2)
        ax.text(L, rod_y + 0.3, f'$T_2$ = {T2}°C', ha='center', va='bottom', color='cyan', fontsize=10)
        ax.annotate('', xy=(L + 0.05, rod_y), xytext=(L, rod_y + 0.2),
                    arrowprops=dict(facecolor='cyan', shrink=0.05, width=1, headwidth=5),
                    horizontalalignment='center', verticalalignment='bottom')
    elif bc_right_type == "Insulated (Neumann)":
        ax.plot([L, L], [rod_y - 0.1, rod_y + 0.1], 'r--', linewidth=2)
        ax.text(L, rod_y + 0.3, r'$\frac{\partial T}{\partial x}=0$', ha='center', va='bottom', color='red', fontsize=12)
        ax.annotate('', xy=(L + 0.05, rod_y + 0.1), xytext=(L, rod_y + 0.2),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                    horizontalalignment='center', verticalalignment='bottom')
    elif bc_right_type == "Convective (Robin)":
        ax.plot([L, L], [rod_y - 0.1, rod_y + 0.1], 'g:', linewidth=2)
        ax.text(L, rod_y + 0.3, f'h={h_right}, $T_\\infty$={T_inf_right}°C', ha='center', va='bottom', color='lightgreen', fontsize=10)
        ax.annotate('', xy=(L + 0.05, rod_y + 0.1), xytext=(L, rod_y + 0.2),
                    arrowprops=dict(facecolor='lightgreen', shrink=0.05, width=1, headwidth=5),
                    horizontalalignment='center', verticalalignment='bottom')


    ax.set_xlim(-0.1 * L, 1.1 * L)
    ax.set_ylim(0, 1.2)
    ax.axis('off') # Hide axes

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    plt.close(fig) # Close the figure to free up memory
    return buf.getvalue()


# --- Sidebar for Inputs (New Feature) ---
with st.sidebar:
    st.header("⚙️ Simulation Inputs")

    mode = st.radio("Select Mode", ["Steady-State", "Transient"])

    st.subheader("Rod Properties")
    L = st.number_input("Rod Length (L) [m]", value=1.0)
    Nx = st.slider("Number of spatial nodes (Nx)", min_value=20, max_value=200, value=100, step=10)
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]
    q_gen = st.number_input("Internal Heat Generation (q_gen) [W/m³]", value=0.0)

    st.subheader("Material Properties")
    materials = {
        "Copper": 401,
        "Aluminum": 237,
        "Steel": 50,
        "Brass": 109,
        "Custom": None
    }
    material = st.selectbox("Select Material", list(materials.keys()))
    k_initial = materials[material] if materials[material] else st.number_input("Enter Thermal Conductivity (k) [W/m·K]", value=200.0)

    # --- Variable Thermal Conductivity k(x) ---
    st.markdown("---")
    st.subheader("🔬 Thermal Conductivity (k)")
    k_vary_type = st.radio("Thermal Conductivity Type", ["Constant", "Linear (k0 + k1*x)"])
    if k_vary_type == "Constant":
        k_func = lambda pos: k_initial
    else: # Linear
        k0 = st.number_input("k0 (Constant part of k(x))", value=k_initial)
        k1 = st.number_input("k1 (Coefficient for x in k(x))", value=0.0)
        k_func = lambda pos: k0 + k1 * pos
        if st.checkbox("Show k(x) profile"):
            k_values = [k_func(val) for val in x]
            k_fig = go.Figure(data=go.Scatter(x=x, y=k_values, mode='lines', name='k(x)', line=dict(color='orange')),
                                layout=go.Layout(title='Thermal Conductivity Profile k(x)',
                                                 xaxis_title='Position (x)',
                                                 yaxis_title='k [W/m·K]',
                                                 font_color="white", paper_bgcolor="#1e1e1e", plot_bgcolor="#2b2b2b",
                                                 xaxis=dict(gridcolor='gray', zerolinecolor='gray', title_font=dict(color='white'), tickfont=dict(color='white')),
                                                 yaxis=dict(gridcolor='gray', zerolinecolor='gray', title_font=dict(color='white'), tickfont=dict(color='white'))))
            st.plotly_chart(k_fig, use_container_width=True)


    # --- Heat Loss to Surroundings ---
    st.markdown("---")
    st.subheader("🌬️ Heat Loss to Surroundings (Convection)")
    enable_heat_loss = st.checkbox("Enable Convective Heat Loss (h*P/A_c * (T - T_infinity))")
    if enable_heat_loss:
        h_loss = st.number_input("Convective Heat Transfer Coeff (h_loss) [W/m²·K]", value=10.0)
        T_infinity_loss = st.number_input("Ambient Temperature (T_infinity_loss) [°C]", value=20.0)
        perimeter = st.number_input("Perimeter (P) [m]", value=0.1) # Assuming a rod, e.g., for a circle 2*pi*r
        cross_sectional_area = st.number_input("Cross-sectional Area (A_c) [m²]", value=0.001) # e.g., for a circle pi*r^2
        # Define a combined heat loss coefficient for easier use in FDM
        hpAc = h_loss * perimeter / cross_sectional_area
    else:
        hpAc = 0.0
        T_infinity_loss = 0.0 # Will not be used if heat loss is disabled


    # --- Boundary Conditions ---
    st.markdown("---")
    st.subheader("📍 Boundary Conditions")
    # Left End (x=0)
    st.markdown("#### Left End (x=0)")
    bc_left_type = st.radio("Type for Left End", ["Fixed Temperature (Dirichlet)", "Insulated (Neumann)", "Convective (Robin)"], key="bc_left_type")
    T1, h_left, T_inf_left = None, None, None # Initialize to avoid UnboundLocalError
    if bc_left_type == "Fixed Temperature (Dirichlet)":
        T1 = st.number_input("Temperature at Left End (T1) [°C]", value=100.0, key="T1")
    elif bc_left_type == "Convective (Robin)":
        h_left = st.number_input("Convective Coeff at Left (h_L) [W/m²·K]", value=5.0, key="h_L")
        T_inf_left = st.number_input("Ambient Temp at Left (T_inf_L) [°C]", value=20.0, key="T_inf_L")

    # Right End (x=L)
    st.markdown("#### Right End (x=L)")
    bc_right_type = st.radio("Type for Right End", ["Fixed Temperature (Dirichlet)", "Insulated (Neumann)", "Convective (Robin)"], key="bc_right_type")
    T2, h_right, T_inf_right = None, None, None # Initialize to avoid UnboundLocalError
    if bc_right_type == "Fixed Temperature (Dirichlet)":
        T2 = st.number_input("Temperature at Right End (T2) [°C]", value=25.0, key="T2")
    elif bc_right_type == "Convective (Robin)":
        h_right = st.number_input("Convective Coeff at Right (h_R) [W/m²·K]", value=5.0, key="h_R")
        T_inf_right = st.number_input("Ambient Temp at Right (T_inf_R) [°C]", value=20.0, key="T_inf_R")

    st.markdown("---")
    st.subheader("⏱️ Simulation Parameters")
    show_calc = st.checkbox("Show Calculation Steps")

    # Initialize transient specific variables to avoid UnboundLocalError if mode is Steady-State
    alpha_input, rho, cp, dt, total_time, Nt = 1e-5, 2700.0, 900.0, 0.01, 10.0, 1000
    initial_cond_type = "Uniform"
    T_uniform_init, T_left_init, T_right_init = 50.0, 100.0, 25.0
    animation_speed_ms = 50

    if mode == "Transient":
        st.markdown("#### Transient Specifics")
        alpha_input = st.number_input("Thermal Diffusivity (α) [m²/s]", value=1e-5, format="%e")
        rho = st.number_input("Density (ρ) [kg/m³]", value=2700.0)
        cp = st.number_input("Specific Heat (c_p) [J/kg·K]", value=900.0)
        
        # Calculate alpha based on k, rho, cp (if k is constant for this calculation)
        k_avg_for_alpha = k_initial if k_vary_type == "Constant" else np.mean([k_func(val) for val in x])
        st.info(f"Thermal diffusivity (α) used: {alpha_input:.2e} m²/s. (Calculated based on average k: {k_avg_for_alpha / (rho * cp):.2e})") # Show both

        dt_auto = 0.5 * dx**2 / alpha_input # Stability criterion (approximate for constant k)
        dt = st.number_input(f"Time Step (Δt) [s] (Recommended max: {dt_auto:.2e}s)", value=0.01)
        if dt > dt_auto:
            st.warning(f"Warning: Time step Δt ({dt:.2e}s) is larger than the stability limit ({dt_auto:.2e}s) for constant k. This may lead to unstable results. Consider reducing Δt or increasing Nx.")
        
        total_time = st.number_input("Total Simulation Time [s]", value=10.0)
        Nt = int(total_time / dt)

        st.subheader("Initial Conditions")
        initial_cond_type = st.radio("Initial Temperature Profile", ["Uniform", "Linear", "From Steady-State (if applicable)"])
        
        T_uniform_init, T_left_init, T_right_init = 50.0, 100.0, 25.0 # Set default values for initial_cond_type
        if initial_cond_type == "Uniform":
            T_uniform_init = st.number_input("Uniform Initial Temperature [°C]", value=50.0)
        elif initial_cond_type == "Linear":
            T_left_init = st.number_input("Initial Temp at x=0 [°C]", value=100.0)
            T_right_init = st.number_input("Initial Temp at x=L [°C]", value=25.0)

        # --- Animation Speed Control (New Feature) ---
        st.markdown("---")
        st.subheader("🚀 Animation Controls")
        animation_speed_ms = st.slider("Animation Speed (ms per frame)", min_value=10, max_value=500, value=50, step=10)


# Display the Rod Diagram (using the actual values from inputs)
st.image(plot_rod_diagram(L, bc_left_type, T1, h_left, T_inf_left,
                           bc_right_type, T2, h_right, T_inf_right),
         caption="Rod Schematic with Boundary Conditions", use_column_width=True)

# --- Calculation ---
if mode == "Steady-State":
    st.subheader("Steady-State Solution")
    
    A = np.zeros((Nx, Nx))
    B = np.zeros(Nx)

    k_vals = np.array([k_func(val) for val in x])

    # Internal Nodes (i=1 to Nx-2)
    for i in range(1, Nx - 1):
        k_i_plus_half = k_func(x[i] + dx/2)
        k_i_minus_half = k_func(x[i] - dx/2)

        A[i, i-1] = k_i_minus_half / dx**2
        A[i, i] = -(k_i_plus_half + k_i_minus_half) / dx**2 - hpAc
        A[i, i+1] = k_i_plus_half / dx**2
        B[i] = -q_gen - hpAc * T_infinity_loss

    # Boundary Conditions - Left End (x=0)
    if bc_left_type == "Fixed Temperature (Dirichlet)":
        A[0, 0] = 1.0
        B[0] = T1
    elif bc_left_type == "Insulated (Neumann)":
        # For insulated, the gradient is 0. Using a fictitious node method:
        # T_virtual = T_1 (node at dx from boundary)
        # So the FDM at node 0 uses T_1 for T_virtual
        # (k_0.5 * (T_1 - T_0)/dx - k_-0.5 * (T_0 - T_virtual)/dx) / dx + q_gen - hpAc * (T_0 - T_infinity_loss) = 0
        # (k_0.5 * (T_1 - T_0)/dx - k_-0.5 * (T_0 - T_1)/dx) / dx + ... = 0
        # This simplifies to 2 * k_0.5 * (T_1 - T_0) / dx^2 + ... = 0
        # A[0,0] = -2 * k_func(x[0] + dx/2) / dx**2 - hpAc
        # A[0,1] = 2 * k_func(x[0] + dx/2) / dx**2
        # B[0] = -q_gen - hpAc * T_infinity_loss
        # Simpler form T_0 = T_1 is often used for steady-state for simplicity:
        A[0, 0] = 1.0
        A[0, 1] = -1.0
        B[0] = 0.0

    elif bc_left_type == "Convective (Robin)":
        # Energy balance on a half-control volume at x=0
        # h_L * (T_inf_L - T_0) + k_0.5 * (T_1 - T_0)/dx + q_gen * dx/2 - hpAc * (T_0 - T_infinity_loss) * dx/2 = 0
        # Rearranging for a linear system:
        k_at_0_plus_dx_half = k_func(x[0] + dx/2)
        A[0,0] = -h_left - k_at_0_plus_dx_half / dx - hpAc * dx / 2
        A[0,1] = k_at_0_plus_dx_half / dx
        B[0] = -h_left * T_inf_left - q_gen * dx / 2 - hpAc * T_infinity_loss * dx / 2


    # Boundary Conditions - Right End (x=L)
    if bc_right_type == "Fixed Temperature (Dirichlet)":
        A[Nx-1, Nx-1] = 1.0
        B[Nx-1] = T2
    elif bc_right_type == "Insulated (Neumann)":
        # Similar to left insulated, using T_{Nx-1} = T_{Nx-2}
        # A[Nx-1, Nx-1] = -2 * k_func(x[Nx-1] - dx/2) / dx**2 - hpAc
        # A[Nx-1, Nx-2] = 2 * k_func(x[Nx-1] - dx/2) / dx**2
        # B[Nx-1] = -q_gen - hpAc * T_infinity_loss
        # Simpler form T_{Nx-1} = T_{Nx-2}:
        A[Nx-1, Nx-2] = 1.0
        A[Nx-1, Nx-1] = -1.0
        B[Nx-1] = 0.0

    elif bc_right_type == "Convective (Robin)":
        # Energy balance on a half-control volume at x=L
        # -k_{L-0.5} * (T_{N-1} - T_{N-2})/dx + h_R * (T_inf_R - T_{N-1}) + q_gen * dx/2 - hpAc * (T_{N-1} - T_infinity_loss) * dx/2 = 0
        k_at_L_minus_dx_half = k_func(x[Nx-1] - dx/2)
        A[Nx-1, Nx-2] = k_at_L_minus_dx_half / dx
        A[Nx-1, Nx-1] = -k_at_L_minus_dx_half / dx - h_right - hpAc * dx / 2
        B[Nx-1] = -h_right * T_inf_right - q_gen * dx / 2 - hpAc * T_infinity_loss * dx / 2

    try:
        T = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        st.error("Error: Could not solve the system for steady-state. Check boundary conditions and input parameters.")
        T = np.zeros(Nx) # Fallback to avoid error

    fig = go.Figure(data=go.Scatter(x=x, y=T, mode='lines', line=dict(color='red', width=2),
                                    # --- NEW: Hover data for Plotly ---
                                    hoverinfo='text',
                                    hovertemplate='<b>Position:</b> %{x:.2f} m<br><b>Temperature:</b> %{y:.2f} °C<extra></extra>'
                                    ),
                      layout=go.Layout(title='Steady-State Temperature Distribution',
                                       xaxis_title='Length (x) [m]',
                                       yaxis_title='Temperature (T) [°C]',
                                       font_color="white", paper_bgcolor="#1e1e1e", plot_bgcolor="#2b2b2b",
                                       xaxis=dict(gridcolor='gray', zerolinecolor='gray', title_font=dict(color='white'), tickfont=dict(color='white')),
                                       yaxis=dict(gridcolor='gray', zerolinecolor='gray', title_font=dict(color='white'), tickfont=dict(color='white')),
                                       title_font=dict(color='white')))
    st.plotly_chart(fig, use_container_width=True)

    if show_calc:
        st.markdown("### 📘 Calculation Steps (Steady-State)")
        st.markdown("""
        The steady-state solution for variable thermal conductivity and heat loss, with general boundary conditions,
        is obtained by solving a system of linear algebraic equations resulting from the Finite Difference Method (FDM).

        The general governing equation is:
        $ \\frac{d}{dx} \\left( k(x) \\frac{dT}{dx} \\right) + q_{gen} - \\frac{h_{loss} P}{A_c} (T - T_{\\infty,loss}) = 0 $

        This equation is discretized using central differences for internal nodes (i=1 to Nx-2),
        and appropriate finite difference approximations (or ghost nodes) for the boundary conditions.
        This forms a linear system of equations $ [A][T] = [B] $, which is solved for the temperature vector $ [T] $.
        """)

else: # Transient Mode
    st.subheader("Transient Solution")
    
    # --- Initial Conditions ---
    T = np.zeros(Nx)
    if initial_cond_type == "Uniform":
        T[:] = T_uniform_init
    elif initial_cond_type == "Linear":
        T = np.linspace(T_left_init, T_right_init, Nx)
    elif initial_cond_type == "From Steady-State (if applicable)":
        st.info("Using steady-state solution as initial condition. Recalculating steady-state for initial profile...")
        A_ss = np.zeros((Nx, Nx))
        B_ss = np.zeros(Nx)
        k_vals_ss = np.array([k_func(val) for val in x])

        for i in range(1, Nx - 1):
            k_i_plus_half_ss = k_func(x[i] + dx/2)
            k_i_minus_half_ss = k_func(x[i] - dx/2)
            A_ss[i, i-1] = k_i_minus_half_ss / dx**2
            A_ss[i, i] = -(k_i_plus_half_ss + k_i_minus_half_ss) / dx**2 - hpAc
            A_ss[i, i+1] = k_i_plus_half_ss / dx**2
            B_ss[i] = -q_gen - hpAc * T_infinity_loss

        # Left Boundary for Steady-State IC
        if bc_left_type == "Fixed Temperature (Dirichlet)":
            A_ss[0, 0] = 1.0
            B_ss[0] = T1
        elif bc_left_type == "Insulated (Neumann)":
            A_ss[0, 0] = 1.0
            A_ss[0, 1] = -1.0
            B_ss[0] = 0.0
        elif bc_left_type == "Convective (Robin)":
            k_at_0_plus_dx_half = k_func(x[0] + dx/2)
            A_ss[0,0] = -h_left - k_at_0_plus_dx_half / dx - hpAc * dx / 2
            A_ss[0,1] = k_at_0_plus_dx_half / dx
            B_ss[0] = -h_left * T_inf_left - q_gen * dx / 2 - hpAc * T_infinity_loss * dx / 2

        # Right Boundary for Steady-State IC
        if bc_right_type == "Fixed Temperature (Dirichlet)":
            A_ss[Nx-1, Nx-1] = 1.0
            B_ss[Nx-1] = T2
        elif bc_right_type == "Insulated (Neumann)":
            A_ss[Nx-1, Nx-2] = 1.0
            A_ss[Nx-1, Nx-1] = -1.0
            B_ss[Nx-1] = 0.0
        elif bc_right_type == "Convective (Robin)":
            k_at_L_minus_dx_half = k_func(x[Nx-1] - dx/2)
            A_ss[Nx-1, Nx-2] = k_at_L_minus_dx_half / dx
            A_ss[Nx-1, Nx-1] = -k_at_L_minus_dx_half / dx - h_right - hpAc * dx / 2
            B_ss[Nx-1] = -h_right * T_inf_right - q_gen * dx / 2 - hpAc * T_infinity_loss * dx / 2
        
        try:
            T = np.linalg.solve(A_ss, B_ss)
        except np.linalg.LinAlgError:
            st.error("Could not use steady-state solution as initial condition. Falling back to uniform (50°C).")
            T[:] = 50.0

    T_hist = [T.copy()]
    time_points = [0]
    
    k_vals = np.array([k_func(val) for val in x]) # Ensure k_vals is defined for transient loop

    # Transient simulation loop
    for n in range(Nt):
        T_new = T.copy()
        
        # Internal Nodes (i=1 to Nx-2)
        for i in range(1, Nx - 1):
            k_i_plus_half = k_func(x[i] + dx/2)
            k_i_minus_half = k_func(x[i] - dx/2)
            
            # Finite difference for d/dx(k*dT/dx)
            term_k_dtdx = (k_i_plus_half * (T[i+1] - T[i]) / dx - k_i_minus_half * (T[i] - T[i-1]) / dx) / dx
            
            heat_loss_term = hpAc * (T_infinity_loss - T[i])

            T_new[i] = T[i] + (dt / (rho * cp)) * (term_k_dtdx + q_gen + heat_loss_term)
        
        # Apply Boundary Conditions for T_new
        # Left End (x=0)
        if bc_left_type == "Fixed Temperature (Dirichlet)":
            T_new[0] = T1
        elif bc_left_type == "Insulated (Neumann)":
            k_half = k_func(x[0] + dx/2)
            # Energy balance on a half-volume at node 0 for insulated boundary
            # (rho*cp*dx/2) * dT/dt = k_half * (T1-T0)/dx + q_gen*dx/2 + hpAc*(T_infinity_loss - T0)*dx/2
            T_new[0] = T[0] + (dt / (rho * cp * dx / 2)) * \
                       (k_half * (T[1] - T[0]) / dx + q_gen * dx / 2 + hpAc * (T_infinity_loss - T[0]) * dx / 2)
        elif bc_left_type == "Convective (Robin)":
            k_at_0_plus_dx_half = k_func(x[0] + dx/2)
            # Energy balance on a half-volume at node 0 for convective boundary
            # (rho*cp*dx/2) * dT/dt = h_left * (T_inf_left - T0) + k_at_0_plus_dx_half * (T1-T0)/dx + q_gen*dx/2 + hpAc*(T_infinity_loss - T0)*dx/2
            T_new[0] = T[0] + (dt / (rho * cp * dx / 2)) * \
                       (h_left * (T_inf_left - T[0]) + k_at_0_plus_dx_half * (T[1] - T[0]) / dx + q_gen * dx / 2 + hpAc * (T_infinity_loss - T[0]) * dx / 2)

        # Right End (x=L)
        if bc_right_type == "Fixed Temperature (Dirichlet)":
            T_new[Nx-1] = T2
        elif bc_right_type == "Insulated (Neumann)":
            k_half = k_func(x[Nx-1] - dx/2)
            # Energy balance on a half-volume at node Nx-1 for insulated boundary
            # (rho*cp*dx/2) * dT/dt = -k_half * (T_Nx-1 - T_Nx-2)/dx + q_gen*dx/2 + hpAc*(T_infinity_loss - T_Nx-1)*dx/2
            T_new[Nx-1] = T[Nx-1] + (dt / (rho * cp * dx / 2)) * \
                          (-k_half * (T[Nx-1] - T[Nx-2]) / dx + q_gen * dx / 2 + hpAc * (T_infinity_loss - T[Nx-1]) * dx / 2)
        elif bc_right_type == "Convective (Robin)":
            k_at_L_minus_dx_half = k_func(x[Nx-1] - dx/2)
            # Energy balance on a half-volume at node Nx-1 for convective boundary
            # (rho*cp*dx/2) * dT/dt = -k_at_L_minus_dx_half * (T_Nx-1 - T_Nx-2)/dx + h_right * (T_inf_right - T_Nx-1) + q_gen*dx/2 + hpAc*(T_infinity_loss - T_Nx-1)*dx/2
            T_new[Nx-1] = T[Nx-1] + (dt / (rho * cp * dx / 2)) * \
                          (-k_at_L_minus_dx_half * (T[Nx-1] - T[Nx-2]) / dx + h_right * (T_inf_right - T[Nx-1]) + q_gen * dx / 2 + hpAc * (T_infinity_loss - T[Nx-1]) * dx / 2)

        T = T_new
        T_hist.append(T.copy())
        time_points.append((n + 1) * dt)

    # --- Plotly Figure for Animation (Always Dark Theme) ---
    fig = go.Figure(
        data=[go.Scatter(x=x, y=T_hist[0], mode='lines', line=dict(color='blue', width=2),
                                    # --- NEW: Hover data for Plotly ---
                                    hoverinfo='text',
                                    hovertemplate='<b>Position:</b> %{x:.2f} m<br><b>Temperature:</b> %{y:.2f} °C<extra></extra>'
                                    )],
        layout=go.Layout(
            xaxis=dict(range=[0, L], title="Length (x) [m]"),
            yaxis=dict(range=[np.min(T_hist) - 5, np.max(T_hist) + 5], title="Temperature (T) [°C]"),
            title="Temperature Profile Over Time (Transient)",
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.0,
                xanchor="left",
                y=1.1,
                yanchor="top",
                buttons=[dict(
                    label="▶️ Play",
                    method="animate",
                    args=[None, {"frame": {"duration": animation_speed_ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0, "easing": "linear"}}],
                    # REMOVED: font, bgcolor, bordercolor, borderwidth (as they are not valid here)
                ),
                dict(
                    label="⏸️ Pause",
                    method="animate",
                    args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    # REMOVED: font, bgcolor, bordercolor, borderwidth
                ),
                dict( # --- NEW: Restart Button ---
                    label="🔄 Restart",
                    method="animate",
                    args=[[f"frame_{0}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": False, "transition": {"duration": 0}}],
                    # REMOVED: font, bgcolor, bordercolor, borderwidth
                )
                ]
            )],
            sliders=[dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[f"frame_{k}"], {"mode": "immediate", "frame": {"duration": animation_speed_ms, "redraw": True}, "transition": {"duration": 0}}],
                        label=f"{time_points[k]:.2f}s"
                    ) for k in range(len(T_hist))
                ],
                transition={"duration": 0},
                xanchor="left",
                len=1.0,
                currentvalue={"font": {"size": 14, "color": "white"}, "prefix": "Time: ", "visible": True, "xanchor": "right"},
                yanchor="top",
                pad={"b": 10, "t": 50},
                active=0,
                # REMOVED: bgcolor, bordercolor, font from slider (as they are not valid here)
                # These are layout properties that might affect the slider itself, but not within 'steps' dict.
            )]
        )
    )

    frames = []
    for k, T_frame in enumerate(T_hist):
        frames.append(go.Frame(data=[go.Scatter(x=x, y=T_frame, mode='lines', line=dict(color='blue'))], name=f"frame_{k}"))
    fig.frames = frames

    # Apply Dark Theme to Plotly Figure
    fig.update_layout(
        font_color="white",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2b2b2b",
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', title_font=dict(color='white'), tickfont=dict(color='white')),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', title_font=dict(color='white'), tickfont=dict(color='white')),
        title_font=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)

    if show_calc:
        st.markdown("### 📘 Calculation Steps (Transient)")
        st.markdown(r"""
        The transient heat conduction is solved using the Finite Difference Method (FDM) with an explicit scheme.
        The general governing equation is:
        $ \rho c_p \frac{\partial T}{\partial t} = \frac{\partial}{\partial x} \left( k(x) \frac{\partial T}{\partial x} \right) + q_{gen} - \frac{h_{loss} P}{A_c} (T - T_{\infty,loss}) $

        Discretization for an internal node $i$:
        $ T_i^{n+1} = T_i^n + \frac{\Delta t}{\rho c_p} \left[ \frac{1}{\Delta x^2} \left( k_{i+1/2}(T_{i+1}^n - T_i^n) - k_{i-1/2}(T_i^n - T_{i-1}^n) \right) + q_{gen} - \frac{h_{loss} P}{A_c} (T_i^n - T_{\infty,loss}) \right] $

        Boundary conditions are applied using appropriate finite difference approximations for the first and last nodes (i=0 and i=Nx-1):
        * **Fixed Temperature (Dirichlet):** $ T_0 = T_1 $, $ T_{Nx-1} = T_2 $ (directly set the boundary node temperature).
        * **Insulated (Neumann):** $ \frac{\partial T}{\partial x} = 0 $. This is implemented using a virtual node approach where $ T_{-1} = T_1 $ (for left boundary) or $ T_{Nx} = T_{Nx-2} $ (for right boundary) to approximate the zero gradient. The update equation for the boundary node is modified to reflect zero flux.
        * **Convective (Robin):** $ -k \frac{\partial T}{\partial x} = h(T - T_{\infty}) $. This is implemented by considering an energy balance on a half-control volume at the boundary, incorporating both conduction and convection fluxes.
        """)

# --- Footer ---
st.markdown(f'<p style="color: white;">Developed by a Mechanical Engineer 💻⚙️</p>', unsafe_allow_html=True)