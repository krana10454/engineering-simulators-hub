# D:\streamlit_apps\simulator\pages\2_Pipe_Flow_Pressure_Drop_Simulator.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt # Import matplotlib for velocity profile visualization

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Pipe Flow Pressure Drop Simulator (Multi-Segment & Plots)", # You can choose which title you prefer
    page_icon="💧",
    layout="wide" # Using wide layout for better table and plot display
)

# --- Add Home Button at the Top ---
if st.button("🏠 Go to Home"):
    st.switch_page("0_Home.py") # Correctly points to the root Home file

st.title("💧 Pipe Flow Pressure Drop Simulator (Multi-Segment & Plots)") # Use this title or the one from set_page_config
st.write("Calculate total major pressure losses for multiple pipe segments and explore parameter sensitivities.")
st.markdown("---")

# --- Fluid Properties (Assumed constant for all segments for simplicity) ---
st.subheader("1. Fluid Properties (Constant for all segments)")
col1, col2 = st.columns(2)
with col1:
    rho = st.number_input("Fluid Density (ρ) [kg/m³]", value=1000.0, min_value=0.1, format="%.2f", key="rho_main")
with col2:
    mu = st.number_input("Fluid Dynamic Viscosity (μ) [Pa·s or kg/(m·s)]", value=0.001, min_value=1e-7, format="%.4e", key="mu_main")

st.subheader("2. Flow Conditions (Constant for all segments)")
Q = st.number_input("Volumetric Flow Rate (Q) [m³/s]", value=0.01, min_value=1e-6, format="%.4e", key="Q_main")

# --- Pipe Segment Properties (Dynamic Input) ---
st.subheader("3. Pipe Segment Properties")
st.markdown("Add or remove pipe segments and edit their properties for total pressure drop calculation.")

# Default data for the data editor
default_segments_data = {
    'Segment': [1, 2],
    'Diameter (m)': [0.1, 0.075],
    'Length (m)': [50.0, 30.0],
    'Roughness (m)': [0.000045, 0.000045]
}
df_segments = pd.DataFrame(default_segments_data)

# Use st.data_editor for interactive table input
edited_df_segments = st.data_editor(
    df_segments,
    num_rows="dynamic", # Allows adding/deleting rows
    column_config={
        "Segment": st.column_config.NumberColumn("Segment", help="Segment number for reference", disabled=True),
        "Diameter (m)": st.column_config.NumberColumn("Diameter (m)", format="%.3f", min_value=0.001),
        "Length (m)": st.column_config.NumberColumn("Length (m)", format="%.1f", min_value=0.1),
        "Roughness (m)": st.column_config.NumberColumn("Roughness (m)", format="%.6e", min_value=0.0)
    },
    hide_index=True,
    key="segment_editor"
)

# --- Calculation Function (reusable for plots) ---
def calculate_major_pressure_drop(D, L, epsilon, rho, mu, Q):
    """Calculates major pressure drop for a single pipe segment."""
    if D <= 0 or Q <= 0:
        return 0.0, 0.0, 0.0, "N/A", 0.0 # Return zeros for invalid inputs

    A = np.pi * D**2 / 4
    V_avg = Q / A # Average velocity
    Re = (rho * V_avg * D) / mu

    if Re <= 2300:
        flow_regime = "Laminar"
        f = 64 / Re
    else:
        flow_regime = "Turbulent"
        # Swamee-Jain equation
        try:
            # Added a small epsilon check for log argument safety
            log_arg = (epsilon / (3.7 * D)) + (5.74 / (Re**0.9))
            if log_arg <= 0:
                f = np.nan # Indicate error
            else:
                f = 0.25 / (np.log10(log_arg))**2
        except (ZeroDivisionError, ValueError): # Catch log of zero/negative
            f = np.nan # Indicate calculation failure
            flow_regime = "Error"
    
    if np.isnan(f) or V_avg == 0:
        delta_P_f = np.nan
    else:
        delta_P_f = f * (L / D) * (rho * V_avg**2 / 2)
    
    return delta_P_f, Re, f, flow_regime, V_avg

# --- Calculation Section ---
st.subheader("4. Calculation Results")

# Option to show calculation steps
show_calculations = st.checkbox("Show Detailed Calculation Steps for Each Segment", key="show_calc_main")

# Store the flow regime of the first segment to highlight in the velocity profile plot
first_segment_flow_regime = None

if st.button("Calculate Total Pressure Drop", key="calculate_button"):
    total_delta_P_f = 0.0
    segment_results_data = [] # To store results for each segment

    if edited_df_segments.empty:
        st.warning("Please add at least one pipe segment to calculate pressure drop.")
    else:
        st.markdown("### Segment-wise Results Summary")
        
        for index, row in edited_df_segments.iterrows():
            segment_num = index + 1
            D_seg = row['Diameter (m)']
            L_seg = row['Length (m)']
            epsilon_seg = row['Roughness (m)']

            delta_P_f_segment, Re, f, flow_regime, V_avg = calculate_major_pressure_drop(D_seg, L_seg, epsilon_seg, rho, mu, Q)
            
            # Capture flow regime of the first segment for visualization
            if index == 0:
                first_segment_flow_regime = flow_regime

            if not np.isnan(delta_P_f_segment):
                total_delta_P_f += delta_P_f_segment
                segment_results_data.append([
                    segment_num,
                    f"{D_seg:.3f}",
                    f"{L_seg:.1f}",
                    f"{epsilon_seg:.6e}",
                    f"{V_avg:.3f}",
                    f"{Re:.2f}",
                    flow_regime,
                    f"{f:.4f}",
                    f"{delta_P_f_segment:.2f}"
                ])
            else:
                st.error(f"Error calculating for Segment {segment_num}. Check inputs.")
                segment_results_data.append([
                    segment_num, D_seg, L_seg, epsilon_seg, "N/A", "N/A", "Error", "N/A", "Error"
                ])

            if show_calculations and not np.isnan(delta_P_f_segment):
                with st.expander(f"Detailed Calculations for Segment {segment_num}"):
                    st.markdown(f"**Segment {segment_num} Properties:** Diameter={D_seg:.3f} m, Length={L_seg:.1f} m, Roughness={epsilon_seg:.6e} m")
                    
                    A = np.pi * D_seg**2 / 4 # Recalculate for display
                    
                    st.markdown(f"""
                    #### Cross-sectional Area ($A$)
                    $ A = \\frac{{\\pi D^2}}{{4}} $
                    $ A = \\frac{{\\pi \\times ({D_seg:.3f})^2}}{{4}} $
                    $ A = {A:.4e} \\text{{ m}}^2 $
                    """)

                    st.markdown(f"""
                    #### Average Flow Velocity ($V$)
                    $ V = \\frac{{Q}}{{A}} $
                    $ V = \\frac{{{Q:.4e}}}{{{A:.4e}}} $
                    $ V = {V_avg:.3f} \\text{{ m/s}} $
                    """)

                    st.markdown(f"""
                    #### Reynolds Number ($Re$)
                    $ Re = \\frac{{\\rho V D}}{{\\mu}} $
                    $ Re = \\frac{{{rho:.2f} \\times {V_avg:.3f} \\times {D_seg:.3f}}}{{{mu:.4e}}} $
                    $ Re = {Re:.2f} $
                    """)

                    if flow_regime == "Laminar":
                        st.markdown(f"""
                        #### Friction Factor ($f$) - Laminar Flow
                        Since $Re \\le 2300$, the flow is laminar.
                        $ f = \\frac{{64}}{{Re}} $
                        $ f = \\frac{{64}}{{{Re:.2f}}} $
                        $ f = {f:.4f} $
                        """)
                    elif flow_regime == "Turbulent": # Turbulent
                        st.markdown(f"""
                        #### Friction Factor ($f$) - Turbulent Flow (Swamee-Jain Equation)
                        Since $Re > 2300$, the flow is turbulent.
                        $ f = \\frac{{0.25}}{{(\\log_{{10}}(\\frac{{\\epsilon}}{{3.7D}} + \\frac{{5.74}}{{Re^{{0.9}}}}))^2}} $
                        $ f = \\frac{{0.25}}{{(\\log_{{10}}(\\frac{{{epsilon_seg:.6e}}}{{3.7 \\times {D_seg:.3f}}} + \\frac{{5.74}}{{{Re:.2f}^{{0.9}}}}))^2}} $
                        $ f = {f:.4f} $
                        """)
                    else: # Error case for friction factor
                        st.markdown("Could not calculate friction factor due to invalid inputs.")
                    
                    st.markdown(f"""
                    #### Major Pressure Drop ($\\Delta P_f$) - Darcy-Weisbach Equation
                    $ \\Delta P_f = f \\frac{{L}}{{D}} \\frac{{\\rho V^2}}{{2}} $
                    $ \\Delta P_f = {f:.4f} \\times \\frac{{{L_seg:.1f}}}{{{D_seg:.3f}}} \\times \\frac{{{rho:.2f} \\times ({V_avg:.3f})^2}}{{2}} $
                    $ \\Delta P_f = {delta_P_f_segment:.2f} \\text{{ Pa}} $
                    """)
                    st.success(f"**Segment {segment_num} Pressure Drop: {delta_P_f_segment:.2f} Pa**")
                    st.markdown("---") # Separator for each segment's details

        segment_results_df_display = pd.DataFrame(segment_results_data, columns=[
            'Segment', 'Diameter (m)', 'Length (m)', 'Roughness (m)', 'Velocity (m/s)', 'Reynolds No.', 'Flow Regime', 'Friction Factor', 'Pressure Drop (Pa)'
        ])
        st.dataframe(segment_results_df_display, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Total Pressure Drop Across All Segments")
        st.success(f"**Total Major Pressure Drop (ΔPf,total): {total_delta_P_f:.2f} Pa**")
        st.success(f"**Total Major Pressure Drop (ΔPf,total): {total_delta_P_f/1000:.2f} kPa**")


# --- Graphical Output (Sensitivity Analysis) ---
st.markdown("---")
st.subheader("5. Sensitivity Analysis Plots")
st.write("Explore how changing a single parameter affects pressure drop or friction factor for a representative pipe.")

with st.expander("Configure Sensitivity Analysis Inputs"):
    st.markdown("Enter parameters for the representative pipe used in the plots below:")
    col_plot_in1, col_plot_in2, col_plot_in3 = st.columns(3)
    with col_plot_in1:
        plot_D = st.number_input("Diameter (D) [m]", value=0.1, min_value=0.001, format="%.3f", key="plot_D")
        plot_L = st.number_input("Length (L) [m]", value=100.0, min_value=0.01, format="%.1f", key="plot_L") # Corrected min_value
    with col_plot_in2:
        plot_rho = st.number_input("Fluid Density (ρ) [kg/m³]", value=1000.0, min_value=0.1, format="%.2f", key="plot_rho")
        plot_mu = st.number_input("Fluid Dynamic Viscosity (μ) [Pa·s]", value=0.001, min_value=1e-7, format="%.4e", key="plot_mu")
    with col_plot_in3:
        plot_epsilon = st.number_input("Roughness (ε) [m]", value=0.000045, min_value=0.0, format="%.6e", key="plot_epsilon")
        plot_Q = st.number_input("Flow Rate (Q) [m³/s]", value=0.01, min_value=1e-6, format="%.4e", key="plot_Q")

# --- Plot 1: Pressure drop vs. pipe length ---
st.markdown("#### Pressure Drop vs. Pipe Length")
num_points = 50
lengths_to_plot = np.linspace(10, 500, num_points) # Vary length from 10m to 500m

pd_vs_L = []
for l_val in lengths_to_plot:
    dp, _, _, _, _ = calculate_major_pressure_drop(plot_D, l_val, plot_epsilon, plot_rho, plot_mu, plot_Q)
    pd_vs_L.append(dp)

fig_L = go.Figure(data=go.Scatter(x=lengths_to_plot, y=pd_vs_L, mode='lines+markers'))
fig_L.update_layout(
    title='Pressure Drop vs. Pipe Length',
    xaxis_title='Pipe Length (L) [m]',
    yaxis_title='Pressure Drop (ΔP) [Pa]',
    hovermode="x unified",
    template="plotly_dark" # Use dark theme for plots
)
st.plotly_chart(fig_L, use_container_width=True)

# --- Plot 2: Pressure drop vs. diameter ---
st.markdown("#### Pressure Drop vs. Pipe Diameter")
diameters_to_plot = np.linspace(0.01, 0.2, num_points) # Vary diameter from 1cm to 20cm

pd_vs_D = []
for d_val in diameters_to_plot:
    dp, _, _, _, _ = calculate_major_pressure_drop(d_val, plot_L, plot_epsilon, plot_rho, plot_mu, plot_Q)
    pd_vs_D.append(dp)

fig_D = go.Figure(data=go.Scatter(x=diameters_to_plot, y=pd_vs_D, mode='lines+markers'))
fig_D.update_layout(
    title='Pressure Drop vs. Pipe Diameter',
    xaxis_title='Pipe Diameter (D) [m]',
    yaxis_title='Pressure Drop (ΔP) [Pa]',
    hovermode="x unified",
    template="plotly_dark"
)
st.plotly_chart(fig_D, use_container_width=True)

# --- Plot 3: Friction factor vs. Reynolds number (Simplified Moody Chart) ---
st.markdown("#### Friction Factor vs. Reynolds Number (Simplified Moody Chart)")
reynolds_to_plot = np.logspace(3, 7, num_points) # Re from 10^3 to 10^7 (log scale)

# Define a few relative roughness values for multiple lines
relative_roughness_values = [0.000001, 0.00001, 0.0001, 0.001] # epsilon/D

fig_Re = go.Figure()

for rel_rough in relative_roughness_values:
    friction_factors = []
    for re_val in reynolds_to_plot:
        # Calculate f for a given Re and relative roughness
        if re_val <= 2300:
            f_val = 64 / re_val
        else:
            try:
                # Ensure log argument is positive
                log_arg = rel_rough / 3.7 + 5.74 / (re_val**0.9)
                if log_arg <= 0:
                    f_val = np.nan
                else:
                    f_val = 0.25 / (np.log10(log_arg))**2
            except (ZeroDivisionError, ValueError):
                f_val = np.nan
        friction_factors.append(f_val)
    
    fig_Re.add_trace(go.Scatter(
        x=reynolds_to_plot,
        y=friction_factors,
        mode='lines',
        name=f'$\epsilon/D$ = {rel_rough:.1e}',
        hovertemplate='Re: %{x:.0f}<br>f: %{y:.4f}<extra></extra>'
    ))

# Add a laminar flow line for reference
re_laminar_range = np.linspace(100, 2300, 20)
f_laminar_range = 64 / re_laminar_range
fig_Re.add_trace(go.Scatter(
    x=re_laminar_range,
    y=f_laminar_range,
    mode='lines',
    name='Laminar Flow (f=64/Re)',
    line=dict(dash='dash'),
    hovertemplate='Re: %{x:.0f}<br>f: %{y:.4f}<extra></extra>'
))

fig_Re.update_layout(
    title='Friction Factor vs. Reynolds Number',
    xaxis_title='Reynolds Number (Re)',
    yaxis_title='Friction Factor (f)',
    xaxis_type="log", # Log scale for Re
    yaxis_type="log", # Log scale for f
    hovermode="x unified",
    template="plotly_dark",
    legend_title_text="Relative Roughness ($\epsilon/D$)"
)
st.plotly_chart(fig_Re, use_container_width=True)


# --- Velocity Profile Visualization ---
st.markdown("---")
st.subheader("6. Velocity Profile Visualization")
st.write("Understand the characteristic differences in velocity distribution across the pipe for laminar and turbulent flows.")

st.markdown("""
<p>
    In **laminar flow**, fluid particles move in smooth, parallel layers without significant mixing. The velocity profile is **parabolic**, with zero velocity at the pipe walls and maximum velocity at the center.
</p>
<p>
    In **turbulent flow**, fluid particles move in irregular, chaotic paths, leading to significant mixing. The velocity profile is **flatter** in the central region compared to laminar flow, and drops sharply near the pipe walls due to increased shear stress.
</p>
""", unsafe_allow_html=True)

# Determine the flow regime of the first segment, if available
# This gives context to the velocity profile plot
if edited_df_segments.empty:
    st.info("Calculate pressure drop for segments first to see the flow regime context for the velocity profile.")
    current_flow_regime = "N/A" # Default if no segments
else:
    # Recalculate Re for the first segment to get its flow regime
    D_first = edited_df_segments.loc[0, 'Diameter (m)']
    L_first = edited_df_segments.loc[0, 'Length (m)'] # Not strictly needed for regime
    epsilon_first = edited_df_segments.loc[0, 'Roughness (m)'] # Not strictly needed for regime
    _, re_first, _, regime_first, _ = calculate_major_pressure_drop(D_first, L_first, epsilon_first, rho, mu, Q)
    current_flow_regime = regime_first

st.info(f"The flow regime for the first segment in your table is: **{current_flow_regime}**")


# Generate the matplotlib plot
fig_profile, ax_profile = plt.subplots(figsize=(8, 4)) # Create a figure and an axes object
pipe_radius = 1 # Use a normalized radius for demonstration
r = np.linspace(-pipe_radius, pipe_radius, 100) # Radial distance from center

# Velocity profiles (normalized to max velocity = 1 for comparison)
# Laminar: Parabolic profile
v_laminar = 1 * (1 - (r/pipe_radius)**2)

# Turbulent: 1/7th power law approximation (conceptual)
# Turbulent profile is typically flatter in the center
v_turbulent = 1 * (1 - np.abs(r)/pipe_radius)**(1/7) 

ax_profile.plot(v_laminar, r, label='Laminar Flow (Parabolic)', 
                color='skyblue', linewidth=3, alpha=0.7,
                linestyle='-' if current_flow_regime != "Laminar" else '-') # No specific highlight color here

ax_profile.plot(v_turbulent, r, label='Turbulent Flow (1/7th Power Law)', 
                color='coral', linewidth=3, alpha=0.7,
                linestyle='-' if current_flow_regime != "Turbulent" else '-') # No specific highlight color here


ax_profile.set_title('Typical Velocity Profiles in Pipe Flow')
ax_profile.set_xlabel('Velocity (Normalized)')
ax_profile.set_ylabel('Radial Position (r/R)')
ax_profile.grid(True, linestyle='--', alpha=0.6)
ax_profile.legend()
ax_profile.set_ylim(-pipe_radius, pipe_radius) # Ensure y-axis covers the pipe diameter
ax_profile.set_xlim(0, 1.1) # Velocity from 0 to slightly above 1 for clarity
ax_profile.set_yticks([-1, -0.5, 0, 0.5, 1])
ax_profile.set_yticklabels(['Wall (-R)', '', 'Center (0)', '', 'Wall (+R)'])

# Add visual highlight based on the actual calculated flow regime (optional)
if current_flow_regime == "Laminar":
    ax_profile.lines[0].set_color('blue') # Make laminar line more prominent
    ax_profile.lines[0].set_linewidth(4)
    ax_profile.lines[0].set_alpha(1)
elif current_flow_regime == "Turbulent":
    ax_profile.lines[1].set_color('red') # Make turbulent line more prominent
    ax_profile.lines[1].set_linewidth(4)
    ax_profile.lines[1].set_alpha(1)

plt.tight_layout()
st.pyplot(fig_profile) # Display the plot

# --- Footer ---
st.markdown("---")

# Inject custom CSS for centering text
st.markdown("""
    <style>
    .centered-text {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(f'<div class="centered-text"><p style="color: gray;">Developed by a Mechanical Engineer ⚙️</p></div>', unsafe_allow_html=True)