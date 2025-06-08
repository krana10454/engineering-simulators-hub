import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from numba import njit
import pandas as pd

st.set_page_config(
    page_title="F1 Brake Thermal Analysis",
    page_icon="🏎️",
    layout="wide"
)

# --- Add Home Button at the Top ---
if st.button("🏠 Go to Home"):
    st.switch_page("0_Home.py") # Correctly points to the root Home file

st.title("🏎️ F1 Brake Thermal Analysis")
st.markdown("Simulate the transient temperature distribution in a carbon-carbon brake disc with dynamic heat generation. Explore key performance indicators and how they change with different design and operating parameters.") # Aesthetic: More descriptive intro

st.markdown("---")

# --- 1. Material Properties (Carbon-Carbon Composite - Typical Values) ---
RHO = 1800.0        # Density (kg/m^3)
CP = 700.0          # Specific Heat Capacity (J/kg.K)
K = 30.0            # Thermal Conductivity (W/m.K)
EMISSIVITY = 0.85   # Material emissivity (dimensionless, 0 to 1)
CRITICAL_TEMP_K = 1473.15 # Critical temperature in Kelvin (e.g., 1200 C)

# --- 2. Brake Disc Dimensions (Typical F1 values) ---
OUTER_DIAMETER_M = 278e-3 # Outer Diameter (m)
INNER_DIAMETER_M = 100e-3 # Inner Diameter (m)
DISC_THICKNESS_M = 32e-3  # Disc Thickness (m)

# Derived Radii
R_INNER = INNER_DIAMETER_M / 2
R_OUTER = OUTER_DIAMETER_M / 2

# --- 3. Environmental/Operating Parameters ---
INITIAL_DISC_TEMP_K = 373.15 # Initial Brake Disc Temperature (K) - 100 C
AMBIENT_AIR_TEMP_K = 300.15  # Ambient Air Temperature (K) - 27 C

H_CONV_DISC_FACES = 1000.0  # Convective Heat Transfer Coefficient for disc faces (W/m^2.K)
H_CONV_CYLINDRICAL_SURFACES = 200.0 # Convective Heat Transfer Coefficient for inner/outer cylindrical surfaces
STEFAN_BOLTZMANN_CONST = 5.670374419e-8 # Stefan-Boltzmann constant (W/m^2.K^4)

# --- 4. Braking Event Parameters (Typical F1 scenario) ---
CAR_MASS_KG = 798.0       # Car Mass (kg)
INITIAL_SPEED_MPS = 90.0  # Initial Speed (m/s)
FINAL_SPEED_MPS = 20.0    # Final Speed (m/s)
DECELERATION_MPS2 = 50.0  # Braking Deceleration (m/s^2)
NUM_BRAKE_DISCS = 4       # Number of Brake Discs

# --- 5. Grid and Time Parameters ---
NR = 50              # Number of radial grid points

# Calculate braking time (will be used to determine braking phase)
BRAKING_TIME_S_CALC = (INITIAL_SPEED_MPS - FINAL_SPEED_MPS) / DECELERATION_MPS2
# Calculate total simulation duration (braking + some cooling time)
SIMULATION_DURATION_S_CALC = BRAKING_TIME_S_CALC * 2.0

# --- Heat Generation Ramp-up Parameter ---
RAMP_UP_TIME_S = 0.05 # Time over which heat generation ramps up at the start of braking


# --- Main Simulation Core (Numba-accelerated for 1D Radial Heat Conduction) ---
@njit(cache=True)
def _run_brake_sim_1d_radial_numba_core(rho, cp, k, emissivity, stefan_boltzmann_const,
                                          r_inner, r_outer, disc_thickness,
                                          initial_disc_temp, ambient_air_temp,
                                          h_conv_disc_faces, h_conv_cylindrical_surfaces,
                                          car_mass, deceleration, initial_speed, final_speed, num_brake_discs,
                                          braking_time_s, nr, dr, dt, num_time_steps,
                                          initial_temp_profile, ramp_up_time_s):

    T = initial_temp_profile.copy()
    T_new = T.copy()

    r_coords = np.linspace(r_inner, r_outer, nr)
    alpha = k / (rho * cp)

    # Total effective contact area for heat generation (two sides)
    total_effective_pad_area = 2 * np.pi * (r_outer**2 - r_inner**2)

    # Store KPI data
    max_temps_list = []
    inner_flux_list = []
    outer_flux_list = []
    avg_temps_list = []
    max_gradient_list = [] # NEW: To store max thermal gradient

    # Store full radial profiles for detailed plot (and 3D)
    radial_profiles_list = []
    # Store corresponding time values for radial profiles
    radial_profile_times = []


    # Store initial values for the KPI lists
    max_temps_list.append(np.max(T))
    inner_flux_list.append(0.0)
    outer_flux_list.append(0.0)
    avg_temps_list.append(np.mean(T))
    # Calculate initial gradient (will be 0 if initial temp is uniform)
    if nr > 1:
        initial_gradient = np.max(np.abs(np.diff(T) / np.diff(r_coords)))
        max_gradient_list.append(initial_gradient)
    else:
        max_gradient_list.append(0.0) # For single point, gradient is 0
    radial_profiles_list.append(T.copy()) # Store initial profile
    radial_profile_times.append(0.0)


    for step in range(num_time_steps):
        current_time = (step + 1) * dt # Adjust current_time to start from dt

        is_braking = (current_time > 0 and current_time <= braking_time_s)

        # --- Dynamic Heat Generation from Pads ---
        current_speed = max(final_speed, initial_speed - deceleration * current_time)

        q_gen_from_pads_instantaneous = 0.0
        if is_braking and current_speed > 0:
            instantaneous_power_per_disc = (car_mass * deceleration * current_speed) / num_brake_discs
            q_gen_from_pads_instantaneous = instantaneous_power_per_disc / total_effective_pad_area

            # Apply ramp-up for heat generation
            if current_time < ramp_up_time_s:
                q_gen_from_pads_instantaneous *= (current_time / ramp_up_time_s)


        # --- Calculate `q_dot_from_faces_node` for all nodes ---
        q_dot_from_faces_nodes_array = np.zeros(nr)
        for i in range(nr):
            q_net_face_flux_per_area_node_i = q_gen_from_pads_instantaneous - \
                                             (h_conv_disc_faces * (T[i] - ambient_air_temp) + \
                                              emissivity * stefan_boltzmann_const * (T[i]**4 - ambient_air_temp**4))
            q_dot_from_faces_nodes_array[i] = (2 * q_net_face_flux_per_area_node_i) / disc_thickness


        # --- Internal Nodes (i from 1 to nr-2) ---
        for i in range(1, nr - 1):
            r_j = r_coords[i]

            conduction_term = (T[i+1] - 2*T[i] + T[i-1]) / dr**2 + \
                              (1.0 / r_j) * (T[i+1] - T[i-1]) / (2.0 * dr)

            T_new[i] = T[i] + dt * (alpha * conduction_term + q_dot_from_faces_nodes_array[i] / (rho * cp))


        # --- Boundary Conditions using Ghost Nodes ---

        # Inner Radial Surface (i=0, at r_inner)
        Q_loss_inner_cyl = h_conv_cylindrical_surfaces * (T[0] - ambient_air_temp) + \
                           emissivity * stefan_boltzmann_const * (T[0]**4 - ambient_air_temp**4)

        T_ghost_inner = T[1] + (2 * dr / k) * Q_loss_inner_cyl

        T_new[0] = T[0] + dt * (alpha * (
            (T[1] - 2*T[0] + T_ghost_inner) / dr**2 + \
            (1.0 / r_coords[0]) * (T[1] - T_ghost_inner) / (2.0 * dr)
        ) + q_dot_from_faces_nodes_array[0] / (rho * cp))


        # Outer Radial Surface (i=nr-1, at r_outer)
        Q_loss_outer_cyl = h_conv_cylindrical_surfaces * (T[nr-1] - ambient_air_temp) + \
                           emissivity * stefan_boltzmann_const * (T[nr-1]**4 - ambient_air_temp**4)

        T_ghost_outer = T[nr-2] - (2 * dr / k) * Q_loss_outer_cyl

        T_new[nr-1] = T[nr-1] + dt * (alpha * (
            (T_ghost_outer - 2*T[nr-1] + T[nr-2]) / dr**2 + \
            (1.0 / r_coords[nr-1]) * (T_ghost_outer - T[nr-2]) / (2.0 * dr)
        ) + q_dot_from_faces_nodes_array[nr-1] / (rho * cp))


        # Clipping temperatures
        T_new = np.clip(T_new, ambient_air_temp, CRITICAL_TEMP_K * 1.5)


        T = T_new.copy()

        # Store KPI data at every step
        max_temps_list.append(np.max(T))
        avg_temps_list.append(np.mean(T))
        inner_flux_list.append(Q_loss_inner_cyl)
        outer_flux_list.append(Q_loss_outer_cyl)

        # NEW: Store max thermal gradient
        if nr > 1:
            dT_dr = np.diff(T) / np.diff(r_coords)
            max_gradient_list.append(np.max(np.abs(dT_dr)))
        else:
            max_gradient_list.append(0.0)

        # Store full radial profile for detailed plot and 3D animation
        # Store every ~20th frame or so for detailed plotting/3D to avoid too much data
        # Ensure we always store the last frame
        if (step % (num_time_steps // 20 + 1) == 0) or (step == num_time_steps - 1):
            radial_profiles_list.append(T.copy())
            radial_profile_times.append(current_time)


    return np.array(max_temps_list), np.array(inner_flux_list), np.array(outer_flux_list), \
           np.array(avg_temps_list), np.array(max_gradient_list), \
           radial_profiles_list, radial_profile_times, r_coords


# --- Wrapper for Streamlit UI and Numba Core ---
def run_brake_sim_wrapper(rho, cp, k, emissivity, stefan_boltzmann_const,
                          r_inner, r_outer, disc_thickness,
                          initial_disc_temp, ambient_air_temp,
                          h_conv_disc_faces, h_conv_cylindrical_surfaces,
                          car_mass, deceleration, initial_speed, final_speed, num_brake_discs,
                          simulation_duration_s_input, nr, ramp_up_time_s,
                          display_info=True):

    dr = (r_outer - r_inner) / (nr - 1)

    braking_time_s = (initial_speed - final_speed) / deceleration
    if braking_time_s <= 0:
        if display_info:
            st.error("Braking time cannot be zero or negative. Ensure initial speed > final speed and deceleration > 0.")
        return None, None, None, None, None, None, None, None, None, None, None

    alpha_local = k / (rho * cp)
    dt_max_cfl = dr**2 / (2 * alpha_local)

    dt = dt_max_cfl * 0.95

    num_time_steps_local = int(simulation_duration_s_input / dt)
    if num_time_steps_local == 0:
        if display_info:
            st.error("Number of time steps is zero. Increase simulation duration or decrease DT.")
        return None, None, None, None, None, None, None, None, None, None, None

    dt = simulation_duration_s_input / num_time_steps_local

    if display_info:
        st.info(f"Calculated Braking Time: {braking_time_s:.2f} s")
        st.info(f"Radial Grid Points: {nr}. Radial Spacing: {dr*1e3:.2f} mm.")
        st.info(f"Estimated Max stable DT (CFL): {dt_max_cfl*1e6:.2f} µs.")
        st.info(f"Actual DT used: {dt*1e6:.2f} µs.")
        st.info(f"Total Time Steps: {num_time_steps_local}.")

    initial_temp_profile = np.full(nr, initial_disc_temp)

    if display_info:
        progress_bar = st.progress(0, text="Simulation progress...")
    else:
        progress_bar = type('ProgressBar', (object,), {'progress': lambda self, x, text=None: None})()

    max_temps_list, inner_flux_list, outer_flux_list, avg_temps_list, max_gradient_list, \
           radial_profiles, radial_profile_times, r_coords_returned = \
        _run_brake_sim_1d_radial_numba_core(
            rho, cp, k, emissivity, STEFAN_BOLTZMANN_CONST,
            r_inner, r_outer, disc_thickness,
            initial_disc_temp, ambient_air_temp,
            h_conv_disc_faces, h_conv_cylindrical_surfaces,
            car_mass, deceleration, initial_speed, final_speed, NUM_BRAKE_DISCS,
            braking_time_s,
            nr, dr, dt, num_time_steps_local,
            initial_temp_profile, ramp_up_time_s
        )

    if display_info:
        progress_bar.progress(1.0)
        st.success("Simulation computation complete!")

    # Calculate Max Cooling Rate
    time_points_for_kpis = np.linspace(0, simulation_duration_s_input, len(max_temps_list))
    idx_braking_end = np.argmin(np.abs(time_points_for_kpis - braking_time_s))

    max_cooling_rate = 0.0
    if idx_braking_end < len(max_temps_list) - 1:
        cooling_temps = max_temps_list[idx_braking_end:]
        cooling_times = time_points_for_kpis[idx_braking_end:]

        if len(cooling_temps) > 1:
            temp_diffs = np.diff(cooling_temps)
            time_diffs = np.diff(cooling_times)
            # Find the most negative (steepest drop)
            cooling_rates = temp_diffs / time_diffs
            if len(cooling_rates) > 0:
                max_cooling_rate = np.min(cooling_rates) # max negative value

    return max_temps_list, inner_flux_list, outer_flux_list, avg_temps_list, \
           max_gradient_list, radial_profiles, radial_profile_times, r_coords_returned, \
           braking_time_s, simulation_duration_s_input, max_cooling_rate





# --- Sidebar Inputs ---
st.sidebar.header("Parameters")

with st.sidebar.expander("Material Properties", expanded=True): # Aesthetic: Expander
    rho_st = st.number_input("Density (kg/m³)", value=RHO, min_value=1000.0, max_value=3000.0, step=10.0,
                            help="Density of the brake disc material. Affects thermal inertia (how quickly it heats up).")
    cp_st = st.number_input("Specific Heat (J/kg.K)", value=CP, min_value=500.0, max_value=2000.0, step=10.0,
                            help="Amount of heat required to raise the temperature of a unit mass by one Kelvin. Higher values mean more energy absorption capacity.")
    k_st = st.number_input("Thermal Conductivity (W/m.K)", value=K, min_value=1.0, max_value=100.0, step=0.1,
                            help="Ability of the material to conduct heat. Higher values mean faster heat spread within the disc and less localized hotspots.")
    emissivity_st = st.slider("Emissivity (ε)", value=EMISSIVITY, min_value=0.0, max_value=1.0, step=0.01,
                      help="Measure of a material's ability to emit thermal radiation. A higher emissivity leads to more radiative cooling from the disc surfaces.")
    st.number_input("Critical Temp (K)", value=CRITICAL_TEMP_K, min_value=1000.0, max_value=2000.0, step=10.0, disabled=True,
                    help="The temperature at which the brake disc material's performance may significantly degrade or fail. Highlighted in plots for reference.")

with st.sidebar.expander("Disc Dimensions", expanded=True): # Aesthetic: Expander
    outer_diameter_st = st.number_input("Outer Diameter (mm)", value=OUTER_DIAMETER_M*1e3, min_value=200.0, max_value=400.0, step=1.0,
                                         help="Overall diameter of the brake disc. Larger diameter means more surface area and thermal mass.") / 1e3
    inner_diameter_st = st.number_input("Inner Diameter (mm)", value=INNER_DIAMETER_M*1e3, min_value=50.0, max_value=150.0, step=1.0,
                                         help="Diameter of the central mounting hole in the brake disc. Heat generation does not occur inside this region.") / 1e3
    disc_thickness_st = st.number_input("Disc Thickness (mm)", value=DISC_THICKNESS_M*1e3, min_value=20.0, max_value=50.0, step=0.1,
                                         help="Thickness of the brake disc. Thicker discs have more thermal mass and can absorb more heat.") / 1e3

with st.sidebar.expander("Environmental / Operating Conditions", expanded=True): # Aesthetic: Expander
    initial_disc_temp_st = st.number_input("Initial Disc Temp (K)", value=INITIAL_DISC_TEMP_K, min_value=273.15, max_value=500.0, step=5.0,
                                            help="The temperature of the brake disc at the beginning of the simulation (before braking).")
    ambient_air_temp_st = st.number_input("Ambient Air Temp (K)", value=AMBIENT_AIR_TEMP_K, min_value=273.15, max_value=400.0, step=5.0,
                                           help="Temperature of the surrounding air influencing convective and radiative cooling.")
    h_conv_disc_faces_st = st.number_input("Conv Coeff (Disc Faces) (W/m².K)", value=H_CONV_DISC_FACES, min_value=100.0, max_value=5000.0, step=10.0,
                                            help="Convective heat transfer coefficient for the main faces of the disc (where brake pads apply pressure). Represents cooling due to airflow.")
    h_conv_cylindrical_surfaces_st = st.number_input("Conv Coeff (Cylindrical) (W/m².K)", value=H_CONV_CYLINDRICAL_SURFACES, min_value=10.0, max_value=1000.0, step=10.0,
                                                     help="Convective heat transfer coefficient for the inner and outer cylindrical edges of the disc.")

with st.sidebar.expander("Braking Event Parameters", expanded=True): # Aesthetic: Expander
    car_mass_st = st.number_input("Car Mass (kg)", value=CAR_MASS_KG, min_value=500.0, max_value=1000.0, step=10.0,
                                   help="Total mass of the car. Directly impacts the kinetic energy to be dissipated by the brakes.")
    initial_speed_st = st.number_input("Initial Speed (m/s)", value=INITIAL_SPEED_MPS, min_value=50.0, max_value=100.0, step=1.0,
                                        help="Vehicle speed at the start of the braking event.")
    final_speed_st = st.number_input("Final Speed (m/s)", value=FINAL_SPEED_MPS, min_value=0.0, max_value=50.0, step=1.0,
                                      help="Vehicle speed at the end of the braking event.")
    deceleration_st = st.number_input("Deceleration (m/s²)", value=DECELERATION_MPS2, min_value=10.0, max_value=80.0, step=1.0,
                                       help="The rate at which the car slows down. Higher deceleration means more intense braking and heat generation.")
    ramp_up_time_s_st = st.number_input("Heat Gen Ramp-up Time (s)", value=RAMP_UP_TIME_S, min_value=0.0, max_value=1.0, step=0.01,
                                         help="Time over which heat generation from the brake pads ramps up from zero to full power at the start of braking. Simulates realistic brake engagement rather than an instantaneous step change.")

with st.sidebar.expander("Simulation Grid & Time Settings", expanded=False): # Aesthetic: Expander (start collapsed)
    nr_st = st.number_input("Radial Grid Points (NR)", value=NR, min_value=10, max_value=200, step=5,
                            help="Number of discrete points along the radial direction used for the simulation. More points lead to higher accuracy but longer computation times.")
    simulation_duration_s_st = st.number_input("Simulation Duration (s)", value=SIMULATION_DURATION_S_CALC, min_value=0.1, max_value=20.0, step=0.1,
                                                help="Total time for which the simulation will run, including the braking phase and subsequent cooling.")

# --- Parametric Study Section ---
st.sidebar.markdown("---") # Aesthetic: Separator
st.sidebar.header("🔬 Parametric Study") # Aesthetic: Emoji
param_to_vary = st.sidebar.selectbox(
    "Select Parameter to Vary",
    ["None", "Thermal Conductivity (K)", "Disc Thickness (D)", "Initial Speed (V_initial)",
     "Deceleration (a)", "Convective Coeff (Faces)"],
    help="Choose a parameter to simulate multiple scenarios. For example, vary thermal conductivity to see its impact on temperature profiles."
)

param_values = []
if param_to_vary != "None":
    param_values_str = st.sidebar.text_input(
        f"Enter {param_to_vary} values (comma-separated)",
        "30.0, 40.0, 50.0" if param_to_vary == "Thermal Conductivity (K)" else
        "0.028, 0.032, 0.036" if param_to_vary == "Disc Thickness (D)" else
        "80.0, 90.0, 100.0" if param_to_vary == "Initial Speed (V_initial)" else
        "40.0, 50.0, 60.0" if param_to_vary == "Deceleration (a)" else
        "800.0, 1000.0, 1200.0", # Convective Coeff (Faces)
        help="Provide a comma-separated list of numerical values for the selected parameter. Up to 5 values are recommended for clear plotting."
    )
    try:
        param_values = [float(x.strip()) for x in param_values_str.split(',')]
        if len(param_values) > 5:
            st.sidebar.warning("Too many values for parametric study. Displaying results for the first 5 for clarity.")
            param_values = param_values[:5]
    except ValueError:
        st.sidebar.error("Invalid input for parameter values. Please enter comma-separated numbers (e.g., '30.0, 40.0').")
        param_values = []
else:
    param_values = [0] # Dummy value for single run

# --- Main Simulation Run Button ---
st.sidebar.markdown("---") # Aesthetic: Separator
if st.sidebar.button("🚀 Run Simulation", help="Click to start the thermal simulation based on the configured parameters. Results will appear below."): # Aesthetic: Emoji
    braking_time_for_ui = (initial_speed_st - final_speed_st) / deceleration_st
    if braking_time_for_ui <= 0:
        st.error("Braking time cannot be zero or negative. Ensure initial speed > final speed and deceleration > 0.")
        st.stop()

    # Calculate overall energy dissipated (common for all runs, for display)
    delta_ke_total = 0.5 * car_mass_st * (initial_speed_st**2 - final_speed_st**2)
    heat_per_disc_for_ui = delta_ke_total / NUM_BRAKE_DISCS # This is a new KPI

    # Calculate overall average heat flux (for reference)
    total_face_area_for_heat_input_for_ui = 2 * np.pi * ((outer_diameter_st/2)**2 - (inner_diameter_st/2)**2)
    avg_heat_gen_rate_per_disc_for_ui = heat_per_disc_for_ui / braking_time_for_ui
    avg_heat_flux_from_pads_for_ui = avg_heat_gen_rate_per_disc_for_ui / total_face_area_for_heat_input_for_ui
    
    st.info(f"⚡ Total Kinetic Energy Dissipated per Disc: **{heat_per_disc_for_ui/1000:.1f} kJ**") # Aesthetic: Emoji, New KPI
    st.info(f"🔥 Overall Average Heat Flux from Pads (for reference): **{avg_heat_flux_from_pads_for_ui:.2e} W/m²**") # Aesthetic: Emoji

    all_max_temps_results = {}
    all_inner_flux_results = {}
    all_outer_flux_results = {}
    all_avg_temps_results = {}
    all_max_gradients_results = {} # NEW: Store max gradient results
    all_radial_profiles_results = {}
    all_radial_profile_times = {}
    all_r_coords = {}
    all_max_cooling_rates = {} # NEW: Store max cooling rate results

    run_info = {}

    first_run_successful = False

    if param_to_vary == "None":
        current_param_values_effective = [0]
    else:
        current_param_values_effective = param_values

    for i, p_value in enumerate(current_param_values_effective):
        current_rho = rho_st
        current_cp = cp_st
        current_k = k_st
        current_emissivity = emissivity_st
        current_disc_thickness = disc_thickness_st
        current_initial_disc_temp = initial_disc_temp_st
        current_ambient_air_temp = ambient_air_temp_st
        current_h_conv_disc_faces = h_conv_disc_faces_st
        current_h_conv_cylindrical_surfaces = h_conv_cylindrical_surfaces_st
        current_car_mass = car_mass_st
        current_initial_speed = initial_speed_st
        current_final_speed = final_speed_st
        current_deceleration = deceleration_st
        current_ramp_up_time_s = ramp_up_time_s_st
        current_nr = nr_st
        current_sim_duration = simulation_duration_s_st
        current_r_inner = inner_diameter_st / 2
        current_r_outer = outer_diameter_st / 2

        label = "Default Run" # Aesthetic: More descriptive default label

        if param_to_vary == "Thermal Conductivity (K)":
            current_k = p_value
            label = f"K = {p_value:.1f} W/m.K"
        elif param_to_vary == "Disc Thickness (D)":
            current_disc_thickness = p_value
            label = f"D = {p_value*1e3:.1f} mm"
        elif param_to_vary == "Initial Speed (V_initial)":
            current_initial_speed = p_value
            label = f"V_init = {p_value:.1f} m/s"
        elif param_to_vary == "Deceleration (a)":
            current_deceleration = p_value
            label = f"a = {p_value:.1f} m/s²"
        elif param_to_vary == "Convective Coeff (Faces)":
            current_h_conv_disc_faces = p_value
            label = f"h_conv_faces = {p_value:.1f} W/m².K"
        
        st.markdown(f"--- Running simulation for: **{label}** ---") # Aesthetic: Clearer indication of current run

        results = run_brake_sim_wrapper(
            current_rho, current_cp, current_k, current_emissivity, STEFAN_BOLTZMANN_CONST,
            current_r_inner, current_r_outer, current_disc_thickness,
            current_initial_disc_temp, current_ambient_air_temp,
            current_h_conv_disc_faces, current_h_conv_cylindrical_surfaces,
            current_car_mass, current_deceleration, current_initial_speed, current_final_speed, NUM_BRAKE_DISCS,
            current_sim_duration, current_nr, current_ramp_up_time_s,
            display_info=(param_to_vary == "None" or i == 0) # Only show detailed info for first run or single run
        )

        if results[0] is not None:
            max_temps, inner_fluxes, outer_fluxes, avg_temps, max_gradients, radial_profiles, radial_profile_times, r_coords_returned, braking_t, sim_duration_t, max_cooling_rate = results

            all_max_temps_results[label] = max_temps
            all_inner_flux_results[label] = inner_fluxes
            all_outer_flux_results[label] = outer_fluxes
            all_avg_temps_results[label] = avg_temps
            all_max_gradients_results[label] = max_gradients # Store new KPI
            all_radial_profiles_results[label] = radial_profiles
            all_radial_profile_times[label] = radial_profile_times
            all_r_coords[label] = r_coords_returned
            all_max_cooling_rates[label] = max_cooling_rate # Store new KPI

            run_info[label] = {'braking_time': braking_t, 'sim_duration': sim_duration_t}

            if not first_run_successful:
                first_run_successful = True
        else:
            st.error(f"Simulation failed for {label}. Please check inputs and ensure valid conditions. Skipping this run.")
            if param_to_vary == "None":
                st.stop()

    # --- Store results in session state after all runs for re-rendering ---
    if first_run_successful:
        st.session_state['all_results'] = {
            'max_temps': all_max_temps_results,
            'inner_fluxes': all_inner_flux_results,
            'outer_fluxes': all_outer_flux_results,
            'avg_temps': all_avg_temps_results,
            'max_gradients': all_max_gradients_results, # Store new KPI
            'radial_profiles': all_radial_profiles_results,
            'radial_profile_times': all_radial_profile_times,
            'r_coords': all_r_coords,
            'run_info': run_info,
            'initial_disc_temp': initial_disc_temp_st,
            'critical_temp': CRITICAL_TEMP_K,
            'max_cooling_rates': all_max_cooling_rates # Store new KPI
        }
    else:
        if 'all_results' in st.session_state:
            del st.session_state['all_results'] # Clear previous results if no runs succeed

# --- Display Results Section (conditionally based on session state) ---
if 'all_results' in st.session_state:
    results_data = st.session_state['all_results']
    all_max_temps_results = results_data['max_temps']
    all_inner_flux_results = results_data['inner_fluxes']
    all_outer_flux_results = results_data['outer_fluxes']
    all_avg_temps_results = results_data['avg_temps']
    all_max_gradients_results = results_data['max_gradients'] # Retrieve new KPI
    all_radial_profiles_results = results_data['radial_profiles']
    all_radial_profile_times = results_data['radial_profile_times']
    all_r_coords = results_data['r_coords']
    run_info = results_data['run_info']
    initial_disc_temp_st = results_data['initial_disc_temp']
    CRITICAL_TEMP_K = results_data['critical_temp']
    all_max_cooling_rates = results_data['max_cooling_rates'] # Retrieve new KPI

    st.markdown("---") # Aesthetic: Separator
    st.header("📊 Simulation Results") # Aesthetic: Emoji

    # --- KPI Dashboard ---
    st.subheader("Key Performance Indicators (KPIs)")
    cols = st.columns(3)

    peak_temp_achieved = 0.0
    if all_max_temps_results:
        peak_temp_achieved = max(np.max(data) for data in all_max_temps_results.values())
        first_label = list(all_max_temps_results.keys())[0]
        final_max_temp_of_first_run = all_max_temps_results[first_label][-1]
        max_cooling_rate_display = all_max_cooling_rates[first_label] # Display for first run
        max_gradient_display = max(np.max(data) for data in all_max_gradients_results.values()) # Overall max gradient

    max_temp_diff = peak_temp_achieved - initial_disc_temp_st
    critical_exceeded = peak_temp_achieved >= CRITICAL_TEMP_K
    temp_status = "⚠️ **CRITICAL (Exceeded)**" if critical_exceeded else "✅ **OK**" # Aesthetic: Emoji

    first_run_braking_time = list(run_info.values())[0]['braking_time']
    first_run_sim_duration = list(run_info.values())[0]['sim_duration']

    # Recalculate Energy Dissipated per Disc for display
    # This assumes these parameters are consistent across runs, which they are unless varied
    total_kinetic_energy_dissipated = 0.5 * car_mass_st * (initial_speed_st**2 - final_speed_st**2)
    energy_per_disc = total_kinetic_energy_dissipated / NUM_BRAKE_DISCS

    with cols[0]:
        st.metric("Peak Disc Temperature", f"{peak_temp_achieved:.1f} K",
                  delta_color="inverse" if critical_exceeded else "off",
                  help="The highest temperature reached anywhere on the disc throughout all simulated runs. Exceeding the critical temperature (red dashed line in plots) can lead to material degradation.")
        st.markdown(f"Status: {temp_status}")

    with cols[1]:
        st.metric("Max Temp Rise", f"{max_temp_diff:.1f} K",
                  help="The total increase in temperature from the initial disc temperature to the peak temperature achieved.")
        st.metric("Energy Dissipated per Disc", f"{energy_per_disc/1000:.1f} kJ", # New KPI
                  help="The total kinetic energy converted into heat by a single brake disc during the braking event.")

    with cols[2]:
        st.metric("Max Cooling Rate (1st Run)", f"{abs(max_cooling_rate_display):.1f} K/s", # New KPI
                  help="The maximum rate at which the disc temperature decreased after the braking event for the first simulated run. A higher value indicates faster cooling.")
        st.metric("Peak Radial Thermal Gradient", f"{max_gradient_display:.1f} K/m", # New KPI
                  help="The maximum temperature difference per unit radial distance observed in the disc. High thermal gradients can lead to thermal stresses and potential cracking.")


    st.markdown("---") # Aesthetic: Separator

    # --- Plot 1: Maximum Temperature Over Time ---
    st.subheader("📈 Maximum Temperature Over Time") # Aesthetic: Emoji
    fig_max_temp, ax_max_temp = plt.subplots(figsize=(10, 4))

    for label, max_temps_data in all_max_temps_results.items():
        current_sim_duration = run_info[label]['sim_duration']
        time_points_for_this_run = np.linspace(0, current_sim_duration, len(max_temps_data))
        ax_max_temp.plot(time_points_for_this_run, max_temps_data, label=label)

    ax_max_temp.axhline(y=CRITICAL_TEMP_K, color='red', linestyle='--', label='Critical Temp')
    ax_max_temp.axvline(x=first_run_braking_time, color='blue', linestyle=':', label='Braking Ends (First Run)')
    ax_max_temp.axvspan(0, first_run_braking_time, color='lightgray', alpha=0.3, label='Braking Phase (First Run)')

    critical_text_added = False
    for label, max_temps_data in all_max_temps_results.items():
        current_sim_duration = run_info[label]['sim_duration']
        time_points_for_this_run = np.linspace(0, current_sim_duration, len(max_temps_data))
        if np.any(max_temps_data >= CRITICAL_TEMP_K):
            first_exceed_idx = np.argmax(max_temps_data >= CRITICAL_TEMP_K)
            # Ensure index is within bounds before accessing
            if first_exceed_idx < len(time_points_for_this_run):
                first_exceed_time = time_points_for_this_run[first_exceed_idx]
                ax_max_temp.axvspan(first_exceed_time, current_sim_duration, color='red', alpha=0.05)
                if not critical_text_added:
                    # Aesthetic: Adjust text position for better visibility
                    text_x = first_exceed_time + (current_sim_duration - first_exceed_time)*0.05
                    text_y = CRITICAL_TEMP_K + (ax_max_temp.get_ylim()[1] - CRITICAL_TEMP_K) * 0.05 # Place relative to top of plot
                    ax_max_temp.text(text_x, text_y, 'Critical Temp Exceeded', color='red', va='bottom', ha='left', fontsize=9,
                                     bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='red', lw=0.5, alpha=0.8)) # Aesthetic: Bounding box
                    critical_text_added = True

    ax_max_temp.set_xlabel("Time (s)")
    ax_max_temp.set_ylabel("Max Temperature (K)")
    ax_max_temp.set_title("Maximum Disc Temperature Over Time")
    ax_max_temp.legend()
    ax_max_temp.grid(True, linestyle='--', alpha=0.7)
    fig_max_temp.tight_layout() # Aesthetic: Prevent overlap
    st.pyplot(fig_max_temp)

    # --- Plot 2: Heat Fluxes at Boundaries Over Time ---
    st.subheader("♨️ Heat Flux at Boundaries Over Time") # Aesthetic: Emoji
    fig_flux, ax_flux = plt.subplots(figsize=(10, 4))

    for label, inner_flux_data in all_inner_flux_results.items():
        current_sim_duration = run_info[label]['sim_duration']
        time_points_for_this_run = np.linspace(0, current_sim_duration, len(inner_flux_data))
        ax_flux.plot(time_points_for_this_run, inner_flux_data, label=f"Inner ({label})")

    for label, outer_flux_data in all_outer_flux_results.items():
        current_sim_duration = run_info[label]['sim_duration']
        time_points_for_this_run = np.linspace(0, current_sim_duration, len(outer_flux_data))
        ax_flux.plot(time_points_for_this_run, outer_flux_data, label=f"Outer ({label})")

    ax_flux.axvline(x=first_run_braking_time, color='blue', linestyle=':', label='Braking Ends (First Run)')
    ax_flux.axvspan(0, first_run_braking_time, color='lightgray', alpha=0.3, label='Braking Phase (First Run)')

    ax_flux.set_xlabel("Time (s)")
    ax_flux.set_ylabel("Heat Flux (W/m²)")
    ax_flux.set_title("Heat Flux at Inner and Outer Cylindrical Surfaces")
    ax_flux.legend()
    ax_flux.grid(True, linestyle='--', alpha=0.7)
    fig_flux.tight_layout() # Aesthetic: Prevent overlap
    st.pyplot(fig_flux)

    # --- Plot 3: Average Temperature Over Time ---
    st.subheader("🌡️ Average Temperature Over Time") # Aesthetic: Emoji
    fig_avg_temp, ax_avg_temp = plt.subplots(figsize=(10, 4))

    for label, avg_temps_data in all_avg_temps_results.items():
        current_sim_duration = run_info[label]['sim_duration']
        time_points_for_this_run = np.linspace(0, current_sim_duration, len(avg_temps_data))
        ax_avg_temp.plot(time_points_for_this_run, avg_temps_data, label=label)

    ax_avg_temp.axhline(y=CRITICAL_TEMP_K, color='red', linestyle='--', label='Critical Temp (Max Avg)')
    ax_avg_temp.axvline(x=first_run_braking_time, color='blue', linestyle=':', label='Braking Ends (First Run)')
    ax_avg_temp.axvspan(0, first_run_braking_time, color='lightgray', alpha=0.3, label='Braking Phase (First Run)')

    ax_avg_temp.set_xlabel("Time (s)")
    ax_avg_temp.set_ylabel("Average Temperature (K)")
    ax_avg_temp.set_title("Average Disc Temperature Over Time")
    ax_avg_temp.legend()
    ax_avg_temp.grid(True, linestyle='--', alpha=0.7)
    fig_avg_temp.tight_layout() # Aesthetic: Prevent overlap
    st.pyplot(fig_avg_temp)


    # --- Detailed Radial Profile Plots at Key Time Instances ---
    st.subheader("📏 Detailed Radial Temperature Profiles") # Aesthetic: Emoji

    if param_to_vary != "None" and len(all_radial_profiles_results) > 1:
        selected_run_label_2d = st.selectbox(
            "Select Parametric Run for Detailed 2D Profiles",
            list(all_radial_profiles_results.keys()),
            key="2d_radial_profile_selector",
            help="Choose a specific parametric run to view its detailed radial temperature profiles at different key time instances."
        )
    else:
        selected_run_label_2d = list(all_radial_profiles_results.keys())[0]

    radial_profiles_to_plot_2d = all_radial_profiles_results[selected_run_label_2d]
    r_coords_to_plot_2d = all_r_coords[selected_run_label_2d]
    sim_duration_selected_2d = run_info[selected_run_label_2d]['sim_duration']
    braking_time_selected_2d = run_info[selected_run_label_2d]['braking_time']
    max_temps_selected_2d = all_max_temps_results[selected_run_label_2d]
    radial_profile_times_2d = all_radial_profile_times[selected_run_label_2d]

    # Identify key time indices/values
    idx_t0_2d = 0
    idx_braking_end_2d = np.argmin(np.abs(np.array(radial_profile_times_2d) - braking_time_selected_2d))
    
    # Time of peak temperature (for this specific run from KPI data)
    time_points_full_res_kpi = np.linspace(0, sim_duration_selected_2d, len(max_temps_selected_2d))
    idx_peak_temp_kpi_2d = np.argmax(max_temps_selected_2d)
    time_at_peak_temp_kpi_2d = time_points_full_res_kpi[idx_peak_temp_kpi_2d]

    # Find the closest stored radial profile frame to this peak temp time
    idx_radial_profile_at_peak_temp_2d = np.argmin(np.abs(np.array(radial_profile_times_2d) - time_at_peak_temp_kpi_2d))

    idx_sim_end_2d = len(radial_profiles_to_plot_2d) - 1

    key_indices_2d = sorted(list(set([idx_t0_2d, idx_braking_end_2d, idx_radial_profile_at_peak_temp_2d, idx_sim_end_2d])))

    fig_radial, ax_radial = plt.subplots(figsize=(10, 5))

    temp_min_radial_plot = min(np.min(radial_profiles_to_plot_2d), initial_disc_temp_st) - 10
    temp_max_radial_plot = max(np.max(radial_profiles_to_plot_2d), CRITICAL_TEMP_K) + 50

    for idx in key_indices_2d:
        time_val = radial_profile_times_2d[idx]
        profile = radial_profiles_to_plot_2d[idx]
        ax_radial.plot(r_coords_to_plot_2d * 1e3, profile, label=f'Time: {time_val:.2f} s')

    ax_radial.axhline(y=CRITICAL_TEMP_K, color='red', linestyle='--', label='Critical Temp')
    ax_radial.set_xlabel("Radial Position (mm)")
    ax_radial.set_ylabel("Temperature (K)")
    ax_radial.set_ylim(temp_min_radial_plot, temp_max_radial_plot)
    ax_radial.set_title(f"Radial Temperature Profiles for: {selected_run_label_2d}")
    ax_radial.legend()
    ax_radial.grid(True, linestyle='--', alpha=0.7)
    fig_radial.tight_layout() # Aesthetic: Prevent overlap
    st.pyplot(fig_radial)


    # --- Downloadable Results ---
    st.markdown("---") # Aesthetic: Separator
    st.subheader("📥 Download Simulation Data") # Aesthetic: Emoji

    all_data_for_download = []
    for label, max_temps_data in all_max_temps_results.items():
        current_sim_duration = run_info[label]['sim_duration']
        time_points_for_this_run = np.linspace(0, current_sim_duration, len(max_temps_data))

        df_data = {
            'Time (s)': time_points_for_this_run,
            'Max Temp (K)': max_temps_data,
            'Avg Temp (K)': all_avg_temps_results[label],
            'Inner Flux (W/m^2)': all_inner_flux_results[label],
            'Outer Flux (W/m^2)': all_outer_flux_results[label],
            'Max Radial Gradient (K/m)': all_max_gradients_results[label] # NEW KPI
        }

        df = pd.DataFrame(df_data)
        df['Parametric Run'] = label
        all_data_for_download.append(df)

    if all_data_for_download:
        combined_df = pd.concat(all_data_for_download, ignore_index=True)
        
        # Aesthetic: Display a preview of the data
        st.write("Preview of combined data:")
        st.dataframe(combined_df.head()) # Show first few rows

        csv_data = combined_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download All Simulation Data as CSV",
            data=csv_data,
            file_name="brake_thermal_data.csv",
            mime="text/csv",
            help="Download a CSV file containing time, max temperature, average temperature, heat fluxes, and max radial gradient for all simulated runs."
        )
    else:
        st.info("No data available to download yet. Run a simulation first!")


else:
    st.info("💡 Get started by adjusting the parameters in the sidebar and clicking 'Run Simulation' to see the results!") # Aesthetic: More welcoming message
