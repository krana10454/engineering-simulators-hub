import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Smart HVAC Optimizer",
    page_icon="💡",
    layout="wide"
)

# --- Add Home Button at the Top ---
if st.button("🏠 Go to Home"):
    st.switch_page("0_Home.py") # Correctly points to the root Home file

st.title("💡 Smart HVAC Optimizer")
st.markdown("""
    This simulator helps optimize HVAC energy usage by modeling heat transfer,
    occupancy, insulation, and control strategies, now including humidity effects and PID control.
""")
st.markdown("---")

st.subheader("1. System Parameters & Environmental Conditions ")
st.info("💡 Get started by adjusting the parameters in the sidebar ") # Aesthetic: More welcoming message


# --- User Inputs (Streamlit Sidebar) ---
st.sidebar.header("Room & HVAC Parameters")

# Room Dimensions
st.sidebar.subheader("Room Dimensions")
room_length = st.sidebar.slider("Room Length (m)", 3.0, 15.0, 5.0, 0.1)
room_width = st.sidebar.slider("Room Width (m)", 3.0, 15.0, 4.0, 0.1)
room_height = st.sidebar.slider("Room Height (m)", 2.5, 5.0, 3.0, 0.1)
room_volume_m3 = room_length * room_width * room_height

# Thermal Insulation Quality (U-values)
st.sidebar.subheader("Thermal Envelope")
insulation_quality = st.sidebar.selectbox("Insulation Quality", ["Excellent", "Good", "Average", "Poor"])

# Define U-values (W/m²·K) based on quality
U_values = {
    "Excellent": {"walls": 0.2, "windows": 1.0, "roof": 0.2},
    "Good": {"walls": 0.3, "windows": 1.5, "roof": 0.3},
    "Average": {"walls": 0.5, "windows": 2.5, "roof": 0.4},
    "Poor": {"walls": 0.8, "windows": 3.5, "roof": 0.5},
}
U_wall = U_values[insulation_quality]["walls"]
U_window = U_values[insulation_quality]["windows"]
U_roof = U_values[insulation_quality]["roof"]

# Area calculations
wall_area = 2 * (room_length * room_height + room_width * room_height)
window_area_ratio = st.sidebar.slider("Window Area Ratio (of total wall area)", 0.05, 0.50, 0.20, 0.01)
window_area = wall_area * window_area_ratio
effective_wall_area = wall_area - window_area # Wall area excluding windows
roof_area = room_length * room_width

# Thermal Mass of Room Structure (Sensible Heat)
st.sidebar.markdown("**Room Thermal Mass (Structure & Contents - Sensible)**")
thermal_mass_per_m3_J_K = st.sidebar.slider(
    "Effective Sensible Thermal Mass per m³ (J/m³·K)",
    10000.0, 500000.0, 250000.0, 10000.0,
    format="%.0f", help="Heat capacity of room walls, furniture etc. Higher value means slower temperature changes."
)
room_structure_sensible_thermal_mass_J_K = thermal_mass_per_m3_J_K * room_volume_m3

# Moisture Buffering Capacity (Latent Heat)
st.sidebar.markdown("**Room Moisture Buffering Capacity (Latent Heat)**")
moisture_buffering_capacity_per_m3_kg_water_per_rh_percent = st.sidebar.slider(
    "Moisture Buffering Capacity (kg_water / (m³ room · %RH))",
    0.001, 0.1, 0.01, 0.001,
    format="%.3f", help="Ability of room materials (walls, furniture) to absorb/release moisture. Higher value means smoother RH changes."
)
room_total_moisture_buffering_capacity_kg_per_rh_percent = moisture_buffering_capacity_per_m3_kg_water_per_rh_percent * room_volume_m3


# Occupancy & Internal Loads
st.sidebar.subheader("Internal Loads")
num_occupants = st.sidebar.slider("Number of Occupants", 0, 10, 2)
heat_per_person_sensible_W = st.sidebar.number_input("Sensible Heat per Person (W)", value=75.0, min_value=30.0, max_value=120.0, step=5.0)
heat_per_person_latent_W = st.sidebar.number_input("Latent Heat per Person (W)", value=25.0, min_value=10.0, max_value=80.0, step=5.0)
lighting_power_W = st.sidebar.number_input("Lighting Power (W)", value=100.0, min_value=0.0, max_value=500.0, step=10.0)
equipment_power_W = st.sidebar.number_input("Equipment Power (W)", value=150.0, min_value=0.0, max_value=1000.0, step=10.0)


# HVAC System Parameters
st.sidebar.subheader("HVAC System")
hvac_cop = st.sidebar.slider("HVAC COP (Coefficient of Performance)", 2.0, 6.0, 3.5, 0.1)
hvac_max_cooling_capacity_kW = st.sidebar.number_input("Max Cooling Capacity (kW)", value=15.0, min_value=1.0, max_value=50.0, step=0.5, help="Total cooling capacity (sensible + latent)")
hvac_shr = st.sidebar.slider("HVAC Sensible Heat Ratio (SHR)", 0.6, 0.9, 0.65, 0.01, help="Fraction of HVAC cooling capacity that is sensible. Lower values mean more moisture removal (latent cooling).", key="hvac_shr_slider")

# NEW: PID Control Parameters
st.sidebar.subheader("PID Controller Tuning")
st.sidebar.markdown("_Tune these carefully to optimize temperature control._")
kp = st.sidebar.number_input("Proportional Gain (Kp)", value=5.0, min_value=0.0, max_value=10.0, step=0.01, format="%.2f", help="Controls immediate response to temperature error. Higher Kp = faster cooling, but can cause oscillations. If temp isn't dropping fast enough, increase Kp.")
ki = st.sidebar.number_input("Integral Gain (Ki)", value=0.1, min_value=0.0, max_value=0.5, step=0.001, format="%.3f", help="Eliminates steady-state error (temperature offset). Higher Ki = more precise setpoint adherence, but can cause overshoot. If temp stabilizes above setpoint, increase Ki.")
kd = st.sidebar.number_input("Derivative Gain (Kd)", value=0.2, min_value=0.0, max_value=5.0, step=0.1, format="%.1f", help="Dampens oscillations and reacts to rate of change. Higher Kd = smoother approach, but can make control noisy. Set to 0.0 initially for basic tuning.")

# Simulation Parameters
st.sidebar.subheader("Simulation Settings")
simulation_duration_hours = st.sidebar.slider("Simulation Duration (Hours)", 1, 24, 8)
time_step_minutes = st.sidebar.slider("Time Step (Minutes)", 1, 60, 5, 1)

# --- Environmental Conditions ---
st.sidebar.subheader("Environmental Conditions")
outdoor_temp_c = st.sidebar.slider("Outdoor Temperature (°C)", -10.0, 50.0, 35.0, 0.5)
outdoor_humidity_rh = st.sidebar.slider("Outdoor Relative Humidity (%)", 0, 100, 70)
indoor_setpoint_temp_c = st.sidebar.slider("Desired Indoor Temperature (°C)", 18.0, 30.0, 24.0, 0.5)
indoor_setpoint_rh = st.sidebar.slider("Desired Indoor Relative Humidity (%)", 30, 80, 50)
initial_indoor_temp_c = st.sidebar.number_input("Initial Indoor Temperature (°C)", value=30.0, min_value=-10.0, max_value=50.0, step=0.5)
initial_indoor_rh = st.sidebar.number_input("Initial Indoor Relative Humidity (%)", value=65, min_value=0, max_value=100)


# --- Constants for Air Properties ---
AIR_DENSITY_KG_M3 = 1.2         # kg/m^3 (approx. at std conditions)
SPECIFIC_HEAT_AIR_J_KGK = 1000  # J/kg·K (approx. for dry air, sensible only)
# Using 2450 kJ/kg at 25C dry bulb as a common approx for latent heat removal
LATENT_HEAT_VAPORIZATION_J_KG_APPROX = 2450 * 1000 # J/kg

# --- Psychrometric Functions ---
def get_saturated_vapor_pressure_kpa(T_celsius):
    """Calculates saturation vapor pressure in kPa given temperature in Celsius."""
    return 0.61094 * np.exp((17.625 * T_celsius) / (T_celsius + 243.04))

def get_humidity_ratio(T_celsius, RH_percent, P_atm_kpa=101.325):
    """Calculates humidity ratio (kg_water/kg_dry_air)"""
    P_ws = get_saturated_vapor_pressure_kpa(T_celsius)
    P_w = P_ws * (RH_percent / 100.0) # Actual vapor pressure
    return 0.62198 * P_w / (P_atm_kpa - P_w)

def get_relative_humidity(T_celsius, W_kg_kg, P_atm_kpa=101.325):
    """Calculates relative humidity (%) given temperature and humidity ratio."""
    P_ws = get_saturated_vapor_pressure_kpa(T_celsius)
    P_w = P_atm_kpa * W_kg_kg / (0.62198 + W_kg_kg)
    
    if P_ws > 0:
        rh = (P_w / P_ws) * 100.0
    else:
        rh = 0
    return max(0, min(100, rh))


# --- Ventilation Air Flow ---
air_changes_per_hour = st.sidebar.slider("Air Changes per Hour (ACH)", 0.5, 2.0, 0.8, 0.1)
ventilation_flow_m3_hr = air_changes_per_hour * room_volume_m3
ventilation_flow_m3_s = ventilation_flow_m3_hr / 3600
mass_flow_ventilation_kg_s = ventilation_flow_m3_s * AIR_DENSITY_KG_M3


st.markdown("---")
st.subheader("2. Thermal Load Calculation (Snapshot)")

# --- Thermal Model: Heat Gain Sources (Snapshot at Initial/Setpoint) ---
snapshot_delta_T = outdoor_temp_c - indoor_setpoint_temp_c

if snapshot_delta_T > 0:
    q_cond_walls_W = U_wall * effective_wall_area * snapshot_delta_T
    q_cond_windows_W = U_window * window_area * snapshot_delta_T
    q_cond_roof_W = U_roof * roof_area * snapshot_delta_T
else:
    q_cond_walls_W = 0
    q_cond_windows_W = 0
    q_cond_roof_W = 0

total_q_cond_W = q_cond_walls_W + q_cond_windows_W + q_cond_roof_W

q_people_sensible_W = num_occupants * heat_per_person_sensible_W
q_people_latent_W = num_occupants * heat_per_person_latent_W
q_lighting_W = lighting_power_W
q_equipment_W = equipment_power_W
total_q_internal_sensible_W = q_people_sensible_W + q_lighting_W + q_equipment_W
total_q_internal_latent_W = q_people_latent_W

W_outdoor = get_humidity_ratio(outdoor_temp_c, outdoor_humidity_rh)
W_indoor_setpoint = get_humidity_ratio(indoor_setpoint_temp_c, indoor_setpoint_rh)

q_ventilation_sensible_W = 0
q_ventilation_latent_W = 0

if outdoor_temp_c > indoor_setpoint_temp_c:
    q_ventilation_sensible_W = mass_flow_ventilation_kg_s * SPECIFIC_HEAT_AIR_J_KGK * (outdoor_temp_c - indoor_setpoint_temp_c)

if W_outdoor > W_indoor_setpoint:
    q_ventilation_latent_W = mass_flow_ventilation_kg_s * LATENT_HEAT_VAPORIZATION_J_KG_APPROX * (W_outdoor - W_indoor_setpoint)

total_sensible_heat_gain_snapshot_W = total_q_cond_W + total_q_internal_sensible_W + q_ventilation_sensible_W
total_latent_heat_gain_snapshot_W = total_q_internal_latent_W + q_ventilation_latent_W
total_heat_gain_snapshot_W = total_sensible_heat_gain_snapshot_W + total_latent_heat_gain_snapshot_W
total_heat_gain_snapshot_kW = total_heat_gain_snapshot_W / 1000.0


# Display results
st.markdown(f"**Room Volume:** `{room_volume_m3:.2f} m³`")
st.markdown(f"**Total Wall Area:** `{wall_area:.2f} m²` (Effective: `{effective_wall_area:.2f} m²`)")
st.markdown(f"**Window Area:** `{window_area:.2f} m²`")
st.markdown(f"**Roof Area:** `{roof_area:.2f} m²`")
st.markdown(f"**Room Sensible Thermal Mass:** `{room_structure_sensible_thermal_mass_J_K/1000:.0f} kJ/K`")
st.markdown(f"**Room Moisture Buffering Capacity:** `{room_total_moisture_buffering_capacity_kg_per_rh_percent:.3f} kg_water/%RH`")


st.markdown("### Calculated Heat Gains (at current settings, assuming `Indoor T = Setpoint T, Indoor RH = Setpoint RH`):")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Conduction (Walls)", f"{q_cond_walls_W:.2f} W")
    st.metric("Conduction (Windows)", f"{q_cond_windows_W:.2f} W")
    st.metric("Conduction (Roof)", f"{q_cond_roof_W:.2f} W")
    st.metric("Total Conduction Gain", f"{total_q_cond_W:.2f} W")
with col2:
    st.metric("Occupant Sensible Gain", f"{q_people_sensible_W:.2f} W")
    st.metric("Occupant Latent Gain", f"{q_people_latent_W:.2f} W")
    st.metric("Lighting Gain", f"{q_lighting_W:.2f} W")
    st.metric("Equipment Gain", f"{q_equipment_W:.2f} W")
    st.metric("Total Internal Sensible Gain", f"{total_q_internal_sensible_W:.2f} W")
    st.metric("Total Internal Latent Gain", f"{total_q_internal_latent_W:.2f} W")
with col3:
    st.metric("Ventilation Sensible Gain", f"{q_ventilation_sensible_W:.2f} W")
    st.metric("Ventilation Latent Gain", f"{q_ventilation_latent_W:.2f} W")
    st.metric("Total Sensible Heat Gain (Snapshot)", f"{total_sensible_heat_gain_snapshot_W/1000:.2f} kW")
    st.metric("Total Latent Heat Gain (Snapshot)", f"{total_latent_heat_gain_snapshot_W/1000:.2f} kW")
    st.metric("Total Heat Gain (Snapshot)", f"**{total_heat_gain_snapshot_kW:.2f} kW**")

st.markdown("""
    *Note: This is a snapshot calculation. The full simulation below will account for heat dynamics over time.*
""")

st.markdown("---")
st.subheader("3. Dynamic Simulation & HVAC Control")

# --- PID Controller Class ---
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, output_min, output_max, integral_max=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max if integral_max is not None else (output_max - output_min) # Prevents excessive integral windup

        self.previous_error = 0
        self.integral = 0
        self.last_output = 0 # To prevent windup if output is saturated

    def update(self, current_value, dt):
        error = current_value - self.setpoint # Changed for cooling control: Positive error means room is too hot

        # Proportional term
        P_term = self.Kp * error

        # Integral term
        self.integral += error * dt
        
        # Integral windup prevention: only accumulate integral if not saturated
        # And cap the integral term
        if self.integral_max is not None:
             self.integral = max(-self.integral_max, min(self.integral_max, self.integral))


        I_term = self.Ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / dt
        D_term = self.Kd * derivative

        # Calculate total output
        output = P_term + I_term + D_term

        # Apply output limits
        output = max(self.output_min, min(self.output_max, output))

        # Anti-windup for integral term if output is clamped
        if output > self.output_max and error > 0: # If output is saturated at max AND still too hot
            self.integral -= error * dt # Prevent integral from building up further
        elif output < self.output_min and error < 0: # If output is saturated at min (0) AND too cold
            self.integral -= error * dt # Prevent integral from building up further (though less relevant for min=0)

        # Ensure integral term doesn't grow indefinitely in the wrong direction
        # if output is clamped at 0.
        # This helps if the temperature is below setpoint and integral is negative.
        if output == self.output_min and error < 0 and self.integral < 0:
            self.integral = 0
            
        self.previous_error = error
        self.last_output = output
        
        return output

# --- Simulation Function ---
def simulate_hvac_system(
    room_volume_m3, effective_wall_area, window_area, roof_area,
    U_wall, U_window, U_roof,
    room_structure_sensible_thermal_mass_J_K,
    room_total_moisture_buffering_capacity_kg_per_rh_percent,
    num_occupants, heat_per_person_sensible_W, heat_per_person_latent_W, lighting_power_W, equipment_power_W,
    air_changes_per_hour, AIR_DENSITY_KG_M3, SPECIFIC_HEAT_AIR_J_KGK, LATENT_HEAT_VAPORIZATION_J_KG_APPROX,
    outdoor_temp_c, outdoor_humidity_rh,
    indoor_setpoint_temp_c, indoor_setpoint_rh, initial_indoor_temp_c, initial_indoor_rh,
    hvac_cop, hvac_max_cooling_capacity_kW, hvac_shr,
    kp, ki, kd, # NEW PID parameters
    simulation_duration_hours, time_step_minutes
):
    
    # Pre-calculate constants
    time_step_seconds = time_step_minutes * 60
    total_simulation_steps = int(simulation_duration_hours * 60 / time_step_minutes)
    
    room_air_sensible_thermal_capacity_J_K = room_volume_m3 * AIR_DENSITY_KG_M3 * SPECIFIC_HEAT_AIR_J_KGK
    total_room_sensible_thermal_capacity_J_K = room_air_sensible_thermal_capacity_J_K + room_structure_sensible_thermal_mass_J_K
    room_air_mass_dry_kg = room_volume_m3 * AIR_DENSITY_KG_M3

    # Initialize PID controller for temperature
    # Output is desired cooling power in kW, between 0 and hvac_max_cooling_capacity_kW
    pid_controller = PIDController(
        Kp=kp, Ki=ki, Kd=kd,
        setpoint=indoor_setpoint_temp_c,
        output_min=0,
        output_max=hvac_max_cooling_capacity_kW,
        integral_max=hvac_max_cooling_capacity_kW * 2 # Heuristic limit for integral windup
    )


    # Lists to store simulation data
    time_points = []
    room_temperatures = []
    room_humidity_ratios = []
    room_relative_humidities = []
    hvac_cooling_load_kW_data = [] # Actual cooling provided
    hvac_power_consumption_kW_data = []
    total_heat_gain_data = []

    current_indoor_temp_c = initial_indoor_temp_c
    current_indoor_rh = initial_indoor_rh
    current_indoor_W = get_humidity_ratio(current_indoor_temp_c, current_indoor_rh)

    total_energy_consumed_kWh = 0

    start_time = datetime.now()

    for step in range(total_simulation_steps):
        time_points.append(start_time + timedelta(minutes=step * time_step_minutes))
        room_temperatures.append(current_indoor_temp_c)
        room_humidity_ratios.append(current_indoor_W)
        room_relative_humidities.append(current_indoor_rh)

        # --- 1. Calculate instantaneous heat and moisture gains (dynamic) ---
        
        # Conduction Heat Gain (Sensible)
        delta_T_cond = outdoor_temp_c - current_indoor_temp_c
        q_cond_sensible_W = (U_wall * effective_wall_area + U_window * window_area + U_roof * roof_area) * delta_T_cond

        # Internal Heat Gains (Sensible & Latent)
        q_internal_sensible_W = (num_occupants * heat_per_person_sensible_W) + lighting_power_W + equipment_power_W
        q_internal_latent_W = (num_occupants * heat_per_person_latent_W)

        # Ventilation Heat/Moisture Gain (Sensible & Latent)
        mass_flow_ventilation_kg_s = (air_changes_per_hour * room_volume_m3) / 3600 * AIR_DENSITY_KG_M3
        W_outdoor = get_humidity_ratio(outdoor_temp_c, outdoor_humidity_rh)

        q_ventilation_sensible_W = mass_flow_ventilation_kg_s * SPECIFIC_HEAT_AIR_J_KGK * (outdoor_temp_c - current_indoor_temp_c)
        q_ventilation_latent_W = mass_flow_ventilation_kg_s * LATENT_HEAT_VAPORIZATION_J_KG_APPROX * (W_outdoor - current_indoor_W)

        # Total Net Heat Gains into the room (W)
        net_sensible_gain_W = q_cond_sensible_W + q_internal_sensible_W + q_ventilation_sensible_W
        net_latent_gain_W = q_internal_latent_W + q_ventilation_latent_W
        total_net_heat_gain_W_per_sec = net_sensible_gain_W + net_latent_gain_W
        total_heat_gain_data.append(total_net_heat_gain_W_per_sec / 1000.0)


        # --- 2. PID Control for HVAC Cooling Capacity ---
        # PID controller calculates the *desired total cooling capacity* in kW based on temperature error
        desired_cooling_capacity_kW = pid_controller.update(current_indoor_temp_c, time_step_seconds)
        
        # Ensure desired capacity is not negative (no heating in this model yet)
        if desired_cooling_capacity_kW < 0:
            desired_cooling_capacity_kW = 0

        hvac_cooling_capacity_total_W = desired_cooling_capacity_kW * 1000 # Convert to Watts

        # Distribute HVAC total cooling into sensible and latent components based on SHR
        hvac_sensible_cooling_W = hvac_cooling_capacity_total_W * hvac_shr
        hvac_latent_cooling_W = hvac_cooling_capacity_total_W * (1 - hvac_shr)

        # Calculate electrical power consumption
        hvac_electrical_power_W = 0
        if hvac_cooling_capacity_total_W > 0: # Only consume power if cooling is requested
            hvac_electrical_power_W = hvac_cooling_capacity_total_W / hvac_cop


        # --- 3. Calculate temperature and humidity changes ---
        
        # Calculate net sensible heat change for temperature update
        net_sensible_change_W = net_sensible_gain_W - hvac_sensible_cooling_W
        
        # Calculate net moisture change (kg_water/s) from gains and HVAC removal
        moisture_gain_from_air_sources_kg_per_s = net_latent_gain_W / LATENT_HEAT_VAPORIZATION_J_KG_APPROX
        moisture_removal_by_hvac_kg_s = hvac_latent_cooling_W / LATENT_HEAT_VAPORIZATION_J_KG_APPROX
        net_moisture_to_air_from_sources_kg_per_s = moisture_gain_from_air_sources_kg_per_s - moisture_removal_by_hvac_kg_s

        # Update temperature
        delta_U_sensible_J = net_sensible_change_W * time_step_seconds
        delta_T_room_c = delta_U_sensible_J / total_room_sensible_thermal_capacity_J_K
        current_indoor_temp_c += delta_T_room_c

        # Update humidity ratio (with Buffering)
        W_range_at_current_T = get_humidity_ratio(current_indoor_temp_c, 100) - get_humidity_ratio(current_indoor_temp_c, 0)
        
        if W_range_at_current_T < 1e-9:
            W_change_per_percent_RH = 0.00016 # fallback
        else:
            W_change_per_percent_RH = W_range_at_current_T / 100.0

        moisture_buffering_effective_dry_air_mass_kg = room_total_moisture_buffering_capacity_kg_per_rh_percent / W_change_per_percent_RH if W_change_per_percent_RH > 0 else 0
        total_effective_moisture_mass_kg = room_air_mass_dry_kg + moisture_buffering_effective_dry_air_mass_kg

        if total_effective_moisture_mass_kg > 0: # Avoid division by zero
            delta_W = (net_moisture_to_air_from_sources_kg_per_s * time_step_seconds) / total_effective_moisture_mass_kg
            current_indoor_W += delta_W

        current_indoor_W = max(0, current_indoor_W) # Ensure humidity ratio doesn't go below 0
        current_indoor_rh = get_relative_humidity(current_indoor_temp_c, current_indoor_W)

        # Store HVAC data
        # For HVAC Status, we can say it's "ON" if any cooling capacity is requested
        hvac_status_on = 1 if desired_cooling_capacity_kW > 0.01 else 0 # Small threshold to avoid floating point issues
        hvac_cooling_load_kW_data.append(hvac_cooling_capacity_total_W / 1000.0) # kW
        hvac_power_consumption_kW_data.append(hvac_electrical_power_W / 1000.0) # kW

        # Accumulate total energy consumption
        total_energy_consumed_kWh += (hvac_electrical_power_W / 1000.0) * (time_step_minutes / 60.0)


    return pd.DataFrame({
        'Time': time_points,
        'Room Temperature (°C)': room_temperatures,
        'Room Relative Humidity (%)': room_relative_humidities,
        'Room Humidity Ratio (kg/kg_dry_air)': room_humidity_ratios,
        'HVAC Status (1=ON, 0=OFF)': hvac_status_on, # Changed to reflect PID's continuous nature
        'HVAC Cooling Load (kW)': hvac_cooling_load_kW_data,
        'HVAC Power Consumption (kW)': hvac_power_consumption_kW_data,
        'Total Heat Gain (kW)': total_heat_gain_data
    }), total_energy_consumed_kWh

# --- Run Simulation Button ---
st.markdown("Click 'Run Simulation' to see how the room temperature and HVAC power change over time.")
if st.button("Run Simulation"):
    with st.spinner("Simulating HVAC performance with PID control..."):
        simulation_df, total_energy_kWh = simulate_hvac_system(
            room_volume_m3, effective_wall_area, window_area, roof_area,
            U_wall, U_window, U_roof,
            room_structure_sensible_thermal_mass_J_K,
            room_total_moisture_buffering_capacity_kg_per_rh_percent,
            num_occupants, heat_per_person_sensible_W, heat_per_person_latent_W, lighting_power_W, equipment_power_W,
            air_changes_per_hour, AIR_DENSITY_KG_M3, SPECIFIC_HEAT_AIR_J_KGK, LATENT_HEAT_VAPORIZATION_J_KG_APPROX,
            outdoor_temp_c, outdoor_humidity_rh,
            indoor_setpoint_temp_c, indoor_setpoint_rh, initial_indoor_temp_c, initial_indoor_rh,
            hvac_cop, hvac_max_cooling_capacity_kW, hvac_shr,
            kp, ki, kd, # Pass PID parameters
            simulation_duration_hours, time_step_minutes
        )

    st.success("Simulation Complete!")

    st.markdown("---")
    st.subheader("4. Simulation Results & Visualization")

    # Display Total Energy Consumption
    st.metric("Total Energy Consumed", f"{total_energy_kWh:.2f} kWh", help="Total electrical energy consumed by HVAC over the simulation period.")

    # --- Plot Room Temperature vs Time ---
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=simulation_df['Time'], y=simulation_df['Room Temperature (°C)'],
                                  mode='lines', name='Room Temperature', line=dict(color='orange', width=2)))
    
    fig_temp.add_hline(y=indoor_setpoint_temp_c, line_dash="dash", line_color="green", annotation_text="Setpoint", annotation_position="top right")

    fig_temp.update_layout(
        title='Room Temperature Over Time (PID Control)',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        template="plotly_dark",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # --- Plot Room Relative Humidity vs Time ---
    fig_rh = go.Figure()
    fig_rh.add_trace(go.Scatter(x=simulation_df['Time'], y=simulation_df['Room Relative Humidity (%)'],
                                mode='lines', name='Room Relative Humidity', line=dict(color='lightgreen', width=2)))
    fig_rh.add_hline(y=indoor_setpoint_rh, line_dash="dash", line_color="blue", annotation_text="Setpoint RH", annotation_position="top right")

    fig_rh.update_layout(
        title='Room Relative Humidity Over Time',
        xaxis_title='Time',
        yaxis_title='Relative Humidity (%)',
        template="plotly_dark",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig_rh, use_container_width=True)

    # --- Plot HVAC Power Consumption & Cooling Load vs Time ---
    fig_hvac = go.Figure()
    fig_hvac.add_trace(go.Scatter(x=simulation_df['Time'], y=simulation_df['HVAC Cooling Load (kW)'],
                                   mode='lines', name='HVAC Cooling Load (kW)', fill='tozeroy', line=dict(color='cyan', width=1)))
    fig_hvac.add_trace(go.Scatter(x=simulation_df['Time'], y=simulation_df['HVAC Power Consumption (kW)'],
                                   mode='lines', name='HVAC Electrical Power (kW)', line=dict(color='red', width=2, dash='dot'))) # Use dot for power
    fig_hvac.update_layout(
        title='HVAC Load and Power Consumption Over Time (PID Control)',
        xaxis_title='Time',
        yaxis_title='Power / Load (kW)',
        template="plotly_dark",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig_hvac, use_container_width=True)


    # --- Plot Total Heat Gain vs Time ---
    fig_heat_gain = go.Figure()
    fig_heat_gain.add_trace(go.Scatter(x=simulation_df['Time'], y=simulation_df['Total Heat Gain (kW)'],
                                       mode='lines', name='Total Heat Gain', line=dict(color='purple', width=2)))
    fig_heat_gain.update_layout(
        title='Total Heat Gain into Room Over Time',
        xaxis_title='Time',
        yaxis_title='Heat Gain (kW)',
        template="plotly_dark",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig_heat_gain, use_container_width=True)


    # --- Data Export (Optional) ---
    csv_data = simulation_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Simulation Data as CSV",
        data=csv_data,
        file_name="hvac_simulation_data.csv",
        mime="text/csv",
        help="Download the full simulation time-series data."
    )