import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Common constants for air (ideal gas)
# Using common values for R, k, Cv, Cp as they are generally applicable across cycles for air
R_gas_common = 0.287  # kJ/(kg·K)
k_common = 1.4      # Specific heat ratio (gamma) for air
Cv_common = R_gas_common / (k_common - 1)  # kJ/(kg·K)
Cp_common = k_common * Cv_common        # kJ/(kg·K)

st.set_page_config(layout="wide", page_title="Thermodynamic Cycle Simulator", page_icon="⚙️")

st.title("⚙️ Thermodynamic Cycle Simulator")

st.markdown("""
    Explore the performance of ideal thermodynamic cycles.
    Enter the required parameters to calculate state point properties and visualize the cycles on P-v and T-s diagrams.
""")

cycle_choice = st.sidebar.selectbox(
    "Select Thermodynamic Cycle",
    ("Otto Cycle", "Diesel Cycle", "Brayton Cycle")
)

results = None
P_vals = []
v_vals = []
T_vals = []

if cycle_choice == "Otto Cycle":
    st.markdown("### Otto Cycle Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Initial State (State 1)")
        P1 = st.number_input("Pressure at State 1 ($P_1$) [kPa]", value=100.0, min_value=1.0, format="%.2f", key="P1_otto")
        T1 = st.number_input("Temperature at State 1 ($T_1$) [K]", value=300.0, min_value=1.0, format="%.2f", key="T1_otto")

    with col2:
        st.markdown("##### Cycle Parameters")
        compression_ratio = st.number_input("Compression Ratio ($r$) [$V_1/V_2$]", value=8.0, min_value=1.1, format="%.1f", key="r_otto")
        T3 = st.number_input("Temperature after Heat Addition ($T_3$) [K]", value=1500.0, min_value=T1 + 100, format="%.2f", key="T3_otto")

    if st.button("Calculate Otto Cycle Performance", key="calc_otto_btn"):
        try:
            # State 1
            v1 = (R_gas_common * T1) / P1

            # State 2 (Isentropic Compression)
            v2 = v1 / compression_ratio
            T2 = T1 * (compression_ratio**(k_common - 1))
            P2 = P1 * (compression_ratio**k_common)

            # State 3 (Constant Volume Heat Addition)
            v3 = v2 # Constant volume
            P3 = P2 * (T3 / T2)

            # State 4 (Isentropic Expansion)
            v4 = v1 # Constant volume
            T4 = T3 * ((v3 / v4)**(k_common - 1)) # or T4 = T3 / (compression_ratio**(k-1))
            P4 = P3 * ((v3 / v4)**k_common) # or P4 = P3 / (compression_ratio**k)

            q_in = Cv_common * (T3 - T2)
            q_out = Cv_common * (T4 - T1)
            w_net = q_in - q_out
            eta_th = w_net / q_in if q_in != 0 else 0
            
            # Mean Effective Pressure (MEP)
            # W_net = P_MEP * (V_max - V_min) = P_MEP * (V1 - V2)
            MEP = w_net / (v1 - v2) # [kPa] if w_net is in kJ/kg and v is m3/kg

            P_vals = [P1, P2, P3, P4]
            v_vals = [v1, v2, v3, v4]
            T_vals = [T1, T2, T3, T4]

            results = {
                "states": {
                    "P": P_vals,
                    "v": v_vals,
                    "T": T_vals
                },
                "performance": {
                    "q_in": q_in,
                    "q_out": q_out,
                    "w_net": w_net,
                    "eta_th": eta_th,
                    "MEP": MEP
                },
                "cycle_type": "otto",
                "state_labels": [1, 2, 3, 4] # For plotting and table
            }
        except Exception as e:
            st.error(f"An error occurred during Otto cycle calculation: {e}. Please check inputs.")

elif cycle_choice == "Diesel Cycle":
    st.markdown("### Diesel Cycle Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Initial State (State 1)")
        P1 = st.number_input("Pressure at State 1 ($P_1$) [kPa]", value=100.0, min_value=1.0, format="%.2f", key="P1_diesel")
        T1 = st.number_input("Temperature at State 1 ($T_1$) [K]", value=300.0, min_value=1.0, format="%.2f", key="T1_diesel")

    with col2:
        st.markdown("##### Cycle Parameters")
        compression_ratio = st.number_input("Compression Ratio ($r$) [$V_1/V_2$]", value=18.0, min_value=1.1, format="%.1f", key="r_diesel")
        cut_off_ratio = st.number_input("Cut-off Ratio ($r_c$) [$V_3/V_2$]", value=2.0, min_value=1.0, format="%.1f", key="rc_diesel")

    if st.button("Calculate Diesel Cycle Performance", key="calc_diesel_btn"):
        try:
            # State 1
            v1 = (R_gas_common * T1) / P1

            # State 2 (Isentropic Compression)
            v2 = v1 / compression_ratio
            T2 = T1 * (compression_ratio**(k_common - 1))
            P2 = P1 * (compression_ratio**k_common)

            # State 3 (Constant Pressure Heat Addition)
            v3 = v2 * cut_off_ratio
            P3 = P2 # Constant pressure
            T3 = T2 * cut_off_ratio # From ideal gas law P3V3/T3 = P2V2/T2, P3=P2 => V3/T3 = V2/T2 => T3 = T2 * (V3/V2)

            # State 4 (Isentropic Expansion)
            v4 = v1 # Constant volume
            T4 = T3 * ((v3 / v4)**(k_common - 1))
            P4 = P3 * ((v3 / v4)**k_common)

            q_in = Cp_common * (T3 - T2)
            q_out = Cv_common * (T4 - T1)
            w_net = q_in - q_out
            eta_th = w_net / q_in if q_in != 0 else 0
            
            # Mean Effective Pressure (MEP)
            MEP = w_net / (v1 - v2) # [kPa] if w_net is in kJ/kg and v is m3/kg

            P_vals = [P1, P2, P3, P4]
            v_vals = [v1, v2, v3, v4]
            T_vals = [T1, T2, T3, T4]

            results = {
                "states": {
                    "P": P_vals,
                    "v": v_vals,
                    "T": T_vals
                },
                "performance": {
                    "q_in": q_in,
                    "q_out": q_out,
                    "w_net": w_net,
                    "eta_th": eta_th,
                    "MEP": MEP
                },
                "cycle_type": "diesel",
                "state_labels": [1, 2, 3, 4] # For plotting and table
            }
        except Exception as e:
            st.error(f"An error occurred during Diesel cycle calculation: {e}. Please check inputs.")

elif cycle_choice == "Brayton Cycle":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Initial State (State 1)")
        P1 = st.number_input("Pressure at State 1 ($P_1$) [kPa]", value=100.0, min_value=1.0, format="%.2f", key="P1_brayton")
        T1 = st.number_input("Temperature at State 1 ($T_1$) [K]", value=300.0, min_value=1.0, format="%.2f", key="T1_brayton")

    with col2:
        st.markdown("##### Cycle Parameters")
        pressure_ratio = st.number_input("Pressure Ratio ($r_p$) [$P_2/P_1$]", value=10.0, min_value=1.1, format="%.1f", key="rp_brayton")
        T3 = st.number_input("Temperature at Turbine Inlet ($T_3$) [K, Max Cycle Temp]", value=1400.0, min_value=T1 + 100, format="%.2f", key="T3_brayton")
        
        st.markdown("##### Advanced Features")
        
        # Regeneration and Reheat can now be enabled together
        enable_regeneration = st.checkbox("Enable Regeneration", key="enable_regen_brayton")
        epsilon_regen = 0.0
        if enable_regeneration:
            epsilon_regen = st.number_input("Regenerator Effectiveness ($\epsilon_{regen}$)", value=0.8, min_value=0.01, max_value=0.99, format="%.2f", key="epsilon_regen_brayton")

        enable_reheat = st.checkbox("Enable Reheat", key="enable_reheat_brayton")
        reheat_pressure = 0.0
        if enable_reheat:
            P_reheat_default = P1 * np.sqrt(pressure_ratio) # A common approximation for optimal intermediate pressure
            reheat_pressure = st.number_input("Intermediate Reheat Pressure ($P_4$) [kPa]", 
                                              value=P_reheat_default, 
                                              min_value=P1 + 1, max_value=(P1 * pressure_ratio) - 1,
                                              format="%.2f", key="reheat_P4_brayton")


    if st.button("Calculate Brayton Cycle Performance", key="calc_brayton_btn"):
        try:
            # --- Base Calculations (always needed) ---
            # State 1
            v1 = (R_gas_common * T1) / P1

            # State 2 (Isentropic Compression)
            P2 = P1 * pressure_ratio
            T2 = T1 * (pressure_ratio**((k_common - 1) / k_common))
            v2 = (R_gas_common * T2) / P2
            
            # State 3 (Constant Pressure Heat Addition 1 / Turbine 1 Inlet)
            P3 = P2 
            v3 = (R_gas_common * T3) / P3 

            # --- Feasibility Checks (General Brayton) ---
            if T2 > T3: 
                 st.error(f"Error: Compressor exit temperature T2 ({T2:.2f} K) is higher than turbine inlet temperature T3 ({T3:.2f} K). Please increase T3 or decrease pressure ratio.")
                 st.stop()
            
            # --- Conditional Calculations based on Advanced Features ---
            cycle_type_result = "brayton" # Default

            # Initialize state points and works (defaults to 4-state unless modified)
            q_in = 0.0
            q_out = 0.0
            w_comp = Cp_common * (T2 - T1) 
            w_turb = 0.0 

            # Define variables that might change based on combination
            T2a, v2a = T2, v2 # For regeneration, 2a is the state after regenerator (cold side)
            
            # Intermediate T4_initial (from 4-state cycle) is used for regeneration calculation if no reheat
            P4_initial = P1 
            T4_initial = T3 / (pressure_ratio**((k_common - 1) / k_common)) 
            v4_initial = (R_gas_common * T4_initial) / P4_initial

            temp_P_vals = []
            temp_v_vals = []
            temp_T_vals = []
            actual_state_labels = []

            if enable_reheat and enable_regeneration:
                cycle_type_result = "brayton_reheat_regen"
                
                # State 4 (Turbine 1 Outlet)
                if not (P1 < reheat_pressure < P2):
                    st.error(f"Error: Reheat pressure ({reheat_pressure:.2f} kPa) must be between compressor outlet pressure ({P2:.2f} kPa) and ambient pressure ({P1:.2f} kPa).")
                    st.stop()
                P4 = reheat_pressure
                T4 = T3 * ((P4 / P3)**((k_common - 1) / k_common)) # Isentropic expansion from T3 to P4
                v4 = (R_gas_common * T4) / P4

                # State 5 (After Reheat, Turbine 2 Inlet)
                P5 = P4 # Constant pressure reheat
                T5 = T3 # Reheat back to max cycle temp (common assumption)
                v5 = (R_gas_common * T5) / P5

                # State 6 (Turbine 2 Outlet)
                P6 = P1 # Final exhaust pressure
                T6 = T5 * ((P6 / P5)**((k_common - 1) / k_common)) # Isentropic expansion from T5 to P6
                v6 = (R_gas_common * T6) / P6

                # Regenerator Cold Side (2-2a)
                # T6 is the hot stream temperature from the second turbine
                if T6 <= T2: # If turbine exhaust is colder than compressor outlet
                    st.warning("Warning: Regeneration is not effective (or possible) because turbine exhaust temperature (T6) is lower than or equal to compressor outlet temperature (T2).")
                    T2a = T2 # No heat gain
                else:
                    T2a = T2 + epsilon_regen * (T6 - T2)
                    if T2a > T6: # Cap T2a at T6 (effectiveness cannot exceed 1)
                        T2a = T6 
                        st.warning("Note: Calculated T2a reached T6, indicating maximum possible regeneration. Effectiveness might be virtually 1 for these conditions if input epsilon was higher.")
                v2a = (R_gas_common * T2a) / P2 

                # Regenerator Hot Side (6-6a)
                T6a = T6 - (T2a - T2) # Heat lost by hot stream equals heat gained by cold stream
                if T6a < T1: # Cap T6a at T1 (ambient) for calculation validity
                    st.warning(f"Warning: Regenerator exhaust temperature T6a ({T6a:.2f} K) fell below T1 ({T1:.2f} K). This implies an ideal regenerator for these conditions is not possible or effectiveness is too high. T6a is capped at T1 for calculation validity.")
                    T6a = T1
                v6a = (R_gas_common * T6a) / P6

                # Calculate Q_in, Q_out, W_turb for combined cycle
                q_in = Cp_common * (T3 - T2a) + Cp_common * (T5 - T4) # Heat added in CC1 and reheater
                q_out = Cp_common * (T6a - T1) # Heat rejected after regenerator
                w_turb = Cp_common * (T3 - T4) + Cp_common * (T5 - T6)

                temp_P_vals = [P1, P2, P2, P3, P4, P4, P1, P1] # P1, P2, P2a(P2), P3, P4, P5(P4), P6(P1), P6a(P1)
                temp_v_vals = [v1, v2, v2a, v3, v4, v5, v6, v6a]
                temp_T_vals = [T1, T2, T2a, T3, T4, T5, T6, T6a]
                actual_state_labels = [1, 2, "2a", 3, 4, 5, 6, "6a"]

            elif enable_regeneration: # Only Regeneration (no reheat)
                cycle_type_result = "brayton_regen"
                # Use T4_initial (calculated above) as the regenerator hot stream source
                
                # Regenerator Cold Side (2-2a)
                if T4_initial <= T2:
                    st.warning("Warning: Regeneration is not effective (or possible) because turbine exhaust temperature (T4) is lower than or equal to compressor outlet temperature (T2).")
                    T2a = T2 # No heat gain
                else:
                    T2a = T2 + epsilon_regen * (T4_initial - T2)
                    if T2a > T4_initial:
                        T2a = T4_initial 
                        st.warning("Note: Calculated T2a reached T4, indicating maximum possible regeneration. Effectiveness might be virtually 1 for these conditions if input epsilon was higher.")
                v2a = (R_gas_common * T2a) / P2 

                # Regenerator Hot Side (4-4a)
                T4a = T4_initial - (T2a - T2)
                if T4a < T1:
                    st.warning(f"Warning: Regenerator exhaust temperature T4a ({T4a:.2f} K) fell below T1 ({T1:.2f} K). This implies an ideal regenerator for these conditions is not possible or effectiveness is too high for these T values. T4a is capped at T1 for calculation validity.")
                    T4a = T1 
                v4a = (R_gas_common * T4a) / P4_initial

                q_in = Cp_common * (T3 - T2a) 
                q_out = Cp_common * (T4a - T1) 
                w_turb = Cp_common * (T3 - T4_initial) 
                
                temp_P_vals = [P1, P2, P2, P3, P4_initial, P4_initial] 
                temp_v_vals = [v1, v2, v2a, v3, v4_initial, v4a]
                temp_T_vals = [T1, T2, T2a, T3, T4_initial, T4a]
                actual_state_labels = [1, 2, "2a", 3, 4, "4a"]
            
            elif enable_reheat: # Only Reheat (no regeneration)
                cycle_type_result = "brayton_reheat"

                # Check reheat pressure validity
                if not (P1 < reheat_pressure < P2):
                    st.error(f"Error: Reheat pressure ({reheat_pressure:.2f} kPa) must be between compressor outlet pressure ({P2:.2f} kPa) and ambient pressure ({P1:.2f} kPa).")
                    st.stop()
                
                # State 4 (Turbine 1 Outlet)
                P4 = reheat_pressure
                T4 = T3 * ((P4 / P3)**((k_common - 1) / k_common)) # Isentropic expansion from T3 to P4
                v4 = (R_gas_common * T4) / P4

                # State 5 (After Reheat, Turbine 2 Inlet)
                P5 = P4 # Constant pressure reheat
                T5 = T3 # Reheat back to max cycle temp
                v5 = (R_gas_common * T5) / P5

                # State 6 (Turbine 2 Outlet)
                P6 = P1 # Final exhaust pressure
                T6 = T5 * ((P6 / P5)**((k_common - 1) / k_common)) # Isentropic expansion from T5 to P6
                v6 = (R_gas_common * T6) / P6

                # Calculate Q_in and W_turb for reheat cycle
                q_in = Cp_common * (T3 - T2) + Cp_common * (T5 - T4)
                q_out = Cp_common * (T6 - T1) # Heat rejected from state 6 to 1

                w_turb = Cp_common * (T3 - T4) + Cp_common * (T5 - T6)

                temp_P_vals = [P1, P2, P3, P4, P5, P6]
                temp_v_vals = [v1, v2, v3, v4, v5, v6]
                temp_T_vals = [T1, T2, T3, T4, T5, T6]
                actual_state_labels = [1, 2, 3, 4, 5, 6]

            else: # Standard 4-state Brayton (no regeneration, no reheat)
                q_in = Cp_common * (T3 - T2)
                q_out = Cp_common * (T4_initial - T1)
                w_turb = Cp_common * (T3 - T4_initial)

                # Initialize temp_P_vals, etc. with original 4 states
                temp_P_vals = [P1, P2, P3, P4_initial]
                temp_v_vals = [v1, v2, v3, v4_initial]
                temp_T_vals = [T1, T2, T3, T4_initial]
                actual_state_labels = [1, 2, 3, 4]

            # Common final calculations
            w_net = w_turb - w_comp 
            eta_th = w_net / q_in if q_in != 0 else 0 

            results = {
                "states": {
                    "P": temp_P_vals,
                    "v": temp_v_vals,
                    "T": temp_T_vals
                },
                "performance": {
                    "q_in": q_in,
                    "q_out": q_out,
                    "w_net": w_net,
                    "eta_th": eta_th,
                    "MEP": "N/A" 
                },
                "cycle_type": cycle_type_result, # 'brayton', 'brayton_regen', 'brayton_reheat', 'brayton_reheat_regen'
                "state_labels": actual_state_labels 
            }
        except Exception as e:
            st.error(f"An error occurred during Brayton cycle calculation: {e}. Please check inputs.")

# --- Display Results ---
if results:
    st.markdown("### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Heat Input ($q_{in}$)", f"{results['performance']['q_in']:.2f} kJ/kg")
    col2.metric("Heat Rejected ($q_{out}$)", f"{results['performance']['q_out']:.2f} kJ/kg")
    col3.metric("Net Work Output ($w_{net}$)", f"{results['performance']['w_net']:.2f} kJ/kg")
    col4.metric("Thermal Efficiency ($\eta_{th}$)", f"{results['performance']['eta_th'] * 100:.2f}%")

    if results['performance']['MEP'] != "N/A":
        st.metric("Mean Effective Pressure (MEP)", f"{results['performance']['MEP']:.2f} kPa")

    st.markdown("### State Point Properties")
    # Dynamically adjust column names based on regeneration
    state_labels_display = results["state_labels"] # Use the labels passed from calculation
    
    state_data_df = pd.DataFrame(results["states"])
    state_data_df.insert(0, "State", state_labels_display)
    st.dataframe(state_data_df.style.format(
        {"P": "{:.2f}", "v": "{:.4f}", "T": "{:.2f}"}
    ).set_table_styles([dict(selector="th", props=[("text-align", "center")])]),
    hide_index=True, use_container_width=True)

    # --- Plotting ---
    P_vals = results["states"]["P"]
    v_vals = results["states"]["v"]
    T_vals = results["states"]["T"]

    # P-v Diagram
    fig_pv = go.Figure()

    if results['cycle_type'] == "otto":
        # 1-2 Isentropic compression
        P_pv_12 = np.linspace(P_vals[0], P_vals[1], 50)
        v_pv_12 = v_vals[0] * (P_vals[0] / P_pv_12)**(1/k_common)
        fig_pv.add_trace(go.Scatter(x=v_pv_12, y=P_pv_12, mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))
        # 2-3 Constant volume heat addition
        fig_pv.add_trace(go.Scatter(x=[v_vals[1], v_vals[2]], y=[P_vals[1], P_vals[2]], mode='lines', name='2-3 Constant Volume Heat Addition', line=dict(color='red')))
        # 3-4 Isentropic expansion
        P_pv_34 = np.linspace(P_vals[2], P_vals[3], 50)
        v_pv_34 = v_vals[2] * (P_vals[2] / P_pv_34)**(1/k_common)
        fig_pv.add_trace(go.Scatter(x=v_pv_34, y=P_pv_34, mode='lines', name='3-4 Isentropic Expansion', line=dict(color='green')))
        # 4-1 Constant volume heat rejection
        fig_pv.add_trace(go.Scatter(x=[v_vals[3], v_vals[0]], y=[P_vals[3], P_vals[0]], mode='lines', name='4-1 Constant Volume Heat Rejection', line=dict(color='purple')))

    elif results['cycle_type'] == "diesel":
        # 1-2 Isentropic compression
        P_pv_12 = np.linspace(P_vals[0], P_vals[1], 50)
        v_pv_12 = v_vals[0] * (P_vals[0] / P_pv_12)**(1/k_common)
        fig_pv.add_trace(go.Scatter(x=v_pv_12, y=P_pv_12, mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))
        # 2-3 Constant pressure heat addition
        fig_pv.add_trace(go.Scatter(x=[v_vals[1], v_vals[2]], y=[P_vals[1], P_vals[2]], mode='lines', name='2-3 Constant Pressure Heat Addition', line=dict(color='red')))
        # 3-4 Isentropic expansion
        P_pv_34 = np.linspace(P_vals[2], P_vals[3], 50)
        v_pv_34 = v_vals[2] * (P_vals[2] / P_pv_34)**(1/k_common)
        fig_pv.add_trace(go.Scatter(x=v_pv_34, y=P_pv_34, mode='lines', name='3-4 Isentropic Expansion', line=dict(color='green')))
        # 4-1 Constant volume heat rejection
        fig_pv.add_trace(go.Scatter(x=[v_vals[3], v_vals[0]], y=[P_vals[3], P_vals[0]], mode='lines', name='4-1 Constant Volume Heat Rejection', line=dict(color='purple')))

    elif results['cycle_type'].startswith("brayton"): # This covers 'brayton', 'brayton_regen', 'brayton_reheat', 'brayton_reheat_regen'
        
        # P-v Diagram specific plotting logic
        if results['cycle_type'] == "brayton_regen":
            P_pv_12 = np.linspace(P_vals[0], P_vals[1], 50)
            v_pv_12 = v_vals[0] * (P_vals[0] / P_pv_12)**(1/k_common)
            fig_pv.add_trace(go.Scatter(x=v_pv_12, y=P_pv_12, mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))
            
            fig_pv.add_trace(go.Scatter(x=[v_vals[1], v_vals[2]], y=[P_vals[1], P_vals[2]], mode='lines', name='2-2a Regeneration Heat In', line=dict(color='orange', dash='dash'))) 
            fig_pv.add_trace(go.Scatter(x=[v_vals[2], v_vals[3]], y=[P_vals[2], P_vals[3]], mode='lines', name='2a-3 Constant Pressure Heat Addition', line=dict(color='red'))) 
            
            P_pv_34 = np.linspace(P_vals[3], P_vals[4], 50) 
            v_pv_34 = v_vals[3] * (P_vals[3] / P_pv_34)**(1/k_common)
            fig_pv.add_trace(go.Scatter(x=v_pv_34, y=P_pv_34, mode='lines', name='3-4 Isentropic Expansion', line=dict(color='green'))) 

            fig_pv.add_trace(go.Scatter(x=[v_vals[4], v_vals[5]], y=[P_vals[4], P_vals[5]], mode='lines', name='4-4a Regeneration Heat Out', line=dict(color='brown', dash='dash'))) 
            fig_pv.add_trace(go.Scatter(x=[v_vals[5], v_vals[0]], y=[P_vals[5], P_vals[0]], mode='lines', name='4a-1 Constant Pressure Heat Rejection', line=dict(color='purple'))) 

        elif results['cycle_type'] == "brayton_reheat":
            P_pv_12 = np.linspace(P_vals[0], P_vals[1], 50)
            v_pv_12 = v_vals[0] * (P_vals[0] / P_pv_12)**(1/k_common)
            fig_pv.add_trace(go.Scatter(x=v_pv_12, y=P_pv_12, mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))

            fig_pv.add_trace(go.Scatter(x=[v_vals[1], v_vals[2]], y=[P_vals[1], P_vals[2]], mode='lines', name='2-3 Constant Pressure Heat Addition 1', line=dict(color='red')))
            
            P_pv_34 = np.linspace(P_vals[2], P_vals[3], 50)
            v_pv_34 = v_vals[2] * (P_vals[2] / P_pv_34)**(1/k_common)
            fig_pv.add_trace(go.Scatter(x=v_pv_34, y=P_pv_34, mode='lines', name='3-4 Isentropic Expansion 1', line=dict(color='green')))

            fig_pv.add_trace(go.Scatter(x=[v_vals[3], v_vals[4]], y=[P_vals[3], P_vals[4]], mode='lines', name='4-5 Constant Pressure Reheat', line=dict(color='darkred', dash='dot'))) 
            
            P_pv_56 = np.linspace(P_vals[4], P_vals[5], 50)
            v_pv_56 = v_vals[4] * (P_vals[4] / P_pv_56)**(1/k_common)
            fig_pv.add_trace(go.Scatter(x=v_pv_56, y=P_pv_56, mode='lines', name='5-6 Isentropic Expansion 2', line=dict(color='darkgreen')))

            fig_pv.add_trace(go.Scatter(x=[v_vals[5], v_vals[0]], y=[P_vals[5], P_vals[0]], mode='lines', name='6-1 Constant Pressure Heat Rejection', line=dict(color='purple')))

        elif results['cycle_type'] == "brayton_reheat_regen":
            # P-v Diagram for Combined Reheat + Regeneration
            P_pv_12 = np.linspace(P_vals[0], P_vals[1], 50)
            v_pv_12 = v_vals[0] * (P_vals[0] / P_pv_12)**(1/k_common)
            fig_pv.add_trace(go.Scatter(x=v_pv_12, y=P_pv_12, mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))

            fig_pv.add_trace(go.Scatter(x=[v_vals[1], v_vals[2]], y=[P_vals[1], P_vals[2]], mode='lines', name='2-2a Regeneration Heat In', line=dict(color='orange', dash='dash')))
            fig_pv.add_trace(go.Scatter(x=[v_vals[2], v_vals[3]], y=[P_vals[2], P_vals[3]], mode='lines', name='2a-3 Constant Pressure Heat Addition 1', line=dict(color='red')))
            
            P_pv_34 = np.linspace(P_vals[3], P_vals[4], 50)
            v_pv_34 = v_vals[3] * (P_vals[3] / P_pv_34)**(1/k_common)
            fig_pv.add_trace(go.Scatter(x=v_pv_34, y=P_pv_34, mode='lines', name='3-4 Isentropic Expansion 1', line=dict(color='green')))

            fig_pv.add_trace(go.Scatter(x=[v_vals[4], v_vals[5]], y=[P_vals[4], P_vals[5]], mode='lines', name='4-5 Constant Pressure Reheat', line=dict(color='darkred', dash='dot')))
            
            P_pv_56 = np.linspace(P_vals[5], P_vals[6], 50)
            v_pv_56 = v_vals[5] * (P_vals[5] / P_pv_56)**(1/k_common)
            fig_pv.add_trace(go.Scatter(x=v_pv_56, y=P_pv_56, mode='lines', name='5-6 Isentropic Expansion 2', line=dict(color='darkgreen')))

            fig_pv.add_trace(go.Scatter(x=[v_vals[6], v_vals[7]], y=[P_vals[6], P_vals[7]], mode='lines', name='6-6a Regeneration Heat Out', line=dict(color='brown', dash='dash')))
            fig_pv.add_trace(go.Scatter(x=[v_vals[7], v_vals[0]], y=[P_vals[7], P_vals[0]], mode='lines', name='6a-1 Constant Pressure Heat Rejection', line=dict(color='purple')))

        else: # Original 4-state Brayton without regeneration or reheat
            fig_pv.add_trace(go.Scatter(x=[v_vals[1], v_vals[2]], y=[P_vals[1], P_vals[2]], mode='lines', name='2-3 Constant Pressure Heat Addition', line=dict(color='red')))
            P_pv_34 = np.linspace(P_vals[2], P_vals[3], 50)
            v_pv_34 = v_vals[2] * (P_vals[2] / P_pv_34)**(1/k_common)
            fig_pv.add_trace(go.Scatter(x=v_pv_34, y=P_pv_34, mode='lines', name='3-4 Isentropic Expansion', line=dict(color='green')))
            fig_pv.add_trace(go.Scatter(x=[v_vals[3], v_vals[0]], y=[P_vals[3], P_vals[0]], mode='lines', name='4-1 Constant Pressure Heat Rejection', line=dict(color='purple')))


    # Add state points to P-v diagram
    fig_pv.add_trace(go.Scatter(x=v_vals, y=P_vals, mode='markers+text', 
                                text=results["state_labels"], textposition="top right", 
                                marker=dict(size=8, color='black'), showlegend=False))
    
    fig_pv.update_layout(
        title=f'Ideal {cycle_choice} P-v Diagram',
        xaxis_title='Specific Volume (v) [m³/kg]',
        yaxis_title='Pressure (P) [kPa]',
        hovermode='x unified',
        template="plotly_dark",
        xaxis_type="log" if results['cycle_type'].startswith("brayton") else "linear", 
        yaxis_type="log" if results['cycle_type'].startswith("brayton") else "linear"
    )
    st.plotly_chart(fig_pv, use_container_width=True)

    # --- T-s Diagram ---
    fig_ts = go.Figure()
    # Entropy calculations (assuming s1 = 0 as reference)
    s_vals = [0] * len(T_vals) 
    
    if results['cycle_type'] == "otto":
        s_vals[0] = 0
        s_vals[1] = s_vals[0] # Isentropic
        s_vals[2] = s_vals[1] + Cv_common * np.log(T_vals[2] / T_vals[1])
        s_vals[3] = s_vals[2] # Isentropic
        
        fig_ts.add_trace(go.Scatter(x=[s_vals[0], s_vals[1]], y=[T_vals[0], T_vals[1]], mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))
        s_ts_23 = np.linspace(s_vals[1], s_vals[2], 50)
        T_ts_23 = T_vals[1] * np.exp((s_ts_23 - s_vals[1]) / Cv_common)
        fig_ts.add_trace(go.Scatter(x=s_ts_23, y=T_ts_23, mode='lines', name='2-3 Constant Volume Heat Addition', line=dict(color='red')))
        fig_ts.add_trace(go.Scatter(x=[s_vals[2], s_vals[3]], y=[T_vals[2], T_vals[3]], mode='lines', name='3-4 Isentropic Expansion', line=dict(color='green')))
        s_ts_41 = np.linspace(s_vals[3], s_vals[0], 50)
        T_ts_41 = T_vals[3] * np.exp((s_ts_41 - s_vals[3]) / Cv_common)
        fig_ts.add_trace(go.Scatter(x=s_ts_41, y=T_ts_41, mode='lines', name='4-1 Constant Volume Heat Rejection', line=dict(color='purple')))

    elif results['cycle_type'] == "diesel":
        s_vals[0] = 0
        s_vals[1] = s_vals[0] # Isentropic
        s_vals[2] = s_vals[1] + Cp_common * np.log(T_vals[2] / T_vals[1]) # Constant Pressure Heat Addition
        s_vals[3] = s_vals[2] # Isentropic
        
        fig_ts.add_trace(go.Scatter(x=[s_vals[0], s_vals[1]], y=[T_vals[0], T_vals[1]], mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))
        s_ts_23 = np.linspace(s_vals[1], s_vals[2], 50)
        T_ts_23 = T_vals[1] * np.exp((s_ts_23 - s_vals[1]) / Cp_common) # For constant P: T = T_i * exp((s-s_i)/Cp)
        fig_ts.add_trace(go.Scatter(x=s_ts_23, y=T_ts_23, mode='lines', name='2-3 Constant Pressure Heat Addition', line=dict(color='red')))
        fig_ts.add_trace(go.Scatter(x=[s_vals[2], s_vals[3]], y=[T_vals[2], T_vals[3]], mode='lines', name='3-4 Isentropic Expansion', line=dict(color='green')))
        s_ts_41 = np.linspace(s_vals[3], s_vals[0], 50)
        T_ts_41 = T_vals[3] * np.exp((s_ts_41 - s_vals[3]) / Cv_common)
        fig_ts.add_trace(go.Scatter(x=s_ts_41, y=T_ts_41, mode='lines', name='4-1 Constant Volume Heat Rejection', line=dict(color='purple')))

    elif results['cycle_type'].startswith("brayton"): # This covers 'brayton', 'brayton_regen', 'brayton_reheat', 'brayton_reheat_regen'
        s_vals[0] = 0
        s_vals[1] = s_vals[0] # 1-2 Isentropic Compression (s1=s2)

        # T-s Diagram specific plotting logic
        if results['cycle_type'] == "brayton_regen":
            s_vals[2] = s_vals[1] + Cp_common * np.log(T_vals[2] / T_vals[1]) # s_2a
            s_vals[3] = s_vals[2] + Cp_common * np.log(T_vals[3] / T_vals[2]) # s_3
            s_vals[4] = s_vals[3] # s_4
            s_vals[5] = s_vals[4] + Cp_common * np.log(T_vals[5] / T_vals[4]) # s_4a

            fig_ts.add_trace(go.Scatter(x=[s_vals[0], s_vals[1]], y=[T_vals[0], T_vals[1]], mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))

            s_ts_2_2a = np.linspace(s_vals[1], s_vals[2], 50)
            T_ts_2_2a = T_vals[1] * np.exp((s_ts_2_2a - s_vals[1]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_2_2a, y=T_ts_2_2a, mode='lines', name='2-2a Regeneration Heat In', line=dict(color='orange', dash='dash'))) 

            s_ts_2a_3 = np.linspace(s_vals[2], s_vals[3], 50)
            T_ts_2a_3 = T_vals[2] * np.exp((s_ts_2a_3 - s_vals[2]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_2a_3, y=T_ts_2a_3, mode='lines', name='2a-3 Constant Pressure Heat Addition', line=dict(color='red'))) 
            
            fig_ts.add_trace(go.Scatter(x=[s_vals[3], s_vals[4]], y=[T_vals[3], T_vals[4]], mode='lines', name='3-4 Isentropic Expansion', line=dict(color='green'))) 

            s_ts_4_4a = np.linspace(s_vals[4], s_vals[5], 50)
            T_ts_4_4a = T_vals[4] * np.exp((s_ts_4_4a - s_vals[4]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_4_4a, y=T_ts_4_4a, mode='lines', name='4-4a Regeneration Heat Out', line=dict(color='brown', dash='dash'))) 

            s_ts_4a_1 = np.linspace(s_vals[5], s_vals[0], 50)
            T_ts_4a_1 = T_vals[5] * np.exp((s_ts_4a_1 - s_vals[5]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_4a_1, y=T_ts_4a_1, mode='lines', name='4a-1 Constant Pressure Heat Rejection', line=dict(color='purple'))) 

        elif results['cycle_type'] == "brayton_reheat":
            s_vals[2] = s_vals[1] + Cp_common * np.log(T_vals[2] / T_vals[1]) # s_3 (after 1st heat add)
            s_vals[3] = s_vals[2] # s_4 (after 1st turbine - isentropic)
            s_vals[4] = s_vals[3] + Cp_common * np.log(T_vals[4] / T_vals[3]) # s_5 (after reheat)
            s_vals[5] = s_vals[4] # s_6 (after 2nd turbine - isentropic)

            fig_ts.add_trace(go.Scatter(x=[s_vals[0], s_vals[1]], y=[T_vals[0], T_vals[1]], mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))
            
            s_ts_23 = np.linspace(s_vals[1], s_vals[2], 50)
            T_ts_23 = T_vals[1] * np.exp((s_ts_23 - s_vals[1]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_23, y=T_ts_23, mode='lines', name='2-3 Constant Pressure Heat Addition 1', line=dict(color='red')))

            fig_ts.add_trace(go.Scatter(x=[s_vals[2], s_vals[3]], y=[T_vals[2], T_vals[3]], mode='lines', name='3-4 Isentropic Expansion 1', line=dict(color='green')))
            
            s_ts_45 = np.linspace(s_vals[3], s_vals[4], 50)
            T_ts_45 = T_vals[3] * np.exp((s_ts_45 - s_vals[3]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_45, y=T_ts_45, mode='lines', name='4-5 Constant Pressure Reheat', line=dict(color='darkred', dash='dot')))
            
            fig_ts.add_trace(go.Scatter(x=[s_vals[4], s_vals[5]], y=[T_vals[4], T_vals[5]], mode='lines', name='5-6 Isentropic Expansion 2', line=dict(color='darkgreen')))

            s_ts_61 = np.linspace(s_vals[5], s_vals[0], 50)
            T_ts_61 = T_vals[5] * np.exp((s_ts_61 - s_vals[5]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_61, y=T_ts_61, mode='lines', name='6-1 Constant Pressure Heat Rejection', line=dict(color='purple')))

        elif results['cycle_type'] == "brayton_reheat_regen":
            # T-s Diagram for Combined Reheat + Regeneration
            s_vals[2] = s_vals[1] + Cp_common * np.log(T_vals[2] / T_vals[1]) # s_2a
            s_vals[3] = s_vals[2] + Cp_common * np.log(T_vals[3] / T_vals[2]) # s_3
            s_vals[4] = s_vals[3] # s_4
            s_vals[5] = s_vals[4] + Cp_common * np.log(T_vals[4] / T_vals[3]) # s_5
            s_vals[6] = s_vals[5] # s_6
            s_vals[7] = s_vals[6] + Cp_common * np.log(T_vals[6] / T_vals[5]) # s_6a

            fig_ts.add_trace(go.Scatter(x=[s_vals[0], s_vals[1]], y=[T_vals[0], T_vals[1]], mode='lines', name='1-2 Isentropic Compression', line=dict(color='blue')))
            
            s_ts_2_2a = np.linspace(s_vals[1], s_vals[2], 50)
            T_ts_2_2a = T_vals[1] * np.exp((s_ts_2_2a - s_vals[1]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_2_2a, y=T_ts_2_2a, mode='lines', name='2-2a Regeneration Heat In', line=dict(color='orange', dash='dash'))) 

            s_ts_2a_3 = np.linspace(s_vals[2], s_vals[3], 50)
            T_ts_2a_3 = T_vals[2] * np.exp((s_ts_2a_3 - s_vals[2]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_2a_3, y=T_ts_2a_3, mode='lines', name='2a-3 Constant Pressure Heat Addition 1', line=dict(color='red'))) 
            
            fig_ts.add_trace(go.Scatter(x=[s_vals[3], s_vals[4]], y=[T_vals[3], T_vals[4]], mode='lines', name='3-4 Isentropic Expansion 1', line=dict(color='green'))) 

            s_ts_45 = np.linspace(s_vals[4], s_vals[5], 50)
            T_ts_45 = T_vals[4] * np.exp((s_ts_45 - s_vals[4]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_45, y=T_ts_45, mode='lines', name='4-5 Constant Pressure Reheat', line=dict(color='darkred', dash='dot')))
            
            fig_ts.add_trace(go.Scatter(x=[s_vals[5], s_vals[6]], y=[T_vals[5], T_vals[6]], mode='lines', name='5-6 Isentropic Expansion 2', line=dict(color='darkgreen')))

            s_ts_6_6a = np.linspace(s_vals[6], s_vals[7], 50)
            T_ts_6_6a = T_vals[6] * np.exp((s_ts_6_6a - s_vals[6]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_6_6a, y=T_ts_6_6a, mode='lines', name='6-6a Regeneration Heat Out', line=dict(color='brown', dash='dash'))) 

            s_ts_6a_1 = np.linspace(s_vals[7], s_vals[0], 50)
            T_ts_6a_1 = T_vals[7] * np.exp((s_ts_6a_1 - s_vals[7]) / Cp_common)
            fig_ts.add_trace(go.Scatter(x=s_ts_6a_1, y=T_ts_6a_1, mode='lines', name='6a-1 Constant Pressure Heat Rejection', line=dict(color='purple'))) 


        else: # Original Brayton without regeneration or reheat
            s_vals[2] = s_vals[1] + Cp_common * np.log(T_vals[2] / T_vals[1]) # s_3
            s_vals[3] = s_vals[2] # s_4

            s_ts_23 = np.linspace(s_vals[1], s_vals[2], 50)
            T_ts_23 = T_vals[1] * np.exp((s_ts_23 - s_vals[1]) / Cp_common) 
            fig_ts.add_trace(go.Scatter(x=s_ts_23, y=T_ts_23, mode='lines', name='2-3 Constant Pressure Heat Addition', line=dict(color='red')))
            fig_ts.add_trace(go.Scatter(x=[s_vals[2], s_vals[3]], y=[T_vals[2], T_vals[3]], mode='lines', name='3-4 Isentropic Expansion', line=dict(color='green')))
            s_ts_41 = np.linspace(s_vals[3], s_vals[0], 50)
            T_ts_41 = T_vals[3] * np.exp((s_ts_41 - s_vals[3]) / Cv_common) 
            fig_ts.add_trace(go.Scatter(x=s_ts_41, y=T_ts_41, mode='lines', name='4-1 Constant Pressure Heat Rejection', line=dict(color='purple')))


    # Add state points to T-s diagram
    fig_ts.add_trace(go.Scatter(x=s_vals, y=T_vals, mode='markers+text', 
                                text=results["state_labels"], textposition="top right", 
                                marker=dict(size=8, color='black'), showlegend=False))
    
    fig_ts.update_layout(
        title=f'Ideal {cycle_choice} T-s Diagram',
        xaxis_title='Specific Entropy (s) [kJ/(kg·K)]',
        yaxis_title='Temperature (T) [K]',
        hovermode='x unified',
        template="plotly_dark"
    )
    st.plotly_chart(fig_ts, use_container_width=True)