# D:\streamlit_apps\simulator\0_Home.py (NOW IN THE ROOT)
import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

# --- Hide the sidebar using CSS ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: none
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# --- END Hide the sidebar ---


st.title("Welcome to the Engineering Simulators Hub! 🔧")
st.markdown("---")
st.write(
    """
    Explore various engineering principles through interactive simulators.
    Click on any project title below to launch the respective simulator.
    """
)

st.subheader("Available Simulators:")

# --- Use st.page_link for in-page navigation, referencing 'pages/' with NO EMOJIS ---

# F1 Brake Thermal Analysis
st.markdown("### 🏎️ F1 Brake Thermal Analysis") # Display emoji here
st.write("""
    A sophisticated simulator modeling the transient temperature distribution in Formula 1 carbon-carbon brake discs during braking events.
    Analyze peak temperatures, cooling rates, thermal gradients, and critical KPIs.
""")
# Corrected PATH: No emoji in filename
st.page_link("pages/5_F1_Brake_Thermal_Analysis.py", label="Launch F1 Brake Simulator", icon="🚀")
st.markdown("---")


# 1D Heat Conduction Simulator
st.markdown("### 🔥 1D Heat Conduction Simulator") # Display emoji here
st.write("""
    Understand the fundamentals of heat transfer in one dimension, exploring steady-state and transient conditions with various boundary conditions.
""")
# Corrected PATH: No emoji in filename
st.page_link("pages/1_Heat_Transfer_Simulator.py", label="Launch Heat Conduction Simulator", icon="🚀")
st.markdown("---")


# Pipe Flow Pressure Drop Simulator (Under Development)
st.markdown("### 💧 Pipe Flow Pressure Drop Simulator (Under Development)") # Display emoji here
st.write("""
    A tool to analyze fluid dynamics and calculate pressure drops in pipe systems, essential for hydraulic design.
""")
# Corrected PATH: No emoji in filename
st.page_link("pages/2_Pipe_Flow_Pressure_Drop_Simulator.py", label="Launch Pipe Flow Simulator", icon="🚀")
st.markdown("---")


# Thermodynamic Cycle Simulator (Under Development)
st.markdown("### ⚙️ Thermodynamic Cycle Simulator (Under Development)") # Display emoji here
st.write("""
    Explore the principles of thermodynamic cycles, vital for power generation and refrigeration systems.
""")
# Corrected PATH: No emoji in filename
st.page_link("pages/3_Thermodynamic_Cycle_Simulator.py", label="Launch Thermodynamic Simulator", icon="🚀")
st.markdown("---")


# Smart HVAC Optimizer (Under Development)
st.markdown("### 💡 Smart HVAC Optimizer (Under Development)") # Display emoji here
st.write("""
    A forward-looking project aimed at optimizing HVAC energy consumption for improved efficiency and comfort.
""")
# Corrected PATH: No emoji in filename
st.page_link("pages/4_Smart_HVAC_Optimizer.py", label="Launch HVAC Optimizer", icon="🚀")
st.markdown("---")

st.info("💡 Tip: Use the '🏠 Go to Home' button on each simulator page to return to this menu.")