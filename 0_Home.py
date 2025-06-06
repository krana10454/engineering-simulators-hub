# D:\streamlit_apps\simulator\pages\0_Home.py
import streamlit as st

# It's good practice to set page config for each page as well,
# to control its specific title and icon in the browser tab.
st.set_page_config(
    page_title="Home",
    page_icon="🏠", # A home icon
    layout="wide"
)

# --- UPDATED TITLE HERE ---
st.title("Welcome to the Engineering Simulators Hub:🔧")
st.markdown("---")
st.write(
    """
    Explore various engineering principles through interactive simulators.
    Select a simulator from the sidebar to begin.
    """
)

st.subheader("Available Simulators:")
st.write("- 1D Heat Conduction Simulator 🔥")
st.write("- Pipe Flow Pressure Drop Simulator 💧")
st.write("- Thermodynamic Cycle Simulator ⚙️")
st.write("- Smart HVAC Optimizer 💡")
st.write("- F1 Brake Thermal Analysis 🏎️")

st.markdown("---")
st.info("💡 Tip: Use the sidebar to navigate between simulators.")