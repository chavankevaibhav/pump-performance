import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import json
import warnings # Import warnings

# Suppress specific matplotlib warnings if needed
warnings.filterfilterwarnings("ignore", category=UserWarning, module="matplotlib")


# --- Define the PumpSizing Class ---
# (Slightly modified for better integration and clarity in calculation context)

class PumpSizing:
    def __init__(self):
        # Fluid properties database (expand as needed) - Data is sample at ~25°C, 1 atm
        # rho: kg/m³, mu: Pa·s, vp: Pa
        self.fluids = {
            'Water': {'rho': 997, 'mu': 0.00089, 'vp': 2.34e3},
            'Ethanol': {'rho': 789, 'mu': 0.00108, 'vp': 7.87e3},
            'Crude Oil': {'rho': 850, 'mu': 0.002, 'vp': 1e3},
            'Glycerin': {'rho': 1260, 'mu': 1.49, 'vp': 0.004e3} # High viscosity example
        }

        # Sample pump efficiency correlation (Flow Q in m³/h)
        # This is a simplified example. Real efficiency curves are needed for accuracy.
        self.efficiency_curve = lambda Q: max(0.1, 0.85 - 0.0005 * (Q - 50)**2) # Example: Peak efficiency around 50 m³/h


        # Energy cost parameters - Can be updated via app inputs
        self.energy_cost_per_kwh = 0.15  # $/kWh
        self.operating_hours_per_year = 8760  # hours per year (24/7)


    def calculate_velocity(self, Q_m3h, D_m):
        """
        Calculate fluid velocity in a pipe.
        Args:
            Q_m3h (float): Flow rate in m³/h.
            D_m (float): Pipe diameter in meters.
        Returns:
            float: Velocity in m/s.
        """
        if D_m <= 0:
            return 0
        Q_m3s = Q_m3h / 3600.0 # Convert m³/h to m³/s
        Area = np.pi * (D_m/2)**2
        return Q_m3s / Area if Area > 0 else 0

    def reynolds_number(self, v_mps, D_m, rho_kgm3, mu_Pas):
        """
        Calculate the Reynolds number.
        Args:
            v_mps (float): Velocity in m/s.
            D_m (float): Pipe diameter in meters.
            rho_kgm3 (float): Fluid density in kg/m³.
            mu_Pas (float): Fluid dynamic viscosity in Pa·s.
        Returns:
            float: Reynolds number (dimensionless).
        """
        if mu_Pas <= 0:
             return np.inf # Avoid division by zero for non-viscous or ideal fluids (treat as turbulent)
        return (rho_kgm3 * v_mps * D_m) / mu_Pas

    def friction_factor(self, Re, epsilon_m, D_m):
        """
        Calculate Darcy friction factor using Churchill's correlation.
        Args:
            Re (float): Reynolds number.
            epsilon_m (float): Pipe absolute roughness in meters.
            D_m (float): Pipe diameter in meters.
        Returns:
            float: Darcy friction factor (dimensionless).
        """
        if Re <= 0:
            return 0 # No flow, no friction
        elif Re < 2000: # Laminar flow
            return 64 / Re
        else: # Turbulent flow - Churchill's correlation (smooth transition)
            Ed = epsilon_m / D_m # Relative roughness
            A = (-2.457 * np.log((7 / Re)**0.9 + 0.27 * Ed))**16
            B = (37530 / Re)**16
            return 8 * ((8 / Re)**12 + 1 / (A + B)**1.5)**(1 / 12)


    def calculate_friction_loss(self, Q_m3h, pipe_length_m, diameter_m, roughness_m, rho_kgm3, mu_Pas, g=9.81):
        """
        Calculate major head loss due to friction (Darcy-Weisbach equation).
        Args:
            Q_m3h (float): Flow rate in m³/h.
            pipe_length_m (float): Pipe length in meters.
            diameter_m (float): Pipe diameter in meters.
            roughness_m (float): Pipe absolute roughness in meters.
            rho_kgm3 (float): Fluid density in kg/m³.
            mu_Pas (float): Fluid dynamic viscosity in Pa·s.
            g (float): Acceleration due to gravity (m/s²).
        Returns:
            float: Friction head loss in meters of fluid column.
        """
        if Q_m3h < 1e-6 or pipe_length_m <= 0 or diameter_m <= 0: # Treat very low flow or invalid geometry as zero loss
            return 0.0
        v = self.calculate_velocity(Q_m3h, diameter_m)
        Re = self.reynolds_number(v, diameter_m, rho_kgm3, mu_Pas)
        f = self.friction_factor(Re, roughness_m, diameter_m)
        # Ensure v is not zero before division
        return f * (pipe_length_m / diameter_m) * (v**2 / (2 * g))


    def system_head_curve(self, Q_m3h, z_static_m, pipe_length_total_m, diameter_m, roughness_m, rho_kgm3, mu_Pas):
        """
        Calculate total system head required by the pump.
        Includes static head and friction head loss (major losses).
        Args:
            Q_m3h (float): Flow rate in m³/h.
            z_static_m (float): Total static head difference (discharge liquid level - suction liquid level) in meters.
            pipe_length_total_m (float): Total pipe length (suction + discharge) in meters.
            diameter_m (float): Main pipe diameter in meters.
            roughness_m (float): Main pipe absolute roughness in meters.
            rho_kgm3 (float): Fluid density in kg/m³.
            mu_Pas (float): Fluid dynamic viscosity in Pa·s.
        Returns:
            float: Total system head requirement in meters of fluid column.
        """
        h_f = self.calculate_friction_loss(Q_m3h, pipe_length_total_m, diameter_m, roughness_m, rho_kgm3, mu_Pas)
        # Minor losses are ignored here for simplicity but could be added (K * v^2 / 2g)
        return z_static_m + h_f

    def calculate_npsha(self, P_suction_Pa, P_vapor_Pa, z_suction_m, h_f_suction_m, rho_kgm3, g=9.81):
        """
        Calculate Net Positive Suction Head Available (NPSHa).
        NPSHa = (P_suction / rho*g) + z_suction - h_f_suction - (P_vapor / rho*g)
        Args:
            P_suction_Pa (float): Absolute pressure at the suction liquid surface (Pa).
            P_vapor_Pa (float): Fluid vapor pressure at pumping temperature (Pa).
            z_suction_m (float): Static head (vertical distance from suction liquid level to pump centerline) (m). Can be negative if pump is above liquid level.
            h_f_suction_m (float): Total head loss in the suction piping (m).
            rho_kgm3 (float): Fluid density in kg/m³.
            g (float): Acceleration due to gravity (m/s²).
        Returns:
            float: NPSHa in meters of fluid column.
        """
        if rho_kgm3 <= 0 or g <= 0:
             return -np.inf # Invalid density or gravity

        P_suction_head = P_suction_Pa / (rho_kgm3 * g)
        P_vapor_head = P_vapor_Pa / (rho_kgm3 * g)

        return P_suction_head + z_suction_m - h_f_suction_m - P_vapor_head

    def pump_power(self, Q_m3s, head_m, rho_kgm3, efficiency):
        """
        Calculate hydraulic and brake power.
        Args:
            Q_m3s (float): Flow rate in m³/s.
            head_m (float): Head delivered by the pump in meters.
            rho_kgm3 (float): Fluid density in kg/m³.
            efficiency (float): Pump efficiency (fraction, 0 to 1).
        Returns:
            tuple: (hydraulic_power_kW, brake_power_kW)
        """
        if efficiency <= 0 or rho_kgm3 <= 0 or head_m <= 0 or Q_m3s <= 0:
             return 0.0, 0.0 # No power if no flow, head, or efficiency

        hydraulic_power_W = Q_m3s * rho_kgm3 * 9.81 * head_m
        hydraulic_power_kW = hydraulic_power_W / 1000.0

        brake_power_kW = hydraulic_power_kW / efficiency if efficiency > 1e-6 else np.inf # Avoid division by zero

        return hydraulic_power_kW, brake_power_kW

    def optimize_operating_point(self, pump_head_func, system_head_func, Q_range_m3h):
        """
        Find operating point where pump curve intersects system curve.
        Args:
            pump_head_func (function): Function that takes Q (m³/h) and returns pump head (m).
            system_head_func (function): Function that takes Q (m³/h) and returns system head (m).
            Q_range_m3h (np.array): Range of flow rates (m³/h) to consider.
        Returns:
            tuple: (operating_flow_m3h, operating_head_m) or (None, None) if not found.
        """
        # Objective function: Difference between pump head and system head
        def head_difference(Q):
            # Ensure Q is within a reasonable, non-negative range for physical meaning
            Q = max(1e-6, Q) # Small positive flow to avoid issues at Q=0
            h_pump = pump_head_func(Q)
            h_system = system_head_func(Q)
            return (h_pump - h_system)**2 # Minimize the squared difference

        # Find a reasonable initial guess and bounds within the Q_range provided for plotting
        q_initial_guess = np.mean(Q_range_m3h)
        q_bounds = (min(Q_range_m3h), max(Q_range_m3h)) # Confine search to the plotting range

        # Use optimization to find where head_difference is close to zero
        # Using basin-hopping or differential_evolution might find global minimum better,
        # but minimize with bounds is usually sufficient for single pump curves.
        result = minimize(head_difference, x0=q_initial_guess, bounds=[q_bounds])

        # Check if the solution is a valid operating point (head difference is close to zero)
        if result.success and abs(result.fun)**0.5 < 0.1: # Check if sqrt(objective) < a small tolerance (0.1 m head difference)
             Q_op_m3h = result.x[0]
             H_op_m = pump_head_func(Q_op_m3h)
             # Check if head is positive at the operating point (physically required)
             if H_op_m > 0:
                return Q_op_m3h, H_op_m

        return None, None # Operating point not found

    def check_cavitation_risk(self, npsha_m, npshr_m):
        """
        Check cavitation risk based on NPSHa and NPSHr.
        Args:
            npsha_m (float): Net Positive Suction Head Available (m).
            npshr_m (float): Net Positive Suction Head Required (m).
        Returns:
            tuple: (Risk_Level, Margin_m)
                   Risk_Level is "Low Risk", "Moderate Risk", or "High Risk".
                   Margin_m is NPSHa - NPSHr.
        """
        margin = npsha_m - npshr_m
        # General guidelines (can vary by pump type and application)
        if margin >= 1.0: # Often NPSHa >= NPSHr + 1m or NPSHa >= 1.1 * NPSHr
            return "Low Risk", margin
        elif margin >= 0.5: # Some margin recommended
             return "Moderate Risk", margin
        else: # Risk of cavitation
            return "High Risk", margin

    def calculate_lifecycle_cost(self, purchase_cost, brake_power_kw, energy_cost_per_kwh, operating_hours_per_year, maintenance_percent, years=10):
        """
        Calculate lifecycle cost of the pump over a period.
        Args:
            purchase_cost (float): Initial purchase price ($).
            brake_power_kw (float): Pump brake power at the operating point (kW).
            energy_cost_per_kwh (float): Energy cost per kWh ($/kWh).
            operating_hours_per_year (float): Annual operating hours.
            maintenance_percent (float): Annual maintenance cost as a fraction of purchase cost.
            years (int): Lifecycle analysis period in years.
        Returns:
            dict: Breakdown of initial, annual, and total lifecycle costs.
        """
        annual_energy_cost = brake_power_kw * energy_cost_per_kwh * operating_hours_per_year
        annual_maintenance_cost = purchase_cost * maintenance_percent
        total_initial_cost = purchase_cost
        total_annual_cost = annual_energy_cost + annual_maintenance_cost
        total_lifecycle_cost = total_initial_cost + total_annual_cost * years

        return {
            'Initial Cost': total_initial_cost,
            'Annual Energy Cost': annual_energy_cost,
            'Annual Maintenance Cost': annual_maintenance_cost,
            'Total Annual Operating Cost': total_annual_cost,
            f'Total Lifecycle Cost ({years} yrs)': total_lifecycle_cost
        }

    # Helper to fit a quadratic pump curve from points
    def fit_quadratic_pump_curve(self, Q_points, H_points):
        """
        Fits a quadratic polynomial (H = aQ² + bQ + c) to pump curve points.
        Args:
            Q_points (list): List of flow rates (m³/h).
            H_points (list): List of corresponding heads (m).
        Returns:
            function: A function that takes Q (m³/h) and returns Head (m), or None if fitting fails.
            np.array: Array of coefficients [a, b, c], or None.
        """
        if len(Q_points) < 3 or len(H_points) < 3 or len(Q_points) != len(H_points):
            return None, None # Need at least 3 points to fit a quadratic

        try:
            # Use numpy to fit a 2nd degree polynomial
            coeffs = np.polyfit(Q_points, H_points, 2)
            # Create a polynomial function from the coefficients
            pump_func = np.poly1d(coeffs)
            return pump_func, coeffs
        except Exception as e:
            st.error(f"Error fitting pump curve: {e}")
            return None, None


    def plot_pump_curve(self, Q_range_m3h, pump_head_curve_values, system_head_curve_values,
                        npsha_curve_values=None, npshr_m=None,
                        op_point_Q_m3h=None, op_point_H_m=None, npsha_op_m=None, cavitation_risk_info=None):
        """
        Plot pump performance curves (System, Pump, NPSHa) and operating point.
        Args:
            Q_range_m3h (np.array): Range of flow rates (m³/h) for plotting.
            pump_head_curve_values (list/np.array): Pump head values (m) for Q_range_m3h.
            system_head_curve_values (list/np.array): System head values (m) for Q_range_m3h.
            npsha_curve_values (list/np.array, optional): NPSHa values (m) for Q_range_m3h.
            npshr_m (float, optional): Pump required NPSHr (m).
            op_point_Q_m3h (float, optional): Operating point flow rate (m³/h).
            op_point_H_m (float, optional): Operating point head (m).
            npsha_op_m (float, optional): NPSHa at the operating point (m).
            cavitation_risk_info (tuple, optional): (Risk_Level, Margin_m) at OP.
        """
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Primary Y-axis for Head curves
        ax1.plot(Q_range_m3h, system_head_curve_values, 'b-', linewidth=2, label='System Curve')
        ax1.plot(Q_range_m3h, pump_head_curve_values, 'r--', linewidth=2, label='Pump Curve')

        ax1.set_xlabel('Flow Rate (m³/h)')
        ax1.set_ylabel('Head (m)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot Operating Point
        if op_point_Q_m3h is not None and op_point_H_m is not None:
            ax1.plot(op_point_Q_m3h, op_point_H_m, 'ko', markersize=8, label='Operating Point') # Use 'k' for black point

            # Annotation
            annotation_text = f"Flow: {op_point_Q_m3h:.2f} m³/h\nHead: {op_point_H_m:.2f} m"
            if npsha_op_m is not None:
                 annotation_text += f"\nNPSHa: {npsha_op_m:.2f} m"
            if cavitation_risk_info:
                 annotation_text += f"\nRisk: {cavitation_risk_info[0]}"

            # Smart annotation placement attempt (basic)
            x_offset = max(Q_range_m3h) * 0.05 # 5% of max flow
            y_offset = max(system_head_curve_values) * 0.05 # 5% of max head
            ax1.annotate(annotation_text,
                         xy=(op_point_Q_m3h, op_point_H_m),
                         xytext=(op_point_Q_m3h + x_offset, op_point_H_m + y_offset),
                         arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"),
                         bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))


        # Secondary Y-axis for NPSH curves
        if npsha_curve_values is not None or npshr_m is not None:
            ax2 = ax1.twinx()
            if npsha_curve_values is not None:
                 ax2.plot(Q_range_m3h, npsha_curve_values, 'g-.', linewidth=2, label='NPSHa')
            if npshr_m is not None:
                 ax2.axhline(y=npshr_m, color='orange', linestyle=':', linewidth=2, label=f'NPSHr ({npshr_m:.1f} m)')

            ax2.set_ylabel('NPSH (m)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')

            # Combine legends from both axes
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='best')
        else:
             # If no NPSH, just show the head legend on ax1
             ax1.legend(loc='best')


        ax1.set_title('Pump and System Curve Analysis')
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        st.pyplot(fig) # Display the plot in Streamlit


# --- Helper function for Streamlit app ---
# No need to redefine fit_quadratic_pump_curve here if it's a method of the class.
# Let's make it a static method or keep it as a separate function outside the class
# for clarity as it's used *by* the app logic to create the pump_head_func.
# Keeping it outside is fine.

def fit_quadratic_pump_curve(Q_points, H_points):
    """
    Fits a quadratic polynomial (H = aQ² + bQ + c) to pump curve points.
    Args:
        Q_points (list): List of flow rates (m³/h).
        H_points (list): List of corresponding heads (m).
    Returns:
        function: A function that takes Q (m³/h) and returns Head (m), or None if fitting fails.
        np.array: Array of coefficients [a, b, c], or None.
    """
    # Filter out empty or zero-head points if necessary, but polyfit handles identical points up to a limit
    # Require at least 3 unique Q points for a robust quadratic fit.
    if len(set(Q_points)) < 3:
         return None, None # Need at least 3 distinct Q values

    try:
        # Use numpy to fit a 2nd degree polynomial
        coeffs = np.polyfit(Q_points, H_points, 2)
        # Create a polynomial function from the coefficients
        pump_func = np.poly1d(coeffs)
        return pump_func, coeffs
    except Exception as e:
        # This might happen if points are collinear or insufficient
        st.warning(f"Could not fit a quadratic curve to the provided points. Please check inputs. Error: {e}")
        return None, None


# --- Streamlit App Layout ---

st.set_page_config(page_title="Pump Sizing & Analysis", layout="wide", icon=" насос") # Add an icon

st.title("Centrifugal Pump Sizing and System Analysis")

st.sidebar.header("Input Parameters")

# --- Sidebar Inputs ---

ps_initial = PumpSizing() # Create a temporary instance to access fluid data

st.sidebar.subheader("Fluid Properties")
fluid_name = st.sidebar.selectbox("Select Fluid", list(ps_initial.fluids.keys()))
fluid_props = ps_initial.fluids[fluid_name]
rho = st.sidebar.number_input("Density (kg/m³)", value=float(fluid_props['rho']), min_value=0.1, format="%.2f")
mu = st.sidebar.number_input("Dynamic Viscosity (Pa·s)", value=float(fluid_props['mu']), min_value=0.0, format="%.4f")
P_vapor = st.sidebar.number_input("Vapor Pressure (Pa)", value=float(fluid_props['vp']), min_value=0.0, format="%.1f")

st.sidebar.subheader("System Parameters")
z_static = st.sidebar.number_input("Total Static Head (m)", value=10.0, min_value=-100.0, help="Difference in liquid level between discharge and suction tanks.")
pipe_length_total = st.sidebar.number_input("Total Pipe Length (m)", value=100.0, min_value=0.1)
pipe_diameter_main = st.sidebar.number_input("Main Pipe Diameter (m)", value=0.15, min_value=0.01)
pipe_roughness_main = st.sidebar.number_input("Main Pipe Roughness (m)", value=0.045e-3, min_value=0.0, format="%.1e", help="Typical values: Steel (0.045mm), PVC (0.0015mm)")


st.sidebar.subheader("Pump Curve Definition (H vs Q)")
st.sidebar.write("Enter at least 3 Flow (m³/h) and Head (m) points to define the pump curve.")

# Use st.columns to align inputs
col1, col2 = st.sidebar.columns(2)
Q_points_input = []
H_points_input = []

# Default points for a sample curve - users can change these
default_points = [(0, 40), (50, 30), (100, 0)] # (Flow, Head)

for i in range(1, 5): # Allow up to 4 points for fitting
    Q_val = col1.number_input(f"Flow {i} (m³/h)", value=float(default_points[i-1][0]) if i <= len(default_points) else 0.0, format="%.1f", key=f"Q_point_{i}", min_value=0.0)
    H_val = col2.number_input(f"Head {i} (m)", value=float(default_points[i-1][1]) if i <= len(default_points) else 0.0, format="%.1f", key=f"H_point_{i}")
    # Only add points if Q is entered (assuming valid points have Q > 0, except shutoff)
    if Q_val > 0 or H_val > 0: # Allow shutoff point (0, H) or other valid points
         Q_points_input.append(Q_val)
         H_points_input.append(H_val)

# Filter out duplicates based on Q if any
unique_points = {}
for q, h in zip(Q_points_input, H_points_input):
    unique_points[q] = h # Keep the last head value for any given flow
Q_points_input = sorted(list(unique_points.keys()))
H_points_input = [unique_points[q] for q in Q_points_input]


st.sidebar.subheader("NPSH Parameters")
P_suction = st.sidebar.number_input("Suction Pressure (Pa)", value=101325.0, min_value=0.0, help="Absolute pressure at the liquid surface of the suction tank.")
z_suction = st.sidebar.number_input("Suction Static Head (m)", value=2.0, help="Vertical distance from suction liquid level to pump centerline (+ if level is above pump, - if below).")
suction_pipe_length = st.sidebar.number_input("Suction Pipe Length (m)", value=5.0, min_value=0.1)
suction_pipe_diameter = st.sidebar.number_input("Suction Pipe Diameter (m)", value=0.15, min_value=0.01)
suction_pipe_roughness = st.sidebar.number_input("Suction Pipe Roughness (m)", value=0.045e-3, min_value=0.0, format="%.1e")
npshr = st.sidebar.number_input("Pump Required NPSH (NPSHr) (m)", value=3.0, min_value=0.1, help="This value comes from the pump manufacturer's data.")


st.sidebar.subheader("Cost Parameters")
purchase_cost = st.sidebar.number_input("Pump Purchase Cost ($)", value=5000.0, min_value=0.0)
energy_cost_per_kwh = st.sidebar.number_input("Energy Cost ($/kWh)", value=0.15, min_value=0.01)
operating_hours_per_year = st.sidebar.number_input("Operating Hours per Year", value=8760, min_value=0)
maintenance_percent = st.sidebar.number_input("Annual Maintenance Cost (% of Purchase Cost)", value=5.0, min_value=0.0) / 100.0 # Convert to fraction


# --- Calculation Button ---
st.header("Analysis Results")

# Initialize PumpSizing instance for calculations
ps_calc = PumpSizing()
# Update cost parameters in the instance based on user input
ps_calc.energy_cost_per_kwh = energy_cost_per_kwh
ps_calc.operating_hours_per_year = operating_hours_per_year


calculate_button = st.button("Run Analysis")

if calculate_button:
    # --- Fit Pump Curve ---
    pump_func, pump_coeffs = fit_quadratic_pump_curve(Q_points_input, H_points_input)

    if pump_func is None:
         st.warning("Please provide at least 3 valid and distinct points to define the pump curve.")
    else:
        st.info("Calculating...")

        # Determine Q range for plotting based on input points
        max_Q_input = max(Q_points_input) if Q_points_input else 100
        Q_range_m3h = np.linspace(0, max_Q_input * 1.5 + 10, 100) # Extend range for plotting

        # Calculate System Head Curve values over the Q range
        system_head_curve_values = [ps_calc.system_head_curve(Q, z_static, pipe_length_total, pipe_diameter_main, pipe_roughness_main, rho, mu) for Q in Q_range_m3h]
        # Calculate Pump Head Curve values over the Q range using the fitted function
        pump_head_curve_values = [pump_func(Q) for Q in Q_range_m3h]

        # Calculate Suction Friction Loss curve values for NPSHa
        h_f_suction_curve_values = [ps_calc.calculate_friction_loss(Q, suction_pipe_length, suction_pipe_diameter, suction_pipe_roughness, rho, mu) for Q in Q_range_m3h]
        # Calculate NPSHa Curve values
        npsha_curve_values = [ps_calc.calculate_npsha(P_suction, P_vapor, z_suction, h_f_suction, rho) for h_f_suction in h_f_suction_curve_values]


        # --- Find Operating Point ---
        # Use interpolation of calculated curve values for the optimizer
        system_curve_interp = lambda Q: np.interp(Q, Q_range_m3h, system_head_curve_values, left=system_head_curve_values[0] + 1e9, right=system_head_curve_values[-1] + 1e9 if len(system_head_curve_values)>1 else system_head_curve_values[-1] + 1e9) # Extrapolate as very high head outside range
        pump_curve_interp = lambda Q: np.interp(Q, Q_range_m3h, pump_head_curve_values, left=pump_head_curve_values[0] - 1e9, right=pump_head_curve_values[-1] - 1e9 if len(pump_head_curve_values)>1 else pump_head_curve_values[-1] - 1e9) # Extrapolate as very low head outside range

        # Call the optimization method
        Q_op_m3h, H_op_m = ps_calc.optimize_operating_point(pump_curve_interp, system_curve_interp, Q_range_m3h)


        # --- Display Results and Plot ---
        if Q_op_m3h is not None and H_op_m is not None:
            st.subheader("Operating Point Details")

            # Calculate results specifically at the operating point
            Q_op_m3s = Q_op_m3h / 3600.0 # Convert to m³/s for power
            v_suction_op = ps_calc.calculate_velocity(Q_op_m3h, suction_pipe_diameter)
            h_f_suction_op = ps_calc.calculate_friction_loss(Q_op_m3h, suction_pipe_length, suction_pipe_diameter, suction_pipe_roughness, rho, mu)
            npsha_op = ps_calc.calculate_npsha(P_suction, P_vapor, z_suction, h_f_suction_op, rho)
            cavitation_risk_op, npsh_margin_op = ps_calc.check_cavitation_risk(npsha_op, npshr)

            # Efficiency at operating point
            efficiency_op = ps_calc.efficiency_curve(Q_op_m3h)
            # Add a check/warning for unrealistic efficiency
            if efficiency_op < 0.01 or efficiency_op > 1.0:
                 st.warning(f"Calculated efficiency at operating point ({efficiency_op*100:.1f}%) is outside the typical range (1%-100%). This may indicate the sample efficiency curve or operating point flow is unrealistic.")
                 efficiency_op = max(0.01, min(0.99, efficiency_op)) # Cap for power calculation if needed


            hydraulic_power_op_kw, brake_power_op_kw = ps_calc.pump_power(Q_op_m3s, H_op_m, rho, efficiency_op)

            # Recalculate lifecycle cost with operating point brake power
            lifecycle_cost_results = ps_calc.calculate_lifecycle_cost(
                purchase_cost=purchase_cost,
                brake_power_kw=brake_power_op_kw,
                energy_cost_per_kwh=energy_cost_per_kwh,
                operating_hours_per_year=operating_hours_per_year,
                maintenance_percent=maintenance_percent,
                years=10 # Default 10 years for display
            )

            # Display key metrics using columns
            st.markdown("#### Key Metrics")
            col_op1, col_op2, col_op3 = st.columns(3)
            col_op1.metric("Operating Flow", f"{Q_op_m3h:.2f} m³/h")
            col_op2.metric("Operating Head", f"{H_op_m:.2f} m")
            col_op3.metric("Pump Efficiency", f"{efficiency_op*100:.1f}%")

            st.markdown("#### NPSH Analysis")
            col_npsh1, col_npsh2, col_npsh3 = st.columns(3)
            col_npsh1.metric("NPSHa", f"{npsha_op:.2f} m")
            col_npsh2.metric("NPSHr", f"{npshr:.1f} m")
            col_npsh3.metric("Cavitation Risk", cavitation_risk_op, delta=f"{npsh_margin_op:.2f} m margin") # Show margin as delta

            st.markdown("#### Power Requirements")
            col_power1, col_power2 = st.columns(2)
            col_power1.metric("Hydraulic Power", f"{hydraulic_power_op_kw:.2f} kW")
            col_power2.metric("Brake Power", f"{brake_power_op_kw:.2f} kW")

            st.markdown("#### Lifecycle Cost Analysis (10 Years)")
            # Convert dict to DataFrame for better display
            lc_df = pd.DataFrame(list(lifecycle_cost_results.items()), columns=['Cost Category', 'Amount ($)'])
            st.dataframe(lc_df.set_index('Cost Category'), use_container_width=True)


            # --- Plotting ---
            st.subheader("Pump, System, and NPSH Curves")
            ps_calc.plot_pump_curve(Q_range_m3h, pump_head_curve_values, system_head_curve_values,
                                    npsha_curve_values=npsha_curve_values, npshr_m=npshr,
                                    op_point_Q_m3h=Q_op_m3h, op_point_H_m=H_op_m, npsha_op_m=npsha_op,
                                    cavitation_risk_info=(cavitation_risk_op, npsh_margin_op)) # Pass risk info for annotation


            # --- Export Results ---
            results_data = {
                'input_parameters': {
                    'fluid': fluid_name,
                    'fluid_properties': {'rho': rho, 'mu': mu, 'vp': P_vapor},
                    'system_parameters': {
                        'z_static_total_m': z_static,
                        'pipe_length_total_m': pipe_length_total,
                        'pipe_diameter_main_m': pipe_diameter_main,
                        'pipe_roughness_main_m': pipe_roughness_main
                    },
                     'pump_curve_input_points': [{'Q_m3h': q, 'H_m': h} for q, h in zip(Q_points_input, H_points_input)],
                     'fitted_pump_curve_coeffs': pump_coeffs.tolist() if pump_coeffs is not None else None,
                    'npsh_parameters': {
                        'P_suction_Pa': P_suction,
                        'z_suction_m': z_suction,
                        'suction_pipe_length_m': suction_pipe_length,
                        'suction_pipe_diameter_m': suction_pipe_diameter,
                        'suction_pipe_roughness_m': suction_pipe_roughness,
                        'npshr_m': npshr
                    },
                    'cost_parameters': {
                        'purchase_cost_$': purchase_cost,
                        'energy_cost_per_kwh_$': energy_cost_per_kwh,
                        'operating_hours_per_year': operating_hours_per_year,
                        'annual_maintenance_percent': maintenance_percent * 100
                    }
                },
                'operating_point_results': {
                    'flow_rate_m3h': Q_op_m3h,
                    'head_m': H_op_m,
                    'npsha_m': npsha_op,
                    'npshr_m': npshr,
                    'npsh_margin_m': npsh_margin_op,
                    'cavitation_risk': cavitation_risk_op,
                    'efficiency': efficiency_op,
                    'hydraulic_power_kw': hydraulic_power_op_kw,
                    'brake_power_kw': brake_power_op_kw
                },
                'lifecycle_cost_results': lifecycle_cost_results,
                'calculated_curves_for_plot': {
                    'Q_range_m3h': Q_range_m3h.tolist(),
                    'system_head_m': system_head_curve_values,
                    'pump_head_m': pump_head_curve_values,
                    'npsha_m': npsha_curve_values
                }
            }

            st.subheader("Export Results")
            # Convert data to JSON string
            json_string = json.dumps(results_data, indent=4)
            st.download_button(
                label="Download Results as JSON",
                file_name="pump_sizing_results.json",
                mime="application/json",
                data=json_string
            )

        else:
            st.warning("Could not find a clear operating point where the pump curve intersects the system curve within the defined flow range.")
            st.info("Check the pump curve points and system parameters. The pump may be too small (curve below system curve) or too large (intersection outside realistic flow range).")

            # Still plot the curves even if OP not found
            st.subheader("Pump, System, and NPSH Curves")
            # Ensure curve values are lists for plotting function
            system_head_curve_values = [ps_calc.system_head_curve(Q, z_static, pipe_length_total, pipe_diameter_main, pipe_roughness_main, rho, mu) for Q in Q_range_m3h]
            pump_head_curve_values = [pump_func(Q) for Q in Q_range_m3h] # Use fitted function if available
            h_f_suction_curve_values = [ps_calc.calculate_friction_loss(Q, suction_pipe_length, suction_pipe_diameter, suction_pipe_roughness, rho, mu) for Q in Q_range_m3h]
            npsha_curve_values = [ps_calc.calculate_npsha(P_suction, P_vapor, z_suction, h_f_suction, rho) for h_f_suction in h_f_suction_curve_values]


            ps_calc.plot_pump_curve(Q_range_m3h, pump_head_curve_values, system_head_curve_values,
                                    npsha_curve_values=npsha_curve_values, npshr_m=npshr) # Don't pass OP info


# --- About Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This application performs basic centrifugal pump hydraulic and cost analysis. "
    "It calculates system head, NPSH available, finds the operating point, estimates power, "
    "and provides a lifecycle cost breakdown. "
    "**Note:** This tool uses simplified models (e.g., quadratic pump curve fit, no minor losses, simple efficiency curve) "
    "and should be used for preliminary estimation only. Consult manufacturer data and detailed engineering for critical applications."
)
