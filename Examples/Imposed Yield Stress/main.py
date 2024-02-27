### This code was written by Matthieu Claudon (Mines Paris - PSL) ###

import sys
sys.path.append("..")
from material_law_identification import read_data, run_all_configurations, hansel_spittel, FORGE_hansel_spittel, plot_stress_vs_strain, plot_force_vs_displacement

### MANDATORY ###
used_function = FORGE_hansel_spittel
exclude_other_function_from_plot = False
folder = "./Examples/Imposed Yield Stress/Output"
imposed_yield_stress = 130
first_percentage = 42.5 # The first x % of the stress-strain curve used to fit.
file_name = "./Examples/Data/CORRECTED_rectangle_specimen.csv"
skiprows = 1
strain_name = "Déformation interpolée VIC"
force_name = "Charge"
time_column_name = "Temps"
displacement_name = "Déplacement Delta_L interpolé"
surface_0 = 22.66 * 1.18 # mm^2
plastic_threshold_strain = 0.00225 # strain
threshold_adter_luders = 0.0144 # strain
stress_thresh_failure = 350
strain_thresh_failure = 0.2

### OTHER PARAMETERS ###

# Temperature
consider_temperature_variation = False
temperature_variable = "temperature"
temperature_to_plot = 20 # only useful for plotting

# To consider
size_rolling_window_strainDot = 40 # nb of points
offset_plot_in_case_m4 = 25

### CONFIGURATIONS ###
configA_m2 = {"A": [True, 650], "m1": [False, 0], "m2": [True, 0.3], "m3": [False, 0], "m4": [False, 0], "m5": [False, 0], "m7": [False, 0], "m8": [False, 0], "m9": [False, 0]}    # the float value is an estimation of the parameter.
configA_m2_m4 = {"A": [True, 650], "m1": [False, 0], "m2": [True, 0.3], "m3": [False, 0], "m4": [True, 0], "m5": [False, 0], "m7": [False, 0], "m8": [False, 0], "m9": [False, 0]}
configA_m2_m3 = {"A": [True, 650], "m1": [False, 0], "m2": [True, 0.3], "m3": [True, 0], "m4": [False, 0], "m5": [False, 0], "m7": [False, 0], "m8": [False, 0], "m9": [False, 0]}
configA_m2_m7 = {"A": [True, 650], "m1": [False, 0], "m2": [True, 0.3], "m3": [False, 0], "m4": [False, 0], "m5": [False, 0], "m7": [True, 0], "m8": [False, 0], "m9": [False, 0]} # Failed
configA_m2_m3_m4 = {"A": [True, 650], "m1": [False, 0], "m2": [True, 0.3], "m3": [True, 0], "m4": [True, 0], "m5": [False, 0], "m7": [False, 0], "m8": [False, 0], "m9": [False, 0]}

# Configurations to plot
configurations = {"A & m2": configA_m2, "A, m2 & m4": configA_m2_m4} 

### EXECUTION ###
if __name__ == "__main__":
    data = read_data(file_name, strain_name, force_name, surface_0, plastic_threshold_strain, strain_thresh_failure, stress_thresh_failure, threshold_adter_luders, first_percentage, size_rolling_window_strainDot, consider_temperature_variation, temperature_variable, temperature_to_plot, skiprows, time_column_name)
    # plot_stress_vs_strain(file_name, strain_name, force_name, surface_0, skiprows)
    # plot_force_vs_displacement(file_name, displacement_name, force_name, skiprows)
    run_all_configurations(configurations, folder, data, temperature_to_plot, first_percentage, offset_plot_in_case_m4, used_function, file_name, strain_name, force_name, plastic_threshold_strain, surface_0, skiprows, imposed_yield_stress=imposed_yield_stress, exclude_other_function_from_plot=exclude_other_function_from_plot)

