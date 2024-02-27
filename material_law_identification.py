### This code was written by Matthieu Claudon (Mines Paris - PSL) ###

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

def hansel_spittel(data, config, *argv): # ex: hansel_spittel([0.5, 2, 3], {"A": [True, 650], "m1": [False, 0], "m2": [True, 0.3], "m3": [False, 0], "m4": [False, 0], "m5": [False, 0], "m7": [False, 0], "m8": [False, 0], "m9": [False, 0]}, 650, 0.3)
    eps, epsDot, T = data
    HS_par = {}
    used_paremeters = []
    for par_name in list(config.keys()):
        HS_par[par_name] = 0
        if config[par_name][0]:
            used_paremeters.append(par_name)
    for i, par_value in enumerate(argv):
        HS_par[used_paremeters[i]] = par_value
    # print(HS_par["A"])
    return HS_par["A"] * np.exp(HS_par["m1"]*T) * T**HS_par["m9"] * eps**HS_par["m2"] * np.exp(HS_par["m4"]/eps) * (1+eps)**(HS_par["m5"]*T) * np.exp(HS_par["m7"]*eps) * epsDot**(HS_par["m3"]+HS_par["m8"]*T)

def hansel_spittel_for_plotting(eps, epsDot, T, config, parameters):
    HS_par = {}
    used_paremeters = []
    for par_name in list(config.keys()):
        HS_par[par_name] = 0
        if config[par_name][0]:
            used_paremeters.append(par_name)
    for i, par_value in enumerate(parameters):
        HS_par[used_paremeters[i]] = par_value
    return HS_par["A"] * np.exp(HS_par["m1"]*T) * T**HS_par["m9"] * eps**HS_par["m2"] * np.exp(HS_par["m4"]/eps) * (1+eps)**(HS_par["m5"]*T) * np.exp(HS_par["m7"]*eps) * epsDot**(HS_par["m3"]+HS_par["m8"]*T)

def FORGE_hansel_spittel(data, config, *argv):
    eps, epsDot, T = data
    HS_par = {}
    used_paremeters = []
    for par_name in list(config.keys()):
        HS_par[par_name] = 0
        if config[par_name][0]:
            used_paremeters.append(par_name)
    for i, par_value in enumerate(argv):
        HS_par[used_paremeters[i]] = par_value
    eb0 = 0
    if "m4" in used_paremeters and "m2" in used_paremeters:
        eb0 = eb0_FORGE_computation(HS_par["m2"], HS_par["m4"])
    # if "m4" in used_paremeters and "m2" not in used_paremeters:
    else:
        eb0 = 1e-2
    # print("EB0 :", eb0)
    return HS_par["A"] * np.exp(HS_par["m1"]*T) * T**HS_par["m9"] * (eps + eb0)**HS_par["m2"] * np.exp(HS_par["m4"]/(eps+eb0)) * (1+eps+eb0)**(HS_par["m5"]*T) * np.exp(HS_par["m7"]*(eps+eb0)) * epsDot**(HS_par["m3"]+HS_par["m8"]*T)

def FORGE_hansel_spittel_known_yield_stress(data, config, yield_stress, A):
    m2 = compute_m2_from_sy_and_A(yield_stress, A)
    eps, epsDot, T = data
    return A * (eps + 1e-2)**m2

def FORGE_hansel_spittel_for_plotting(eps, epsDot, T, config, parameters): # see C:/Transvalor_Solutions/Forge_NxT_4.0/Resources/Tools/MaterialGenerator/laws_descriptions/hanselspittel.html and documentation part 5 Process Data
    HS_par = {}
    used_paremeters = []
    for par_name in list(config.keys()):
        HS_par[par_name] = 0
        if config[par_name][0]:
            used_paremeters.append(par_name)
    for i, par_value in enumerate(parameters):
        HS_par[used_paremeters[i]] = par_value
    eb0 = 1e-2
    if "m4" in used_paremeters and "m2" in used_paremeters:
        eb0 = eb0_FORGE_computation(HS_par["m2"], HS_par["m4"])
    # print("EB0 :", eb0)
    return HS_par["A"] * np.exp(HS_par["m1"]*T) * T**HS_par["m9"] * (eps + eb0)**HS_par["m2"] * np.exp(HS_par["m4"]/(eps+eb0)) * (1+eps+eb0)**(HS_par["m5"]*T) * np.exp(HS_par["m7"]*(eps+eb0)) * epsDot**(HS_par["m3"]+HS_par["m8"]*T)

def eb0_FORGE_computation(m2, m4): # cf. doc FORGE part5 Process data
    if m2 == 0 or m2 == 1: # pas sûr que FORGE le fasse
        return 1e-2
    eb0 = m4 / m2 * (1 - (1/(1-m2))**0.5)
    if eb0 == 0: # pas sûr non plus
        return 1e-2
    elif eb0 > 0:
        return eb0
    else:
        return max(1e-2, - eb0)


def plot_stress_vs_strain(file_name, strain_name, force_name, surface_0, skiprows):
    df = pd.read_csv(file_name, skiprows=skiprows, sep=";")
    df["stress"] = df[force_name] * np.exp(df[strain_name]) / surface_0
    plt.plot(df[strain_name], df["stress"])
    plt.xlabel("Strain")
    plt.ylabel("Stress (MPa)")
    plt.title("Stress vs. Strain")
    plt.grid()
    plt.show()

def plot_force_vs_displacement(file_name, displacement_name, force_name, skiprows):
    df = pd.read_csv(file_name, skiprows=skiprows, sep=";")
    plt.plot(df[displacement_name], df[force_name])
    plt.xlabel("Displacement")
    plt.ylabel("Force")
    plt.title("Force vs. Displacement")
    plt.grid()
    plt.show()

def elastic_part(name_file, x_name, y_name, surface_0, skiprows, elastic_limit):
    df = pd.read_csv(name_file, skiprows=skiprows, sep=";")
    try:
        df = df.set_index("Index [1]")
    except:
        pass
    df["stress"] = df[y_name] * np.exp(df[x_name]) / surface_0
    df = df[(df[x_name] <= elastic_limit)]
    df[x_name] = df[x_name] - df[x_name].iloc[-1]
    return df

def read_data(name_file, x_name, y_name, surface_0, plastic_threshold_strain, strain_thresh_failure, stress_thresh_failure, threshold_adter_luders, first_percentage, rolling_window_size, consider_temperature_variation, temperature_variable, temperature_to_plot, skiprows, time_column_name):
    df = pd.read_csv(name_file, skiprows=skiprows, sep=";")
    try:
        df = df.set_index("Index [1]")
    except:
        pass
    df["stress"] = df[y_name] * np.exp(df[x_name]) / surface_0
    if not consider_temperature_variation:
        df["temperature"] = temperature_to_plot
        df = df[[time_column_name, x_name, "stress", "temperature"]]
    else:
        df = df[[time_column_name, x_name, "stress", temperature_variable]]
    df.columns = ["Time", "strain", "stress", "temperature"]
    df["strainDot"] = df["strain"].diff() / df["Time"].diff()
    df["strainDot"] = df["strainDot"].rolling(window=rolling_window_size, center=True, min_periods=1).sum() / rolling_window_size
    df = df.set_index("Time")
    df.name = "With Lüders band, with final drop"
    df = df[(df["strain"]>= plastic_threshold_strain)]
    df["strain"] = df["strain"] - df["strain"].iloc[0]
    threshold_adter_luders -= plastic_threshold_strain
    strain_thresh_failure -= plastic_threshold_strain
    df_without_final_drop = df[((df["strain"] < strain_thresh_failure) | (df["stress"] > stress_thresh_failure))].copy()
    df_without_final_drop.name = "With Lüders band, without final drop"
    df_after_luders = df[(df["strain"]>=threshold_adter_luders)]
    df_after_luders.name = "Without Lüders band, with final drop"
    df_after_luders_without_final_drop = df_without_final_drop[(df_without_final_drop["strain"]>=threshold_adter_luders)]
    df_after_luders_without_final_drop.name = "Without Lüders band, without final drop"
    df_beginning = df_without_final_drop[(df_without_final_drop["strain"] <= df_without_final_drop["strain"].max() * first_percentage / 100)]
    df_beginning.name = "First {} %, with Lüders band".format(first_percentage)
    df_beginning_after_luders = df_beginning[(df_beginning["strain"] >= threshold_adter_luders)]
    df_beginning_after_luders.name = "First {} %, without Lüders bands".format(first_percentage)
    data = [df, df_after_luders, df_without_final_drop, df_after_luders_without_final_drop, df_beginning, df_beginning_after_luders]
    return data

def output_configuration(pandas_dataframe, config, used_function, imposed_yield_stress=-1):
    par_to_calibrate = []
    estimated_values = [] # the estimated value of each paremeter
    for (par_name, par_values) in config.items():
        if par_values[0]:
            par_to_calibrate.append(par_name)
            estimated_values.append(par_values[1])
    entries = [pandas_dataframe["strain"], 0, 0] # index 0: strain, index 1: strainDot, index 2: temperature
    if "m3" not in par_to_calibrate and "m8" not in par_to_calibrate: # strainDot variation not taken into account
        entries[1] = np.ones((len(pandas_dataframe),))
    else:
        entries[1] = pandas_dataframe["strainDot"]
    if "m5" not in par_to_calibrate and "m9" not in par_to_calibrate: # T variation not taken into account
        entries[2] = np.ones((len(pandas_dataframe),))
    else:
        entries[2] = pandas_dataframe["temperature"]
    simplified_hansel_function = lambda entries, *argv: used_function(entries, config, *argv)
    res = curve_fit(simplified_hansel_function, entries, pandas_dataframe["stress"], p0=estimated_values)
    if imposed_yield_stress >= 0 and used_function == hansel_spittel:
        del par_to_calibrate[0], estimated_values[0]
        print(par_to_calibrate, estimated_values)
        simplified_hansel_function = lambda entries, *argv: hansel_spittel(entries, config, imposed_yield_stress, *argv)
        res = curve_fit(simplified_hansel_function, entries, pandas_dataframe["stress"], p0=estimated_values)
        print(res)
    if imposed_yield_stress >= 0 and used_function == FORGE_hansel_spittel:
        simplified_hansel_function2 = lambda entries, A: FORGE_hansel_spittel_known_yield_stress(entries, config, imposed_yield_stress, A)
        res = curve_fit(simplified_hansel_function2, entries, pandas_dataframe["stress"], p0=estimated_values[0])
    return {"parameters": res[0], "standard_deviations": np.sqrt(np.diag(res[1]))}

def compute_m2_from_sy_and_A(stress_yield, A):
    return 0.5 * np.log10(A / stress_yield)

def print_parameters_and_std_deviations(par_precision, std_dev_precision, config, par, std_dev, used_function, imposed_yield_stress=-1): # le nombre de chiffres après la virgule dans chacun des cas
    calibrated_par = []
    for par_name in list(config.keys()):
        if config[par_name][0]:
            calibrated_par.append(par_name)
    if imposed_yield_stress > 0:
        if used_function == FORGE_hansel_spittel:
            calibrated_par = ["A"]
        elif used_function == hansel_spittel:
            try:
                calibrated_par.remove("A")
            except: 
                pass
    output_string = ""
    for i, par_name in enumerate(calibrated_par):
        output_string += "{}={}".format(par_name, round(par[i], par_precision))
        if par_name == "A":
            output_string += " MPa"
        if i != len(calibrated_par) - 1:
            output_string += ", "
    if imposed_yield_stress > 0 and used_function == FORGE_hansel_spittel:
        output_string += ", m2={}".format(round(compute_m2_from_sy_and_A(imposed_yield_stress, par[0]), 5))
    output_string += "\nStandard deviations: "
    for i, par_name in enumerate(calibrated_par):
        output_string += "{}".format(round(std_dev[i], std_dev_precision))
        if i != len(calibrated_par) - 1:
            output_string += ", "
    return output_string

def plot_config(saving_folder, data, config, config_name, temperature_to_plot, first_percentage, offset_plot_in_case_m4, used_function, file_name, strain_name, force_name, elastic_strain_limit, surface_0, skiprows, imposed_yield_stress=-1, exclude_other_function_from_plot=False):
    df = data[0]
    df_elastic = elastic_part(file_name, strain_name, force_name, surface_0, skiprows, elastic_strain_limit)
    offset_plot = 0
    if config["m4"][0]:
        offset_plot = offset_plot_in_case_m4
    plt.clf()
    output_configurations = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, dataF in enumerate(data[4:]):
        lgn, col = i // 2 , i % 2
        output_config = output_configuration(dataF, config, used_function, imposed_yield_stress=imposed_yield_stress)
        output_configurations.append(output_config)
        params, std_dev = output_config["parameters"], output_config["standard_deviations"]
        par = []
        for j in range(len(params)):
            par.append(params[j])
        if imposed_yield_stress >= 0 and used_function == FORGE_hansel_spittel:
            par.append(compute_m2_from_sy_and_A(imposed_yield_stress, par[0]))
        """ elif imposed_yield_stress >= 0 and used_function == hansel_spittel:
            par.insert(0, imposed_yield_stress)
            print(par) """
        simplified_hansel_function_for_plotting = lambda strain, strainDot, temperature: hansel_spittel_for_plotting(strain, strainDot, temperature, config, par)
        simplified_FORGE_hansel_function_for_plotting = lambda strain, strainDot, temperature: FORGE_hansel_spittel_for_plotting(strain, strainDot, temperature, config, par)
        axes[lgn, col].plot(dataF["strain"], dataF["stress"], label="Data")
        if not exclude_other_function_from_plot:
            axes[lgn, col].plot(dataF["strain"], simplified_hansel_function_for_plotting(dataF["strain"], dataF["strainDot"], temperature_to_plot * np.ones((len(dataF),))), color="red", label="Hansel & Spittel")
            axes[lgn, col].plot(dataF["strain"], simplified_FORGE_hansel_function_for_plotting(dataF["strain"], dataF["strainDot"], temperature_to_plot * np.ones((len(dataF),))), color="orange", label="Hansel & Spittel FORGE")
        else:
            if used_function == hansel_spittel:
                axes[lgn, col].plot(dataF["strain"], simplified_hansel_function_for_plotting(dataF["strain"], dataF["strainDot"], temperature_to_plot * np.ones((len(dataF),))), color="red", label="Hansel & Spittel", linewidth=2)
            else:
                axes[lgn, col].plot(dataF["strain"], simplified_FORGE_hansel_function_for_plotting(dataF["strain"], dataF["strainDot"], temperature_to_plot * np.ones((len(dataF),))), color="red", label="Hansel & Spittel FORGE", linewidth=2)
        axes[lgn, col].grid()
        axes[lgn, col].set_xlabel("Plastic strain")
        axes[lgn, col].set_ylabel("Stress (MPa)")
        axes[lgn, col].set_title("{}\n{}".format(dataF.name, print_parameters_and_std_deviations(4, 4, config, par, std_dev, used_function, imposed_yield_stress=imposed_yield_stress)))
        axes[lgn, col].legend()
        if i == 0:
            axes[lgn + 1, col].plot(df_elastic[strain_name], df_elastic["stress"], color="blue")
            axes[lgn + 1, col].plot(df["strain"], df["stress"], label="Data", color="blue")
            axes[lgn + 1, col].plot(dataF["strain"], dataF["stress"], label="Data {} %".format(first_percentage), color="green", linewidth=3)
            if not exclude_other_function_from_plot:
                axes[lgn + 1, col].plot(df["strain"].iloc[offset_plot:], simplified_hansel_function_for_plotting(df["strain"].iloc[offset_plot:], df["strainDot"].iloc[offset_plot:], temperature_to_plot * np.ones((len(df)-offset_plot,))), color="red", label="Hansel & Spittel")
                axes[lgn + 1, col].plot(df["strain"], simplified_FORGE_hansel_function_for_plotting(df["strain"], df["strainDot"], temperature_to_plot * np.ones((len(df),))), color="orange", label="Hansel & Spittel FORGE")
            else:
                if used_function == hansel_spittel:
                    axes[lgn + 1, col].plot(df["strain"].iloc[offset_plot:], simplified_hansel_function_for_plotting(df["strain"].iloc[offset_plot:], df["strainDot"].iloc[offset_plot:], temperature_to_plot * np.ones((len(df)-offset_plot,))), color="red", label="Hansel & Spittel", linewidth=2)
                else:
                    axes[lgn + 1, col].plot(df["strain"], simplified_FORGE_hansel_function_for_plotting(df["strain"], df["strainDot"], temperature_to_plot * np.ones((len(df),))), color="red", label="Hansel & Spittel FORGE", linewidth=2)
        if i == 1:
            axes[lgn + 1, col].plot(df_elastic[strain_name], df_elastic["stress"], color="blue")
            axes[lgn + 1, col].plot(df["strain"], df["stress"], label="Data", color="blue")
            axes[lgn + 1, col].plot(dataF["strain"], dataF["stress"], label="Data {} %".format(first_percentage), color="green", linewidth=3)
            if not exclude_other_function_from_plot:
                axes[lgn + 1, col].plot(df["strain"].iloc[offset_plot:], simplified_hansel_function_for_plotting(df["strain"].iloc[offset_plot:], df["strainDot"].iloc[offset_plot:], temperature_to_plot * np.ones((len(df)-offset_plot,))), color="red", label="Hansel & Spittel")
                axes[lgn + 1, col].plot(df["strain"], simplified_FORGE_hansel_function_for_plotting(df["strain"], df["strainDot"], temperature_to_plot * np.ones((len(df),))), color="orange", label="Hansel & Spittel FORGE")            
            else:
                if used_function == hansel_spittel:
                    axes[lgn + 1, col].plot(df["strain"].iloc[offset_plot:], simplified_hansel_function_for_plotting(df["strain"].iloc[offset_plot:], df["strainDot"].iloc[offset_plot:], temperature_to_plot * np.ones((len(df)-offset_plot,))), color="red", label="Hansel & Spittel", linewidth=2)
                else:
                    axes[lgn + 1, col].plot(df["strain"], simplified_FORGE_hansel_function_for_plotting(df["strain"], df["strainDot"], temperature_to_plot * np.ones((len(df),))), color="red", label="Hansel & Spittel FORGE", linewidth=2)            
        axes[lgn + 1, col].grid()
        axes[lgn + 1, col].set_xlabel("Plastic strain")
        axes[lgn + 1, col].set_ylabel("Stress (MPa)")
        axes[lgn + 1, col].set_title("{}\n{}".format(dataF.name, print_parameters_and_std_deviations(4, 4, config, par, std_dev, used_function, imposed_yield_stress=imposed_yield_stress)))
        axes[lgn + 1, col].legend()
    title = config_name
    if imposed_yield_stress > 0:
        title += ", Yield Stress = {} MPa".format(round(imposed_yield_stress, 5))
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(saving_folder + "/{}.png".format(config_name), dpi=400)
    simplified_hansel_function_for_plotting2 = lambda dataframe: hansel_spittel_for_plotting(dataframe["strain"], dataframe["strainDot"], temperature_to_plot * np.ones((len(dataframe),)), config, output_configurations[0]["parameters"])
    simplified_FORGE_hansel_function_for_plotting2 = lambda dataframe: FORGE_hansel_spittel_for_plotting(dataframe["strain"], dataframe["strainDot"], temperature_to_plot * np.ones((len(dataframe),)), config, output_configurations[0]["parameters"])
    simplified_hansel_function_for_plotting2_WITHOUT_LUDERS = lambda dataframe: hansel_spittel_for_plotting(dataframe["strain"], dataframe["strainDot"], temperature_to_plot * np.ones((len(dataframe),)), config, output_configurations[1]["parameters"])
    simplified_FORGE_hansel_function_for_plotting2_WITHOUT_LUDERS = lambda dataframe: FORGE_hansel_spittel_for_plotting(dataframe["strain"], dataframe["strainDot"], temperature_to_plot * np.ones((len(dataframe),)), config, output_configurations[1]["parameters"])
    zoom = simplified_hansel_function_for_plotting2(data[4])
    FORGE_zoom = simplified_FORGE_hansel_function_for_plotting2(data[4])
    global_view = simplified_hansel_function_for_plotting2(data[0])
    FORGE_global_view = simplified_FORGE_hansel_function_for_plotting2(data[0])
    zoom_without_luders = simplified_hansel_function_for_plotting2_WITHOUT_LUDERS(data[5])
    FORGE_zoom_without_luders = simplified_FORGE_hansel_function_for_plotting2_WITHOUT_LUDERS(data[5])
    global_view_without_luders = simplified_hansel_function_for_plotting2_WITHOUT_LUDERS(data[0])
    FORGE_global_view_without_luders = simplified_FORGE_hansel_function_for_plotting2_WITHOUT_LUDERS(data[0])
    return zoom, global_view, zoom_without_luders, global_view_without_luders, FORGE_zoom, FORGE_global_view, FORGE_zoom_without_luders, FORGE_global_view_without_luders

def plot_comparison(data, curves, mode, configs, saving_folder, offset_plot_in_case_m4):
    plt.clf()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i in range(4):
        dataF = data[0]
        title = "With Lüders Band - Global View"
        if i == 0:
            dataF = data[4]
            title = "With Lüders Band - Zoom"
        if i == 2:
            dataF = data[5]
            title = "Without Lüders Bands - Zoom"
        if i == 3:
            title = "Without Lüders Bands - Global View"
        lgn, col = i % 2, i // 2
        category = curves[i]
        axes[lgn, col].plot(dataF["strain"], dataF["stress"], label="Data")
        for config_name in list(category.keys()):
            if mode == 1 and configs[config_name]["m4"][0]:
                axes[lgn, col].plot(dataF["strain"].iloc[offset_plot_in_case_m4:], category[config_name].iloc[offset_plot_in_case_m4:], label=config_name)
            else:
                axes[lgn, col].plot(dataF["strain"], category[config_name], label=config_name)
        axes[lgn, col].grid()
        axes[lgn, col].legend()
        axes[lgn, col].set_xlabel("Plastic strain")
        axes[lgn, col].set_ylabel("Stress (MPa)")
        axes[lgn, col].set_title(title)
    if mode == 1:
        fig.suptitle("Comparison - ORIGINAL Hansel & Spittel")
        fig.tight_layout()
        plt.savefig(saving_folder + "/Comparison_H&S.png", dpi=400)
    if mode == 2:
        fig.suptitle("Comparison - FORGE Hansel & Spittel")
        fig.tight_layout()
        plt.savefig(saving_folder + "/Comparison_FORGE_H&S.png", dpi=400)

def run_all_configurations(configs, saving_folder, data, temperature_to_plot, first_percentage, offset_plot_in_case_m4, used_function, file_name, strain_name, force_name, elastic_strain_limit, surface_0, skiprows, imposed_yield_stress=-1, exclude_other_function_from_plot=False):
    if os.path.exists(saving_folder):
        print("Error: folder '{}' already exists".format(saving_folder))
        return
    os.mkdir(saving_folder)
    zoom, global_view, zoom_without_luders, global_view_without_luders, FORGE_zoom, FORGE_global_view, FORGE_zoom_without_luders, FORGE_global_view_without_luders = {}, {}, {}, {}, {}, {}, {}, {}
    if imposed_yield_stress < 0:
        for config_name in list(configs.keys()):
            res = plot_config(saving_folder, data, configs[config_name], config_name, temperature_to_plot, first_percentage, offset_plot_in_case_m4, used_function, file_name, strain_name, force_name, elastic_strain_limit, surface_0, skiprows, imposed_yield_stress=imposed_yield_stress, exclude_other_function_from_plot=exclude_other_function_from_plot)
            zoom[config_name], global_view[config_name], zoom_without_luders[config_name], global_view_without_luders[config_name], FORGE_zoom[config_name], FORGE_global_view[config_name], FORGE_zoom_without_luders[config_name], FORGE_global_view_without_luders[config_name] = res
        comparison_natural_cuves = [zoom, global_view, zoom_without_luders, global_view_without_luders]
        comparison_FORGE_curves = [FORGE_zoom, FORGE_global_view, FORGE_zoom_without_luders, FORGE_global_view_without_luders]
        plot_comparison(data, curves=comparison_natural_cuves, mode=1, configs=configs, saving_folder=saving_folder, offset_plot_in_case_m4=offset_plot_in_case_m4)
        plot_comparison(data, curves=comparison_FORGE_curves, mode=2, configs=configs, saving_folder=saving_folder, offset_plot_in_case_m4=offset_plot_in_case_m4)
    elif imposed_yield_stress >= 0 and used_function == hansel_spittel:
        print("Error: Yield Stress cannot be imposed with Original Hansel & Spittel formulation.")
    else: # imposed yield stress & FORGE Hansel and Spittel
        plot_config(saving_folder, data, configs["A & m2"], "A & m2", temperature_to_plot, first_percentage, offset_plot_in_case_m4, used_function, file_name, strain_name, force_name, elastic_strain_limit, surface_0, skiprows, imposed_yield_stress=imposed_yield_stress, exclude_other_function_from_plot=exclude_other_function_from_plot)