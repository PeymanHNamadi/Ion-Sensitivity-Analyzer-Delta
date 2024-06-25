# %%
import numpy as np
import pandas as pd
import panel as pn
import joblib
import keras
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import hvplot.pandas
hv.extension('bokeh', 'matplotlib')

file_paths = {
    "Sea_low": "inputs/Sea_low.csv",
    "Sea_high": "inputs/Sea_high.csv",
    "SJ": "inputs/SJ.csv",
    "IND": "inputs/IND.csv"
}
Sea_l = pd.read_csv(file_paths["Sea_low"])
Sea_h = pd.read_csv(file_paths["Sea_high"])
SJ = pd.read_csv(file_paths["SJ"])
IND = pd.read_csv(file_paths["IND"])

# Load models
models = {}
for ion in ['Alkalinity', 'Br', 'Ca', 'Cl', 'K', 'Mg', 'Na', 'SO4', 'TDS']:
    models[ion] = {
        'ANN': keras.models.load_model(f'Models/ANN_{ion}.h5'),
        'RT': joblib.load(filename=f'Models/RT_{ion}.pkl'),
        'RF': joblib.load(filename=f'Models/RF_{ion}.pkl'),
        'GB': joblib.load(filename=f'Models/GB_{ion}.pkl')
    }

    

def Ion_Simulator_TT(EC, Sacramento_X2, ion, WYT, month):
    # Convert Month name to numerical representation
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    month = month_names.index(month) + 1  # Convert month name to number

    results = {}

    for region in ["OMR", "SJRcorridor", "SouthDelta"]:
        # Define conditions for SEA, IND, and SJR
        if Sacramento_X2 < 81:
            sea_conditions = (
                (region == "OMR" and WYT in ["W", "AN"] and month in [1, 9, 10, 11, 12]) or
                (region == "OMR" and WYT in ["BN", "D"] and month in [1, 2, 8, 9, 10, 11, 12]) or
                (region == "OMR" and WYT == "C" and month in [1, 2, 6, 7, 8, 9, 10, 11, 12])
            )
            ind_conditions = (
                (region == "SJRcorridor" and WYT == "D" and month in [6, 7, 8, 9, 10, 11]) or
                (region == "SJRcorridor" and WYT == "C" and month in [6, 7, 8, 9, 10, 11]) or
                (region == "SouthDelta" and WYT in ["AN", "BN", "D"] and month in [1, 2, 8, 9, 10, 11, 12]) or
                (region == "SouthDelta" and WYT == "C" and month in [1, 2, 6, 7, 8, 9, 10, 11, 12])
            )
        else:
            sea_conditions = (
                (region == "OMR" and WYT in ["W", "AN"] and month in [1, 9, 10, 11, 12]) or
                (region == "OMR" and WYT in ["BN", "D", "C"] and month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            )
            ind_conditions = (
                (region == "SouthDelta" and WYT in ["BN", "D"] and month in [1, 2, 8, 9, 10, 11, 12]) or
                (region == "SouthDelta" and WYT == "C" and month in [1, 2, 6, 7, 8, 9, 10, 11, 12])
            )

        if sea_conditions:
            coeffs = Sea_l if EC < 250 else Sea_h
        elif ind_conditions:
            coeffs = IND
        else:
            coeffs = SJ

        coeffs_row = coeffs[coeffs['ion'] == ion]

        
        if sea_conditions:  # Use the complex formula for SEA and IND
            simulated = coeffs_row['K1'] + coeffs_row['K2']*np.sqrt(EC) + coeffs_row['K3']*EC + coeffs_row['K4']*EC**1.5 + coeffs_row['K5']*EC**2 + coeffs_row['K6']*EC**2.5
        else:  # Use the simpler formula for SJR
            simulated = coeffs_row['a']*EC**2 + coeffs_row['b']*EC + coeffs_row['c']

        results[region] = simulated.iloc[0]  # Assuming only one row matches

    return results


def Ion_Simulator(EC, Sacramento_X2, Ion, WYT, month):
    regions = ["OMR", "SJRcorridor", "SouthDelta"]  
    results = [] 

    for Region in regions:
        X = pd.DataFrame(columns=['EC', 'Sacramento X2', 'AN', 'BN', 'C', 'D', 'W', 'OMR', 'SJRcorridor', 'SouthDelta', 'April', 'August', 'December', 'February', 'January', 'July', 'June', 'March', 'May', 'November', 'October', 'September'])

        X.loc[0, "EC"] = (EC - 50) / (3500 - 50)
        X.loc[0, 'Sacramento X2'] = Sacramento_X2 / 100
        X.loc[0, 'AN'] = 1 if WYT == "AN" else 0
        X.loc[0, 'BN'] = 1 if WYT == "BN" else 0
        X.loc[0, 'C'] = 1 if WYT == "C" else 0
        X.loc[0, 'D'] = 1 if WYT == "D" else 0
        X.loc[0, 'W'] = 1 if WYT == "W" else 0

        X.loc[0, 'OMR'] = 1 if Region == "OMR" else 0
        X.loc[0, 'SJRcorridor'] = 1 if Region == "SJRcorridor" else 0
        X.loc[0, 'SouthDelta'] = 1 if Region == "SouthDelta" else 0

        X.loc[0,'April']=1 if month=="April" else 0
        X.loc[0,'August']=1 if month=="August" else 0
        X.loc[0,'December']=1 if month=="December" else 0
        X.loc[0,'February']=1 if month=="February" else 0
        X.loc[0,'January']=1 if month=="January" else 0
        X.loc[0,'July']=1 if month=="July" else 0
        X.loc[0,'June']=1 if month=="June" else 0
        X.loc[0,'March']=1 if month=="March" else 0
        X.loc[0,'May']=1 if month=="May" else 0
        X.loc[0,'November']=1 if month=="November" else 0
        X.loc[0,'October']=1 if month=="October" else 0
        X.loc[0,'September']=1 if month=="September" else 0

        X1 = X.astype(float)


        df = pd.DataFrame(columns=['Region', 'RT', 'GB', 'RF', 'ANN'])
        df.loc[0, 'Region'] = Region
        df.loc[0, 'RT'] = round(models[Ion]['RT'].predict(X1)[0], 2)
        df.loc[0, 'GB'] = round(models[Ion]['GB'].predict(X1)[0], 2)
        df.loc[0, 'RF'] = round(models[Ion]['RF'].predict(X1)[0], 2)
        df.loc[0, 'ANN'] = round(models[Ion]['ANN'].predict(X1, verbose=0)[0][0].astype(float), 2)

        results.append(df)

    final_results = pd.concat(results, ignore_index=True)
    return final_results

def combined_Ion_Simulation(EC, Sacramento_X2, Ion, WYT, month):
    ml_results = Ion_Simulator(EC, Sacramento_X2, Ion, WYT, month)
    tt_results = Ion_Simulator_TT(EC, Sacramento_X2, Ion, WYT, month)
    combined_data = []
    # machine learning + TT
    for index, row in ml_results.iterrows():
        region = row['Region']
        combined_data.append({
            'Region': region,
            'RT': row['RT'],
            'GB': row['GB'],
            'RF': row['RF'],
            'ANN': row['ANN'],
            'TT': tt_results[region]
        })
    
    combined_results = pd.DataFrame(combined_data)
    return combined_results
  

# Define custom widget styles
custom_style1 = {'width': '500px'}

# Define widgets for user input
x_axis = pn.widgets.Select(name='X-axis', options=['EC', 'Sacramento_X2', 'WYT', 'Month'], value='EC', styles=custom_style1)
regions = pn.widgets.MultiSelect(name='Regions', options=['SouthDelta', 'OMR', 'SJRcorridor'], value=['SouthDelta'], styles=custom_style1)
Ion = pn.widgets.Select(name='Ion', options=['Alkalinity', 'Br', 'Ca', 'Cl', 'K', 'Mg', 'Na', 'SO4', 'TDS'], value='Br', styles=custom_style1)
Sacramento_X2 = pn.widgets.FloatSlider(name='Sacramento_X2', start=0, end=100, value=90, styles=custom_style1)
WYT = pn.widgets.Select(name='Water Year Type (WYT)', options=['AN', 'BN', 'C', 'D', 'W'], value='AN', styles=custom_style1)
month = pn.widgets.Select(name='Month', options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], value='September', styles=custom_style1)
EC = pn.widgets.IntSlider(name='EC', start=100, end=2000, step=100, value=1000, styles=custom_style1)

@pn.cache
def simulate_ion_levels(x_axis, ion, wyt, month, sacramento_x2, ec):
    results = []

    if x_axis == 'EC':
        for ec_value in range(100, 3001, 200):  # Increased step size
            result = combined_Ion_Simulation(ec_value, sacramento_x2, ion, wyt, month)
            result['EC'] = ec_value
            results.append(result)
    elif x_axis == 'Sacramento_X2':
        for sac_x2_value in range(30, 101, 20):  # Increased step size
            result = combined_Ion_Simulation(ec, sac_x2_value, ion, wyt, month)
            result['Sacramento_X2'] = sac_x2_value
            results.append(result)
    elif x_axis == 'WYT':
        for wyt_value in ['AN', 'BN', 'C', 'D', 'W']:
            result = combined_Ion_Simulation(ec, sacramento_x2, ion, wyt_value, month)
            result['WYT'] = wyt_value
            results.append(result)
    else:  # x_axis == 'Month'
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        for month_value in months:
            result = combined_Ion_Simulation(ec, sacramento_x2, ion, wyt, month_value)
            result['Month'] = month_value
            results.append(result)

    return pd.concat(results, ignore_index=True)

# Define a function to update the plot based on user input
@pn.depends(x_axis, regions, Ion, Sacramento_X2, WYT, month, EC)
def update_plot(x_axis, regions, ion, sacramento_x2, wyt, month, ec):
    data = simulate_ion_levels(x_axis, ion, wyt, month, sacramento_x2, ec)
    data = data[data['Region'].isin(regions)]

    if x_axis in ['WYT', 'Month']:
        plot = data.hvplot.bar(x_axis, ['RT', 'GB', 'RF', 'ANN', 'TT'], xlabel=x_axis, ylabel=f'{ion} [mg/L]', rot=90, height=600, width=1200)
    else:
        plot = data.hvplot.line(x_axis, ['RT', 'GB', 'RF', 'ANN', 'TT'], by='Region', height=500, ylabel=f'{ion} [mg/L]')

    return plot

# disable the parameter widget based on the selected x-axis
def disable_parameter_widget(x_axis):
    EC.disabled = (x_axis == 'EC')
    Sacramento_X2.disabled = (x_axis == 'Sacramento_X2')
    WYT.disabled = (x_axis == 'WYT')
    month.disabled = (x_axis == 'Month')

# Dashboard
# dashboard = pn.Column(
#     pn.Row(x_axis, regions),
#     pn.Row(Ion, Sacramento_X2, WYT, month, EC),
#     update_plot
# )


# x_axis.param.watch(lambda event: disable_parameter_widget(event.new), 'value')

# # Serve the dashboard
# dashboard.servable()

# Define logo and study area image
logo = pn.pane.PNG('Logo.png', width=551, height=91, sizing_mode='fixed')

fig1_title = pn.pane.Markdown("### Study Area", width=605, align='center')
spacer_left = pn.Spacer(width=250)  # Adjust width as needed
spacer_right = pn.Spacer(width=250)  # Adjust width as needed
fig1 = pn.pane.PNG('Fig1.png', width=605, height=753, sizing_mode='fixed')
fig1_with_title = pn.Column(pn.Row(spacer_left, fig1_title, spacer_right), fig1)

# Define notes
notes = pn.pane.Markdown("""
### Notes:

- Electrical conductivity (EC) is measured in microsiemens per centimeter (µS/cm).
- Sacramento_X2 is the percentage of Sacramento River flow that is estimated to reach the Delta.
The exact location of the Sacramento X2 point is determined by the California Department of Water Resources (DWR)
 based on the specific hydraulic conditions and water flows in the Sacramento River. 
The DWR uses a combination of hydrological models, flow measurements, and other data to determine the location of the Sacramento X2 point.
- The Water Year Type (WYT) is a classification of the water year based on its hydrological characteristics.
Water Year Type that includes the following categories: 1- Wet (W), 2- Critical (C), 3- Dry (D), 4- Above-Normal (AN), 5- Below-Normal (BN)
- Region refers to monitoring regions that includes: 1- Old-Middle River (OMR), 2- San Joaquin River Corridor (SJRcorridor), and 3- South Delta (SouthDelta).
- Month refers to the month of the year.
- Prediction models: Regression Trees: RT, Gradient Boosting: GB, Random Forest: RF, Artificial Neural Networks: ANN, Parametric Regression method prepared by Tetra Tech: TT
                                    
""")

# Define references and disclaimer
references = pn.pane.Markdown("""
### References:
- Namadi, P., He, M. & Sandhu, P. Salinity-constituent conversion in South Sacramento-San Joaquin Delta of California via machine learning. 
                                       Earth Sci Inform 15, 1749–1764 (2022). [https://doi.org/10.1007/s12145-022-00828-1](https://doi.org/10.1007/s12145-022-00828-1)

- Namadi, P., He, M. & Sandhu, P. Modeling ion constituents in the Sacramento-San Joaquin Delta using multiple machine learning approaches. 
                                       Journal of Hydroinformatics (2023). [https://doi.org/10.2166/hydro.2023.158](https://doi.org/10.2166/hydro.2023.158)

- Paul Hutton, Arushi Sinha, and Sujoy Roy. 2022. Simplified Approach for Estimating Salinity Constituent Concentrations in the San Francisco Estuary & Sacramento-San Joaquin River Delta. 
                                       [Report's link](https://rtdf.info/public_docs/Miscellaneous%20RTDF%20Web%20Page%20Information/Other%20MWQP%20and%20DWR%20Publications/2022-07-21%20MWQI%20Conservative%20Constituents%20User%20Guide_formatted.pdf)

                             
### Disclaimer: this dashboard is still in beta.

Thank you for evaluating the Sensitivity tool for Ion Simulator.  

If you have feedback, suggestions or questions, please contact Peyman Namadi or Nicky Sandhu (Peyman.Hosseinzadehnamadi@Water.ca.gov  or  prabhjot.sandhu@water.ca.gov)
""")

# Dashboard
dashboard = pn.Column(
    logo,
    notes,
    pn.Row(x_axis, regions),
    pn.Row(Ion, Sacramento_X2, WYT, month, EC),
    update_plot,
    fig1_with_title,
    references
)

x_axis.param.watch(lambda event: disable_parameter_widget(event.new), 'value')

# Serve the dashboard using the built-in Python HTTP server
dashboard.show(port=5006)






