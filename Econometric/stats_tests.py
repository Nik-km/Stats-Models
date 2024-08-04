# -*- coding: utf-8 -*-
"""
Created on Sat Aug 3 13:06:56 2024
@author: M
"""

#%% Preliminaries ---------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.stattools as smt

# Set path to working directory of this script file
path_file = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_file)

# Import data
df = pd.read_csv("../.data/full_data.csv", index_col='Date')

# Build Model
X = df[['Mkt-RF', 'SMB', 'HML']]    # Define the independent variables (Fama-French factors)
X = sm.add_constant(X)  # Add a constant (intercept) to the model
y = df['excess_return'] # Define the dependent variable (Excess Return)
model = sm.OLS(y, X).fit()


#%% Stats Tests -----------------------------------------------------------------------------------

#>> Testing Multicollinearity using VIF
def calc_VIF(df_exog):
    '''
    Parameters
    ----------
    df_exog : dataframe, (n_obs, k_vars)
        Design matrix with all explanatory variables, as for example used in regression.

    Returns
    -------
    VIF : Series
        Variance inflation factors
    '''
    df_exog = sm.add_constant(df_exog)
    mod = [sm.OLS(df_exog[col].values, df_exog.loc[:, df_exog.columns != col].values) for col in df_exog]
    r_sq = mod.fit().rsquared

    vifs = pd.Series(
        [1 / (1 - r_sq)],
        index=df_exog.columns,
        name='VIF'
    )
    return vifs

calc_VIF(X)


#>> Other normality tests
tests = sm.stats.stattools.jarque_bera(model.resid)
print(pd.Series(tests, index=['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']))



#%% Model Diagnostics -----------------------------------------------------------------------------
# Residual analysis section

#>> QQ Plot
# with plt.style.context('ggplot'):
#     plt.rc('figure', figsize=(6,6))
#     fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)
#     fig.suptitle("Normal QQ Plot")
#     plt.show()

def gen_QQ_plot(mod_res):
    df_res = pd.DataFrame(sorted(mod_res), columns=['residual'])
    # Calculate the Z-score for the residuals
    df_res['z_actual'] = (df_res['residual'].map(lambda x: (x - df_res['residual'].mean()) / df_res['residual'].std()))
    # Calculate the theoretical Z-scores
    df_res['rank'] = df_res.index + 1
    df_res['percentile'] = df_res['rank'].map(lambda x: x/len(df_res.residual))
    df_res['theoretical'] = stats.norm.ppf(df_res['percentile'])
    # Construct QQ plot
    with plt.style.context('ggplot'):
        plt.figure(figsize=(9,9))
        plt.scatter(df_res['theoretical'], df_res['z_actual'], color='blue')
        plt.xlabel('Theoretical Quantile')
        plt.ylabel('Sample Quantile')
        plt.title('Normal QQ Plot')
        plt.plot(df_res['theoretical'], df_res['theoretical'])
        plt.gca().set_facecolor('white')    # (0.95, 0.95, 0.95)
        plt.gca().spines['top'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.savefig(path_file + "\\output\\QQ_plot.png")
        plt.show()
    return(df_res)

gen_QQ_plot(model.resid)

#>> Partial Regression Plots
# Residual plots
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
fig.savefig(path_file + "\\output\\partial_reg_plots.png")
fig

#>> Residuals vs. Fitted Plot
with plt.style.context('ggplot'):
    plt.figure(figsize=(9,9))
    plt.scatter(model.fittedvalues, model.resid, color='orange')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('Residual by Predicted')
    plt.axhline(y = 0, color = 'black', linestyle = '-') 
    plt.gca().set_facecolor('white')    # (0.95, 0.95, 0.95)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.savefig(path_file + "\\output\\fitted_res_plot.png")
    plt.show()

#>> Scale-Location Plot
with plt.style.context('ggplot'):
    plt.figure(figsize=(9,9))
    plt.scatter(model.fittedvalues, np.sqrt(model.resid), color='orange')
    plt.xlabel('Predicted Values')
    plt.ylabel('Standardized Residuals')
    plt.title('Scale-Location')
    plt.gca().set_facecolor('white')    # (0.95, 0.95, 0.95)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.savefig(path_file + "\\output\\scale_location.png")
    plt.show()


#%% Other -----------------------------------------------------------------------------------------
model[0].eigenvals
model[0].llf  # Log-likelihood of model
model[0].aic
model[0].resid.kurtosis
model[0].kurtosis

smt.jarque_bera(model[0].resid)[3]

model[0].summary2()
model[0].centered_tss
model[0].summary()

model.fittedvalues
res = model.resid
