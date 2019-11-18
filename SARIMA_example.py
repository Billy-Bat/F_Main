import warnings
import sys
warnings.filterwarnings("ignore")
## PANDAS import
import pandas as pd
from pandas import read_csv
from pandas.plotting import lag_plot, autocorrelation_plot

# STATSMODELS import
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.gofplots import qqplot
import statsmodels.graphics.tsaplots as tsa

# MATPLOTLIB import
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["text.color"] = 'k'
matplotlib.rcParams["font.size"] = 5
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams["lines.linewidth"] = 0.5
matplotlib.rcParams["figure.subplot.hspace"] = 0.5
matplotlib.rcParams["axes.labelsize"] = 2
matplotlib.rcParams["lines.markersize"] = 0.5
matplotlib.rcParams['figure.figsize'] = 15, 7

import itertools
from sklearn.metrics import mean_squared_error


#################CUSTOM FUNCTION#####################################
"""
This code gives the plot necessary for model identification (see plot_Model_Identify)
It enables for Grid Search of SARIMA parameters with AIC and BIC criterion (see Get_Config and main code)
It also enables for model evaluation using MSE criterion.


Ref : How to Grid Search SARIMA Hyperparameters for Time Series Forecasting
      Deep Learning for Times Series
Author : Jason Brownlee

Ref : An End-to-End Project on Time Series Analysis and Forecasting with Python
Author : Susan Li

Author : GG
"""

def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model_fit = model.fit(disp=False)
    # make one step Forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]

def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

def Get_Config(pdq_range, PDQ_range, Seasonal):
    p = range(0, pdq_range[0]); d = range(0, pdq_range[1]); q = range(0, pdq_range[2])
    pdq = list(itertools.product(p, d, q))
    PDQ = []
    P = range(0, PDQ_range[0]); D = range(0, PDQ_range[1]); Q = range(0, PDQ_range[2])
    for Season_Period in Seasonal :
        PDQ = PDQ + [(x[0], x[1], x[2], Season_Period) for x in list(itertools.product(P, D, Q))]

    return pdq, PDQ

def evaluate_model(Train_set, Test_set, pdq_val, PDQ_val, _print=True, _plot=True) :
    predictions = []
    history = [int(x) for x in Train_set.values]
    for step in range(len(Test_set)) :
        mod = sm.tsa.statespace.SARIMAX(history,
                                        order=pdq_val,
                                        seasonal_order=PDQ_val,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False,
                                        disp=False)
        mod_fit = mod.fit(disp=0)
        yhat = mod_fit.forecast()[0]
        predictions.append(yhat)
        history.append(Test_set.iloc[step,0])

    error = mean_squared_error(Test_set.iloc[:,0], predictions)

    if _print :
        print('Config{}{} MSE : {}'.format(pdq_val, PDQ_val, error))

    if _plot == True :
        Actual = pd.concat([Train_set, Test_set])
        ax = Actual.plot()
        pd.DataFrame(predictions ,index=Test_set.index).plot(ax=ax)
        ax.legend(['Actual', 'Rolling forcast'])
        plt.show()


    return error

def evaluate_models(Train_set, Test_set, pdqPDQ_vals, _print=True): # TAKEN
    best_score, best_cfg = float('inf'), None
    for pdq_PDQ in pdqPDQ_vals :
            try :
                mse = evaluate_model(Train_set, Test_set, pdq_PDQ[0], pdq_PDQ[1], _print=False)
                if mse < best_score:
                    best_score, best_cfg = mse, (pdq_PDQ[0], pdq_PDQ[1])
                if _print==True :
                    print('SARIMA{}{} : MSE : {}'.format(pdq_PDQ[0], pdq_PDQ[1], mse))
            except:
                print('WARNING : Evaluation Failed')
                print(sys.exc_info()[0])
                continue
    print('Best SARIMA{} MSE : {}'.format(best_cfg, best_score))

    return best_cfg, best_score


######################################################################

def SampleSeasonality(DataSet, frequency, _plot=False) :
    resample = DataSet.resample(frequency)
    mean = resample.mean()


    if _plot == True :
        fig = mean.plot()
        fig.set_title('seasonality sampled with {}'.format(frequency))
        plt.show()

    return mean

def plot_Model_Identify(DataSet, frequency=1, acf_lag=12, pacf_lag=12):
    """
    DataSet : dataframe with the type of first column either int()
    or panda datetime
    Frequency : int, Seasonal Component period (in time step)
    """
    # Organize plot
    fig, ax = plt.subplots(3, 4)

    # Plot the Observed Data
    DataSet.plot(ax=ax[0,0])
    ax[0, 0].set_title('Observed Value')
    ax[0, 0].set_xlabel("")

    # Plot the autocorrelation plot
    autocorrelation_plot(DataSet.iloc[:,0], ax=ax[1,0])
    ax[1, 0].set_title('Autocorrelation')

    # Plot the QQ plot
    qqplot(DataSet.iloc[:,0], ax=ax[2,0], )
    ax[2, 0].set_title('Q-Q Plot')

    # Lag plot
    lag_plot(DataSet.iloc[:,0], ax=ax[0, 1])
    ax[0, 1].set_title('Lag Plot')
    ax[0, 1].set_ylabel("")
    ax[0, 1].set_xlabel("")

    # ACF Plot
    tsa.plot_acf(DataSet.iloc[:,0], ax=ax[1, 1], lags=acf_lag, alpha=0.05)
    ax[1, 1].set_title('ACF')

    # PACF Plot
    tsa.plot_pacf(DataSet.iloc[:,0], ax=ax[2, 1], lags=pacf_lag, alpha=0.05)
    ax[2, 1].set_title('PACF')

    # decomposition plot
    decomposition = sm.tsa.seasonal_decompose(DataSet.iloc[:,0], freq=frequency)
    decomposition.resid.plot(ax=ax[0,2])
    decomposition.resid.plot(ax=ax[0,3], kind='kde')
    decomposition.seasonal.plot(ax=ax[1,2])
    decomposition.trend.plot(ax=ax[2,2])
    ax[0, 2].set_title('Residual')
    ax[1, 2].set_title('Seasonal')
    ax[2, 2].set_title('Trend')
    ax[0, 3].set_title('Residual Prob. Distrib')

    plt.show()


############################MAIN CODE##########################################

if __name__ == '__main__' :
    # Extract dataSet, convert dates to the correct format, divide dataSet
    df = read_csv('SARIMA/Dataset.csv', usecols=["Temp", "Date"])
    df.index = pd.to_datetime(df.index, unit='d')

    # Dry Up the DataSet (MONTHLY)
    df = SampleSeasonality(df, 'M', _plot=False)

    # Divide the Set
    Training_set, Test_set = train_test_split(df, 12)

    # # Plot model identification
    # plot_Model_Identify(Training_set, frequency=12)

    # # SARIMAX Config parameters
    # pdq, PDQ = Get_Config([5, 2, 2], [3, 3, 3], [12])
    #
    # min_BIC = float('inf') # quite arbitrary
    # min_AIC = float('inf')
    # top_BICParam = []
    # top_AICParam = []
    #
    # for pdq_param in pdq :
    #     for PDQ_param in PDQ :
    #         mod = sm.tsa.statespace.SARIMAX(Training_set['Temp'],
    #                                                 order=pdq_param,
    #                                                 seasonal_order=PDQ_param,
    #                                                 enforce_stationarity=False,
    #                                                 enforce_invertibility=False,
    #                                                 disp=False)
    #         result = mod.fit(disp=False)
    #         print('Model pdq :{}, PDQ :{} AIC : {}'.format(pdq_param, PDQ_param, result.bic))
    #
    #
    #         if result.bic < min_BIC :
    #             min_BIC = result.bic
    #             top_BICParam = [pdq_param, PDQ_param]
    #         if result.aic < min_AIC :
    #             min_AIC = result.aic
    #             top_AICParam = [pdq_param, PDQ_param]
    # print('Top AIC Config : {} with AIC : {}'.format(top_AICParam, min_AIC))
    # print('Top BIC Config : {} with BIC : {}'.format(top_BICParam, min_BIC))


    """
    Top AIC Config : [(4, 0, 0), (2, 2, 1, 12)] with AIC : 189.280667743 -> Lower MSE
    Top BIC Config : [(4, 0, 0), (2, 2, 0, 12)] with BIC : 204.849870113 ->
    """
    evaluate_model(Training_set, Test_set, (4, 0, 0), (2, 2, 1, 12))
    # evaluate_models(Training_set, Test_set, [[(4, 0, 0), (2, 2, 1, 12)], [(4, 0, 0), (2, 2, 0, 12)]])
    #
