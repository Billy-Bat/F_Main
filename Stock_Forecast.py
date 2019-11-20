import warnings
import sys
warnings.filterwarnings("ignore")
## PANDAS import
import pandas as pd
from pandas import read_csv
from pandas.plotting import lag_plot, autocorrelation_plot

import pandas_datareader
from pandas_datareader import data

# STATSMODELS import
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.gofplots import qqplot
import statsmodels.graphics.tsaplots as tsa

# MATPLOTLIB import
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["text.color"] = 'k'
matplotlib.rcParams["font.size"] = 5
matplotlib.rcParams["xtick.labelsize"] = 5
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams["lines.linewidth"] = 0.5
matplotlib.rcParams["figure.subplot.hspace"] = 0.5
matplotlib.rcParams["axes.labelsize"] = 6
matplotlib.rcParams["lines.markersize"] = 0.5
matplotlib.rcParams['figure.figsize'] = 15, 7

import itertools
from sklearn.metrics import mean_squared_error
from datetime import timedelta

########################################################################

def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

def SampleSeasonality(DataSet, frequency, _plot=False) :
    resample = DataSet.resample(frequency)
    mean = resample.mean()


    if _plot == True :
        fig = mean.plot()
        fig.set_title('seasonality sampled with {}'.format(frequency))
        plt.show()

    return mean

def Get_Config(pdq_range, PDQ_range, Seasonal):
    p = range(pdq_range[0][0], pdq_range[0][1]+1)
    d = range(pdq_range[1][0], pdq_range[1][1]+1)
    q = range(pdq_range[2][0], pdq_range[2][1]+1)

    pdq = list(itertools.product(p, d, q))
    PDQ = []
    P = range(PDQ_range[0][0], PDQ_range[1][1]+1)
    D = range(PDQ_range[1][0], PDQ_range[1][1]+1)
    Q = range(PDQ_range[2][0], PDQ_range[2][1]+1)
    for Season_Period in Seasonal :
        PDQ = PDQ + [(x[0], x[1], x[2], Season_Period) for x in list(itertools.product(P, D, Q))]

    return pdq, PDQ

def evaluate_model(Train_set, Test_set, pdq_val, PDQ_val, _print=True, _plot=True) :
    predictions = []
    history = [int(x) for x in Train_set.iloc[:,0]]
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
    Error_Array = Test_set.iloc[:,0] - predictions

    if _print :
        print('Config{}{} MSE : {}'.format(pdq_val, PDQ_val, error))

    if _plot == True :
        fig = plt.figure()
        fig.suptitle('SARIMA{}{} Model Analysis'.format(pdq_val, PDQ_val), size=14)
        gs = gridspec.GridSpec(3, 2) # For some reason add_gridspec DW
        ax1 = plt.subplot(gs[:,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[1,1])
        ax4 = plt.subplot(gs[2,1])

        #1# Rolling Forecast
        Actual = pd.concat([Train_set.iloc[:,0], Test_set.iloc[:,0]])
        Actual.plot(ax=ax1)
        pd.DataFrame(predictions ,index=Test_set.index).plot(ax=ax1)
        ax1.legend(['Actual', 'Rolling forcast'])
        #2# Residual Distrib
        Error_Array.plot(ax=ax2, kind='kde')
        ax2.set_title('Residual Distribution')
        #3# Residual ACF
        tsa.plot_acf(Error_Array, ax=ax3, lags=12, alpha=0.05)
        #4# Residual PACF
        tsa.plot_pacf(Error_Array, ax=ax4, lags=12, alpha=0.05)

        plt.show()


    return error

def plot_Model_Identify(DataSet, frequency=1, acf_lag=12, pacf_lag=12):
    """
    DataSet : dataframe with the type of first column either int()
    or panda datetime
    Frequency : int, Seasonal Component period (in time step)
    """
    # Organize plot
    fig, ax = plt.subplots(3, 5)
    fig.suptitle('Model Identification Season length :{}'.format(frequency), size=14)

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
    try :
      decomposition.resid.plot(ax=ax[0,3], kind='kde')
      tsa.plot_acf(decomposition.resid.dropna(), ax=ax[1,3], lags=acf_lag*2, alpha=0.05)
      tsa.plot_pacf(decomposition.resid.dropna(), ax=ax[2,3], lags=pacf_lag*2, alpha=0.05)
    except :
        print('WARNING : Residual Probability plot Failed')
        print(sys.exc_info()[0])

    decomposition.seasonal.plot(ax=ax[1,2])
    decomposition.trend.plot(ax=ax[2,2])
    Ljung_BoxTest(decomposition.resid, frequency)

    # lag_plot(decomposition.seasonal.dropna(), ax=ax[0,4]) #Seasonality doesnt need DIFF, i.e it has no (obvious) trend
    tsa.plot_acf(decomposition.seasonal, ax=ax[0,4], lags=acf_lag, alpha=0.05)
    tsa.plot_pacf(decomposition.seasonal, ax=ax[1,4], lags=pacf_lag, alpha=0.05)


    ax[0, 2].set_title('Residual')
    ax[1, 2].set_title('Seasonal')
    ax[2, 2].set_title('Trend')
    ax[0, 3].set_title('Residual Prob. Distrib')
    ax[1,3].set_title('Residual ACF')
    ax[2,3].set_title('Residual PACF')

    plt.show()
    return 1

def DFuller_Test(Data, _print=True):
    """
    Perform Dickey Fuller on the seasonal/Residual Data
    Identify if the Data is defined by a trend
    p < 0.05 Data is stationary
    p > 0.05 Data is non-stationary
    """
    Data = Data.dropna() # Some NaN entries
    X = Data.values
    result = adfuller(X)
    if _print :
        print('ADF Result: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        for key, value in result[4].items():
            print('Critical Value {}: {}'.format(key, value))

    return result

def Ljung_BoxTest(Data, lag, BP=False, _print=True,):
    """
    Perform a Ljung-Box Test on the Residual Data up to lag
    By Default use lag=Season_Periodx2
    Test only check if values are dependent, doesnt assert that values are independant
    """ # CHECKS FOR NO AUTOCORELLATION
    try :
        X = Data.dropna()
        Res = acorr_ljungbox(X, lags=lag, boxpierce=BP)
    except :
        print('WARNING : Ljung_BoxTest Failed')
        print(sys.exc_info()[0])

    if _print :
        print('Ljung-Box value: {}'.format(Res[0].mean()))
        print('LB p value: {}'.format(Res[1].mean()))
        if BP :
            print('Box-Pierce value: {}'.format(Res[2]))
            print('BP p value: {}'.format(Res[3]))


################################################################################
#1# Get Data
timeSpan = 60 # in weeks
start_date = '2018-01-01'
end_date = (pd.to_datetime(start_date) + timedelta(weeks=60)).strftime("%Y-%m-%d")

#1.1# Write Data to File
# df = data.DataReader('MSFT', 'yahoo', start_date, end_date)
# df.to_csv('SARIMA/MSFT.csv')

#1.2# Format Data
df = read_csv('SARIMA/MSFT.csv', usecols=["Date", "High", "Low"])
df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.drop(df.columns[0], axis=1)
df.dropna()
mean = df.resample('M').mean()

#2# Identify model charaterstics
# plot_Model_Identify(df, frequency=15, acf_lag=20, pacf_lag=20)

#3# Divid the set into training and test
Training_set, Test_set = train_test_split(df, 24)

#4# Grid Search Optimal parameters
# pdq, PDQ = Get_Config([(2,5), (5,8), (0,2)], [(0,0), (0,0), (0,0)], [45])
#
# min_BIC = float('inf') # quite arbitrary
# min_AIC = float('inf')
# min_sharedBIC = float('inf')
# min_sharedAIC = float('inf')
#
# top_BICParam = []
# top_AICParam = []
# top_shared = []
#
# for pdq_param in pdq :
#     for PDQ_param in PDQ :
#         mod = sm.tsa.statespace.SARIMAX(Training_set.iloc[:,0],
#                                                 order=pdq_param,
#                                                 seasonal_order=PDQ_param,
#                                                 enforce_stationarity=False,
#                                                 enforce_invertibility=False,
#                                                 disp=False)
#         result = mod.fit(disp=False)
#         print('Model pdq :{}, PDQ :{} AIC : {}'.format(pdq_param, PDQ_param, result.bic))
#
#         if result.bic < min_BIC :
#             min_BIC = result.bic
#             top_BICParam = [pdq_param, PDQ_param]
#         if result.aic < min_AIC :
#             min_AIC = result.aic
#             top_AICParam = [pdq_param, PDQ_param]
#         if result.bic < min_sharedBIC and result.aic < min_sharedAIC :
#             min_sharedBIC = result.bic
#             min_sharedAIC = result.aic
#             top_shared = [pdq_param, PDQ_param]
# print('Top AIC Config : {} with AIC : {}'.format(top_AICParam, min_AIC))
# print('Top BIC Config : {} with BIC : {}'.format(top_BICParam, min_BIC))
# print('Shared top Config : {} with BIC : {} AIC : {}'.format(top_shared, min_sharedBIC, min_sharedAIC))


#5# Evaluate Model(s)
evaluate_model(Training_set, Test_set, (3, 0, 1), (0, 0, 0, 80))


"""
INFO ABOUT MODEL PERF :
Model pdq :(80, 0, 0), PDQ :(0, 0, 0, 45) BIC : 1014.24966081
Model pdq :(80, 1, 0), PDQ :(0, 0, 0, 45) BIC : 1009.16963999
Model pdq :(80, 1, 1), PDQ :(0, 0, 0, 45) BIC : 1010.33599926
Model pdq :(80, 0, 1), PDQ :(0, 0, 0, 45) BIC : 1022.04934976
Top AIC Config : [(80, 1, 1), (0, 0, 0, 45)] with AIC : 741.950035457
Top AIC Config : [(9, 1, 0), (0, 0, 0, 45)] with AIC : 926.368999944
Top BIC Config : [(0, 1, 0), (0, 0, 0, 45)] with BIC : 943.625717968
Top AIC Config : [(2, 1, 4), (0, 0, 0, 45)] with AIC : 890.877282277
Top BIC Config : [(2, 1, 0), (0, 0, 0, 45)] with BIC : 913.382874537 # Best One shared !!

# MSE on 24 steps Rolling Forecast M
Config(2, 1, 2)(0, 0, 0, 45) MSE : 1.15520566641
Config(2, 1, 1)(0, 0, 0, 45) MSE : 1.15826477698
Config(5, 0, 1)(0, 0, 0, 45) MSE : 1.11962684636
Config(7, 0, 1)(0, 0, 0, 45) MSE : 1.13870691385
Config(5, 1, 3)(0, 0, 0, 45) MSE : 1.34289218215
Config(4, 0, 1)(0, 0, 0, 45) MSE : 1.06257566771
Config(3, 0, 1)(0, 0, 0, 45) MSE : 1.04004984687
Config(2, 0, 1)(0, 0, 0, 45) MSE : 1.07724288015
Config(3, 1, 0)(0, 0, 0, 45) MSE : 1.09876421509

"""
