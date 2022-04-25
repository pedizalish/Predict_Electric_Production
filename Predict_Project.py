
from statistics import mean
import pandas as pnd
import matplotlib.pyplot as mplot
import numpy as np

File_Name = r'C:\Users\User\Desktop\Excel\Electric_Production.csv'

def Last_Period_Demand (DataFrame):
    Predicted_1 = ['-']
    for i in range(0,len(DataFrame['PRODUCT'])):
        Predicted_1.append(DataFrame['PRODUCT'][i])
    return Predicted_1

def Simple_Moving_Average_k3 (DataFrame):
    Predicted_2 = ['-','-','-']
    for i in range(0,len(DataFrame)):
        Predicted_2.append(sum(DataFrame['PRODUCT'][i:i+3])/3)
    return Predicted_2

def Modified_Moving_Average (k,DataFrame):
    Predicted_3 = []
    DataFrame_DF = pnd.DataFrame(DataFrame)
    for i in range(k):
        Predicted_3.append('-')
    DataFrame_DF_index = []
    DataFrame_DF_index = DataFrame_DF.index + 1
    for i in range(0,len(DataFrame)):
        Predicted_3.append( sum(DataFrame_DF['PRODUCT'][i:i+k])/k + (6)/(k*(k+1))*(sum(DataFrame_DF_index[i:i+k]*(DataFrame_DF['PRODUCT'][i:i+k]))-(((2*(i+k)+1-k)/2))*(sum(DataFrame_DF['PRODUCT'][i:i+k]))))
    return Predicted_3

def Weighted_Moving_Average (DataFrame):
    Weight = [0.02,0.08,0.15,0.25,0.5]
    Predicted_4 = []
    for i in range(5):
        Predicted_4.append('-')
    Weight_array = np.array(Weight)
    for i in range(0,len(DataFrame)-5):
        DataFrame_array = np.array(DataFrame['PRODUCT'][i:i+5])
        Predicted_4.append(sum(Weight_array * DataFrame_array))
    return Predicted_4
    
def Simple_Exponential_Smoothing (a,DataFrame):
    Predicted_5 = ['-',DataFrame['PRODUCT'][0]]
    for i in range(0,len(DataFrame)):
        Predicted_5.append(Predicted_5[i+1] + (a)*(DataFrame['PRODUCT'][i+1] - Predicted_5[i+1]))
    return Predicted_5

def Trend_Adjusted_Exponential_Smoothing (a,B,DataFrame):
    Predicted_6 = ['-']
    Adjusted = [DataFrame['PRODUCT'][0]]
    Trend = [0]
    for i in range(0,len(DataFrame)-1):
        Adjusted.append((a * DataFrame['PRODUCT'][i+1]) + ((1-a)*(Adjusted[i] + Trend[i])))
        Trend.append((B * (Adjusted[i+1] - Adjusted[i])) + ((1-B) * Trend[i]))
        Predicted_6.append(Adjusted[i+1] + Trend[i+1])
    return Predicted_6

def Seasonal_Adjusted_Exponential_Smoothing (a,c,DataFrame):
    Predicted_7 = []
    Adjusted = []
    seasonal_index = []
    for i in range(0,12):
        Predicted_7.append('-')
    for i in range(0,11):
        Adjusted.append('-')
    Adjusted.append(DataFrame['PRODUCT'][0])
    sum_Period_Product = sum(DataFrame['PRODUCT'][0:12])
    for i in range(0,12):
        seasonal_index.append(DataFrame['PRODUCT'][i] / (sum_Period_Product / 12))
    for i in range(12,len(DataFrame)) :
        Adjusted.append(a*((DataFrame['PRODUCT'][i]) / seasonal_index[i-12]) + ((1-a)*Adjusted[i-1]))
        seasonal_index.append(c*(DataFrame['PRODUCT'][i] / Adjusted[i]) + ((1-c)*seasonal_index[i-12]))
    for i in range(0,len(seasonal_index),12):
        if sum(seasonal_index[i:i+12]) != 12:
            seasonal_index_array = np.array(seasonal_index[i:i+12])
            seasonal_index_array = seasonal_index_array * ( 12 / (12 + seasonal_index[i] - seasonal_index[i-12]))
            seasonal_index[i:i+12] = seasonal_index_array
    for i in range(12,len(DataFrame)):
        Predicted_7.append((Adjusted[i]*seasonal_index[i]))
    return Predicted_7

def Simple_Regression(DataFrame):
    Predicted_8 = []
    sum_i = 0
    sum_Product = 0
    for i in range(0,len(DataFrame)//2):
        Predicted_8.append('-')
        sum_i += i+1
        sum_Product += DataFrame['PRODUCT'][i]
    DataFrame_DF = pnd.DataFrame(DataFrame)
    DataFrame_DF_index = DataFrame_DF.index
    b = (sum((DataFrame_DF_index[0:len(DataFrame)//2]+1)*DataFrame['PRODUCT'][0:len(DataFrame)//2]) - (len(DataFrame)//2)*(sum_i/(len(DataFrame)//2))*(sum_Product/(len(DataFrame)//2))) / (sum((DataFrame_DF_index[0:len(DataFrame)//2]+1)*(DataFrame_DF_index[0:len(DataFrame)//2]+1)) - (len(DataFrame)//2)*(sum_i/(len(DataFrame)//2))*(sum_i/(len(DataFrame)//2)))
    a = sum_Product/(len(DataFrame)//2) - (b * (sum_i/(len(DataFrame)//2)))
    for i in range(len(DataFrame)//2 , len(DataFrame)):
        Predicted_8.append( a + b * (i+1) )
    return Predicted_8
        
def Seasonal_Modified_Simple_Regression(DataFrame):
    Predicted_9 = []
    seasonal_index = []
    for i in range(0,len(DataFrame)//2):
        Predicted_9.append('-')
        seasonal_index.append('-')
    Sum_Period_Product = 0
    for i in range(len(DataFrame)//2,len(DataFrame),12):
        Sum_Period_Product = sum(DataFrame['PRODUCT'][i:i+12])
        seasonal_index.extend(DataFrame['PRODUCT'][i:i+12] / (Sum_Period_Product / 12))
    for i in range(len(DataFrame)//2,len(DataFrame)):
            Predicted_9.append(np.array(Simple_Regression(DataFrame)[i]) * np.array(seasonal_index[i]))
    return Predicted_9

def MAE(Method_function,DataFrame):
    Sum_error = 0
    Count = 0
    for i in range(0,len(DataFrame)):
        if (isinstance(Method_function(DataFrame)[i],float)) :
            Count += 1
            Sum_error += abs(DataFrame['PRODUCT'][i] - Method_function(DataFrame)[i])
    MAE = Sum_error / Count
    return MAE

def MSE(Method_function,DataFrame):
    Sum_Error = 0
    Count = 0
    for i in range(0,len(DataFrame)):
        if isinstance(Method_function(DataFrame)[i],float):
            Count += 1
            Sum_Error += pow(DataFrame['PRODUCT'][i] - Method_function(DataFrame)[i] , 2)
    MSE = Sum_Error / Count
    return MSE

def MAPE(Method_function , DataFrame):
    Sum_Error = 0
    Count = 0
    for i in range(0,len(DataFrame)):
        if isinstance(Method_function(DataFrame)[i],float):
            Count += 1
            Sum_Error += abs(((DataFrame['PRODUCT'][i] - Method_function(DataFrame)[i]) / DataFrame['PRODUCT'][i]) * 100)
    MAPE = Sum_Error / Count
    return MAPE

def ME(Method_function , DataFrame):
    Sum_Error = 0
    Count = 0
    for i in range(0,len(DataFrame)):
        if isinstance(Method_function(DataFrame)[i],float):
            Count += 1
            Sum_Error += (DataFrame['PRODUCT'][i] - Method_function(DataFrame)[i])
    ME = Sum_Error / Count
    return ME

def MPE(Method_function , DataFrame):
    Sum_Error = 0
    Count = 0
    for i in range(0,len(DataFrame)):
        if isinstance(Method_function(DataFrame)[i],float):
            Count += 1
            Sum_Error += (((DataFrame['PRODUCT'][i] - Method_function(DataFrame)[i]) / DataFrame['PRODUCT'][i]) * 100)
    MPE = Sum_Error / Count
    return MPE

def TS(Method_function , DataFrame):
    TS_Sum = 0
    for i in range(0,len(DataFrame)):
        if isinstance(Method_function(DataFrame)[i] , float):
            TS_Sum += (DataFrame['PRODUCT'][i] - Method_function(DataFrame)[i])
    TS = TS_Sum / MAE(Method_function,DataFrame)
    return TS

def Draw_Diagarm(Method_function, DataFrame , *arg):
    X_Axis = []
    Y1_Axis = []
    Y2_Axis = []
    for i in range(0,len(DataFrame['PRODUCT'])):
        if isinstance(Method_function(*arg , DataFrame)[i],float):
            X_Axis.append(DataFrame['DATE'][i])
            Y1_Axis.append(Method_function(*arg , DataFrame)[i])
            Y2_Axis.append(DataFrame['PRODUCT'][i])
    mplot.plot(X_Axis,Y1_Axis,color='r',label='Predicted')
    mplot.plot(X_Axis,Y2_Axis,color='g',label='Real')
    mplot.xlabel('DATE')
    mplot.ylabel('Predicted')
    mplot.title(Method_function)
    mplot.legend()
    mplot.show()

DataFrame_1 = pnd.read_csv(File_Name)

print('Period\tDATE\t\tElectric_Production\tPredicted')
for i in range(0 , len(DataFrame_1)):
    print(str(i) + '\t' + str(DataFrame_1['DATE'][i]) + '\t' + str(DataFrame_1['PRODUCT'][i]) + '\t\t\t' + str(Trend_Adjusted_Exponential_Smoothing(0.1,0.3,DataFrame_1)[i]))
print('The Error in Last Period Demand is ' + str(TS(Last_Period_Demand,DataFrame_1)))
Draw_Diagarm(Simple_Moving_Average_k3 , DataFrame_1)