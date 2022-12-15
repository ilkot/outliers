
import pandas as pd
import numpy as np
from outliers import smirnov_grubbs as grubbs

# =============================================================================
# Univariate
# =============================================================================

# =============================================================================
# #Grubbs test
# =============================================================================
def grubbsTest(data,testType='2side',alpha=.05,removeOutliers=False):
    '''
    Parameters
    ----------
    data : 1d array or pd.Series
        DESCRIPTION inputData as np.array.
    testType : string
        The default is '2side', can work with 'min' or 'max'.
    alpha : TYPE, float
        The significance level to use for the test, default is .05.
    Returns
    -------
    removeOutliers = False, input data returns
    removeOutliers = True, manipulated data returns

    '''
    if testType == '2side':
        removedData = grubbs.test(data,alpha)
    elif testType == 'min':
        removedData = grubbs.min_test(data,alpha)
    elif testType == 'max':
        removedData = grubbs.max_test(data,alpha)
    else:
        return print('wrong testType passed, you can only use\
                     2side, min or max')
    
    if len(removedData) == len(data):
        print('No outliers found, consider changing alpha value which is default .05')
    else:
        print("Detected outliers(grubbsTest-{}) : ".format(testType),np.setdiff1d(data, removedData))
        
    if removeOutliers==True:
        return removedData
    
#%%
#usage#  
sampleData1 = np.array([20, 21, 26, 24, 29, 22, 21, 50, 28, 27])
grubbsTest(sampleData1,'2side')
grubbsTest(sampleData1,'min')
grubbsTest(sampleData1,'max')

sampleData2 = pd.Series(sampleData1)

grubbsTest(sampleData2,'2side',removeOutliers=True)
#%%
# =============================================================================
# STD Based
# =============================================================================
data = sampleData1.copy()

def stdBasedOutlier(data,sigma=3,outType='2side',removeOutlier=False):   
    data_mean, data_std = np.mean(data), np.std(data)
    thresh = data_std * sigma
    lower, upper = (data_mean - thresh), (data_mean + thresh)
    
    if outType == '2side':
        outliers = [x for x in data if x < lower or x > upper]
    elif outType == 'min':
        outliers = [x for x in data if x < lower]
    elif outType == 'max':
        outliers = [x for x in data if x > upper]
    else:
       return print('wrong testType passed, you can only use 2side, min or max')
    
    if len(outliers) == 0:
        print('No outlier found, consider change sigma value which is default 3')
    else: 
        print('Identified outliers count: %d' % len(outliers))
        print(outliers)
    
    if removeOutlier==True:
        outliers_removed = [x for x in data if x >= lower and x <= upper]
        return np.array(outliers_removed)
    else:
        return data



def stdBasedOutlier_df(df,col_name, sigma=3, removeOutlier=False):
    # Compute the mean and standard deviation of the given column
    col_mean = np.mean(df[col_name])
    col_std = np.std(df[col_name])
    
    # Compute the z-score of each value in the column
    z_scores = [(val - col_mean) / col_std for val in df[col_name]]
    
    # Create a boolean mask to identify the values that are considered outliers
    # based on the z-score and the specified number of standard deviations (sigma)
    mask = np.abs(z_scores) > sigma
    
    if removeOutlier:
        # Remove the outliers from the data
        df = df[~mask]
    else:
        # Add a new column to the data indicating whether each value is an outlier
        df['is_outlier'] = mask
    
    return data

    
#%%
aa = stdBasedOutlier(data,sigma=2,outType='min',removeOutlier=True)
#%%
# =============================================================================
# IQR Based
# =============================================================================

data = 5 * np.random.randn(10000) + 50

def IQRBasedOutlier(data,iqrThresh=1.5,outType='2side',removeOutlier=False):   
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    cut_off = iqr * iqrThresh
    lower, upper = q25 - cut_off, q75 + cut_off
    
    if outType == '2side':
        outliers = [x for x in data if x < lower or x > upper]
    elif outType == 'min':
        outliers = [x for x in data if x < lower]
    elif outType == 'max':
        outliers = [x for x in data if x > upper]
    else:
       return print('wrong testType passed, you can only use 2side, min or max')
    
    if len(outliers) == 0:
        print('No outlier found, consider change sigma value which is default 3')
    else: 
        print('Identified outliers count: %d' % len(outliers))
        print(outliers)
    
    if removeOutlier==True:
        outliers_removed = [x for x in data if x >= lower and x <= upper]
        return np.array(outliers_removed)
    else:
        return data


## detect outliers based on a column_name in dataframe
def IQRBasedOutlier_df(df,col_name,iqrThresh=1.5,removeOutlier=False):
    # calculate quartiles
    q1, q3 = df[col_name].quantile([0.25, 0.75])

    # calculate IQR
    iqr = q3 - q1

    # calculate lower and upper bounds
    lower_bound = q1 - (iqr * iqrThresh)
    upper_bound = q3 + (iqr * iqrThresh)

    # create a new boolean column to indicate whether a row is an outlier
    df['is_outlier'] = (df[col_name] < lower_bound) | (df[col_name] > upper_bound)
    
    if removeOutlier:
        df_without_outliers = df[df['is_outlier']==False]
        return df_without_outliers
    else:
        return df






    
    
    
    
    
    
    
    
    
    
    
