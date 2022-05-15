
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



