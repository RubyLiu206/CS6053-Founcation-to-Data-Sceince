# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:56:42 2019

@author: ruby_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ads = pd.read_csv('data/ads_dataset.tsv',sep = '\t',encoding='utf-8')
def getDfSummary(input_data):
    # Place your code here
    # Place your code here
    #new a new data frame
    #for each variable compute the number_nan, num_distinct, mean, max, min, std
    #put the input_data's column to the output_data's index
    #put the new features to the output_data's columns
    each_row = []
    for feature in input_data:
        num_nan = np.count_nonzero(input_data[feature].isnull())
        num_distinct = input_data[feature].nunique()
        des = input_data[feature].describe()[['mean','max','min','std','25%','50%','75%']].values.tolist()
        each_row.append([num_nan,num_distinct]+des)
    output_data = pd.DataFrame(each_row)
    output_data.columns = ['Number_NaN','Number_Distinct','Mean','Max','Min','Std','25%','50%','75%']
    output_data.index = [input_data.columns]
    return output_data

getDfSummary(ads)

new_dataframe = getDfSummary(ads)
new_dataframe[new_dataframe.Number_NaN>0]
print(new_dataframe._stat_axis.values.tolist())



# seeing the dataset we can find that the Nan values always happen in [video_freq]
# try to contain all the variance with the nan data
dataframe_only_with_nan = ads[ads["video_freq"].isnull()]
#Then use the getDfSummary to analysis the new dataframe
Summary_with_nan = getDfSummary(dataframe_only_with_nan)
Summary_dataframe = getDfSummary(ads)
print(Summary_with_nan)
print(Summary_dataframe)
# according to the staff we did before, we can conclude that [video_interval] always be 0 or bigger than 0, 
# and to look at the daraframe, when [video_interval] = 0, the [video_freq] = Nan
# so the first thing want to try is to focus on the correlation between [video_interval] and [video_freq]
# the function I choose is corr() from pandas

correlation_between_video_interval_video_freq = ads.video_freq.corr(ads['video_interval'])
correlation_between_expected_video_time_video_freq= ads.video_freq.corr(ads['expected_video_time'])
# we can see the correlation between video_freq with expected_video_time


# Conclusion :
# So we can only look at the describle charts: [video_interval] this feature from 1 to 121 in number distinct, and [expected_video_time] also change from 1 to 134
# if [video_interval] <= 0 and [expected_video_time] <= 0, then the [video_freq] = Nan




# Place your code here
#number distinct equal to 2 maybe are binary
# question: might be all 0 or all 1 and other might be 10, not binary
new_dataframe = getDfSummary(ads)
result_binary = new_dataframe[new_dataframe.Number_Distinct==2]
print(result_binary._stat_axis.values.tolist())


x = ads.num_texts
plt.hist(x, bins = 100)
plt.title('Histogram of Number of Texts')
plt.show()


# guassion distribution, a bell-shaped frequency distribution curve. 
# to make the distribution more bell curved, at the mean range increase the num texts, to make standard deviation of the mean more higher.
# the advantage is we can see the data distribute more easily with the guassion distribution.

ads_num_texts = np.array(ads["num_texts"].values)
#print(type(ads_num_texts))
ads_num_texts = np.log(ads_num_texts)
ads["log_num_texts"] = pd.DataFrame(ads_num_texts)

x_new = ads.log_num_texts
plt.hist(x_new, bins=100)
plt.title('Histogram of Number of Texts ( log )')
plt.show()