# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 22:00:40 2019

@author: ruby_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#question one

ads = pd.read_csv('data/ads_dataset.tsv',sep = '\t',encoding='utf-8')


# question two
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

#question four
# Place your code here
new_dataframe = getDfSummary(ads)
result_missing_values = new_dataframe[new_dataframe.Number_NaN>0]


# question five
# seeing the dataset we can find that the Nan values always happen in [video_freq]
# try to drop that variance using the dropna()
dataframe_after_drop = ads.dropna()
#Then use the getDfSummary to analysis the new dataframe
Summary_after_drop = getDfSummary(dataframe_after_drop)

# according to the staff we did before, we can conclude that [video_interval] always be 0 or bigger than 0, 
# and to look at the daraframe, when [video_interval] = 0, the [video_freq] = Nan
# so the first thing want to try is to focus on the correlation between [video_interval] and [video_freq]
# the function I choose is corr() from pandas

correlation_between_video_interval_video_freq = dataframe_after_drop.video_interval.corr(dataframe_after_drop['video_freq'])
print(correlation_between_video_interval_video_freq)
# but you can see is not too high 
correlation_between_all_columns = ads.corr()
# we can see the correlation between video_freq with other features
print(correlation_between_all_columns.video_freq)


#is_video_user               NaN
#video_freq             1.000000
#call_freq              0.487548
#video_interval        -0.024565
#call_interval          0.000280
#expected_video_time    0.011030
#expected_call_time     0.060817
#last_bill             -0.126793
#next_bill             -0.126793
#multiple_video         0.028302
#multiple_carrier       0.154837
#uniq_urls              0.042624
#num_texts              0.042764
#is_churn               0.128118

# the highest one is [call_freq ]
# then print that two columns, to see the relation

between_video_freq_call_freq_after_drop = dataframe_after_drop.video_freq.corr(dataframe_after_drop['call_freq'])
between_video_freq_call_freq = ads.video_freq.corr(ads['call_freq'])
print(between_video_freq_call_freq_after_drop)
print(between_video_freq_call_freq)
#print(between_video_freq_multiple)

# when video interval>0, video_freq is not equal to Nan.



# question five
# Place your code here
#number distinct equal to 2 maybe are binary
# question: might be all 0 or all 1 and other might be 10, not binary
new_dataframe = getDfSummary(ads)
result_binary = new_dataframe[new_dataframe.Number_Distinct==2]


# question six
x = ads.num_texts
plt.hist(x, bins=100)
plt.title('Histogram of Number of Texts')
plt.show()



# question seven
# guassion distribution
ads_num_texts = np.array(ads["num_texts"].values)
#print(type(ads_num_texts))
ads_num_texts = np.log(ads_num_texts)
ads["log_num_texts"] = pd.DataFrame(ads_num_texts)

x_new = ads.log_num_texts
plt.hist(x_new, bins=100)
plt.title('Histogram of Number of Texts ( log )')
plt.show()

