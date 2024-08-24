#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 12:19:24 2020

@author: binhnguyen
"""


import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import *
import csv
from sklearn.metrics import confusion_matrix 

########################### Import sensor values ->  (subject, sensor_num)
from functions import sensor_val
########################### Import EMA values ->  (subject, ema_num)
from functions import ema_val
# Kind of Useless
from functions import activity_ 


########################### Subject 
s1 =  [('u0'+str(x)) for x in range(9 + 1)]
s2 =  ['u'+str(x) for x in range(10,57)]
subject = s1+s2


'''
# PASSIVE SENSOR ANALYSIS - see what is available and what isn't 
####### Activity
sensor_num = 0
from functions import activity_passive 
activity_passive (subject, sensor_num)


# ACTIVE SENSOR ANALYSIS - see what is available and what isn't 
####### Stress        
stress_num = 10
from functions import stress_active
stress_active(subject, stress_num) 
'''

########################### PHQ-9 Analysis
from functions import survey_reader
from functions import survey_encoder_phq

surv_num = 0
phq = survey_reader (surv_num)
dataset_score_pre = []
dataset_score_post = []
user_pre = []
user_post = []

for i in range (len(phq)):
    user = phq.iloc [i,2:-1]
    phq_score = np.sum (survey_encoder_phq(user))
    if (phq.iloc[i,1]=='pre'):
        dataset_score_pre.append(phq_score)
        user_pre.append (phq.iloc[i,0])
    elif (phq.iloc[i,1]=='post'):
        dataset_score_post.append(phq_score)
        user_post.append (phq.iloc[i,0])
        
phq_post = np.vstack((user_post,dataset_score_post))
phq_pre = np.vstack((user_pre,dataset_score_pre))

phq_final = phq_post

##### Save phq files for pre and post scores
# with open('phq_post.txt', 'w') as f:
#     for item in user_post:
#         f.write("%s\n" % item)
#     for item in dataset_score_post:
#         f.write("%d\n" % item)

# with open('phq_pre.txt', 'w') as f:
#     for item in user_pre:
#         f.write("%s\n" % item)
#     for item in dataset_score_pre:
#         f.write("%d\n" % item)

         

########################### Survey idenfitication 
survey = phq_final


########################### Create the dataset

from functions import time_difference
from functions import haversine
from functions import kmeans_cluster
from functions import daily_analysis_1
from functions import daily_analysis_2
from functions import GPS_csv_reader
from functions import entropy_fct
from functions import percentage_features
from functions import total_distance


# Feature vector
avg_view = []
loc_view = []

#### Beginning of file
# Trends
walk_f1 = open("python_walk.txt", "w")
aud_f2 = open("python_noise.txt", "w")
convo_f3 = open("python_convo.txt", "w")

# For loop to run through all subjects
for subject in survey[0]:

    ############################ Average view ############################
    
    ############################ Activity feature 
    sensor_num = 0
    act = sensor_val (subject, sensor_num)
    activity = sensor_val (subject, sensor_num).iloc[:,1]

    # Get daily samples of each category
    x,x,x,act_s,act_w,act_r = daily_analysis_1 (act,0)
        
    # Take length between 2 - 3 seconds (sampling freq)
    r1 = random.uniform(2, 3, len(act_s))
    r2 = random.uniform(2, 3, len(act_w))
    r3 = random.uniform(2, 3, len(act_r))
    
    # (Time it takes for one sample) x (Samples per day)
    t1_act = r1*act_s
    t2_act = r2*act_w
    t3_act = r3*act_r
    
    # Average over all of the days
    avg_act_s = np.mean(t1_act)
    avg_act_w = np.mean(t2_act)
    avg_act_r = np.mean(t3_act)

    ############################ Conversation 
    sensor_num = 3
    convo = sensor_val (subject, sensor_num)
    convo_dur,n_convo = daily_analysis_2 (convo,0)
    
    avg_convo_dur = np.mean(convo_dur)
    avg_convo_num = np.mean(n_convo)

    ############################ Dark features    
    sensor_num = 4
    dark = sensor_val (subject, sensor_num)
    dark_dur, n_dark = daily_analysis_2 (dark,1)
    
    avg_dark_dur = np.mean (dark_dur)
    avg_dark_num = np.mean (n_dark)

    
    ############################ Audio features 
    # The total duration when the audio is classified as quiet, noisy and voice in a day.
    sensor_num = 1
    audio = sensor_val (subject, sensor_num)
    x,x,x,aud_q,aud_v,aud_n = daily_analysis_1 (audio,1)

    # Take length between 2 - 3 seconds (sampling freq)
    r1 = random.uniform(2, 3, len(aud_q))
    r2 = random.uniform(2, 3, len(aud_v))
    r3 = random.uniform(2, 3, len(aud_n))
    
    # (Time it takes for one sample) x (Samples per day)
    t1_aud = r1*aud_q
    t2_aud = r2*aud_v
    t3_aud = r3*aud_n

    # Average over all of the days
    avg_aud_s = np.mean(t1_aud)
    avg_aud_v = np.mean(t2_aud)
    avg_aud_n = np.mean(t3_aud)
    

    ############################ Phone Lock
    sensor_num = 7
    lock = sensor_val (subject, sensor_num)
    lock_dur, n_lock = daily_analysis_2 (lock,1)

    
    avg_lock_dur = np.mean(lock_dur)
    avg_lock_num = np.mean(n_lock)

    hold = avg_act_s, avg_act_w, avg_act_r,\
     avg_convo_dur,avg_convo_num,\
        avg_dark_dur,avg_dark_num,\
            avg_aud_s,avg_aud_v,avg_aud_n,\
                avg_lock_dur, avg_lock_num
    avg_view.append (hold)

    ############################ Trend view ############################

    # Daily trends in each day for activity
    # act_w is daily walk
    
    
    # Daily trends in each day for Noise
    # aud_n is the daily audio noise
    
    # Daily trends in each day for Conversation
    # convo_dur is the daily conversation dur
    
    # Walking, Noise, Conversation to be placed in MATLAB
    # Write to file Daily Activity
    for row in act_w:
        # print (row)
        walk_f1.write (str(row))
        walk_f1.write ('\n')
    walk_f1.write('-1000\n')
    
    # Write to file Daily Noise
    for row in aud_n:
        # print (row)
        aud_f2.write (str(row))
        aud_f2.write ('\n')
    aud_f2.write('-1000\n')
    
    # Write to file Daily Convo
    for row in convo_dur:
        # print (row)
        convo_f3.write (str(row))
        convo_f3.write ('\n')
    convo_f3.write('-1000\n')

    ############################ Location view ############################

    ############################ GPS feature 

    # Next two lines are for testing purposes
    # for subject in survey[0]:
    # subject= survey[0][2]
    
    
    sensor_num = 5
    gps_df = sensor_val (subject, sensor_num)
    gps_time = GPS_csv_reader(subject,5,'time',1) 
    
    gps_lat = gps_df.iloc [:,3]
    gps_lon = gps_df.iloc [:,4]

    # GPS variance is the equation used
    gps_lat_var = np.var (gps_lat)
    gps_lon_var = np.var (gps_lon)
    location_var = (gps_lat_var+gps_lon_var)
    

    
    # Time in location clusters
    # NOTE: these values are basically a percentage of times spent in
    # each clusters. Although data was not collected continuously for all 66
    # days, the variables will not add up to 100%
    # Specifically speaking: tc will not be added to 100%
    tc1,tc2,tc3,tc,n_clust = kmeans_cluster(gps_time,gps_lat,gps_lon, 1)
    
    # Entropy and Normalized entropy
    entropy, norm_entropy = entropy_fct (n_clust,tc)

    # Percentage at home and moving
    home_d, move_p = percentage_features (gps_lat,gps_lon,gps_time,0,subject)
    
    # Total distance using Haversine Formula
    # https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        
    norm_tot_dist = total_distance (gps_lat,gps_lon,gps_time)
    

    hold = location_var, tc1,tc2,tc3, entropy, norm_entropy,\
        home_d,move_p,norm_tot_dist
    
    loc_view.append (hold)
    


############################ Outside loop ############################


########################### Changing label
from sklearn.preprocessing import LabelEncoder
from functions import phq_severity_sri
from functions import phq_severity
from functions import panas_label_seperater

label_encoder = LabelEncoder()
phq_lab = label_encoder.fit_transform(phq_severity_sri(\
    np.asfarray(phq_final[1],float)))


#### Write to files
# PHQ-9
np.savetxt("phq_score.txt", np.array (phq_lab), delimiter=',')  

# Average 
np.savetxt("avg_view.txt", np.array (avg_view), delimiter=',')  

# Location 
np.savetxt("loc_view.txt", np.array (loc_view), delimiter=',')  


#### End of file
# Trends
walk_f1.close()
aud_f2.close()
convo_f3.close()



'''
# Loading files from MATLAB
from functions import matlab2python
trend_walk = matlab2python('mat_walk.txt', len(survey[0]))
'''  
        

############################ END ############################

'''

    # Testing purposes below

    signal = daily_aud_r
    signal = daily_act_w
    
    
    # Wavelet Decomposition
    import pywt
   
    wv_signal = wavelet_denoise(signal,1) # better option 
    wv_signal2 = wavelet_denoise2(signal,1)


    # Least squares: Option 1
    
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    signal_lsr = wv_signal
    x = np.vstack((np.arange (0,len(signal))))
    
    lr = LinearRegression()
    lr.fit(x,signal_lsr)

    # Rank of the signal
    r = np.linalg.matrix_rank(signal_lsr)
    
    # Assign values
    w = lr.coef_[0]
    intercept = lr.intercept_
    
    # Plot
    x_reg = np.arange (0,len(signal))
    y_reg = w*x + intercept
    
    plt.scatter(x_reg,y_reg)
    
    
    
    # Least squares: Option 2
    signal_lsr = wv_signal
    x = np.arange (0,len(signal))
    popt = np.polyfit(x, signal_lsr, 10)
    
    y_reg = np.polyval(popt, x_reg)
    f = np.poly1d(y_reg)
    

    
    plt.figure()
    plt.scatter(x_reg,y_reg)
    plt.plot (x, signal_lsr)


    # Least squares: Option 3    
    from scipy import optimize
    def f(x, a, b):
        return a*np.sin(b*np.pi*x)

    x = np.arange (0,len(signal))
    signal_lsr = wv_signal
    
    popt, pcov = optimize.curve_fit(f, x, signal_lsr)
    
    print (popt)
    
    x_reg = np.arange (0,len(signal))
    
    plt.figure()
    plt.plot(x, f(x_reg, *popt))
    plt.plot (x, signal_lsr)
    
    # Least squares: Option 4
    import numpy, scipy.optimize

    def fit_sin(tt, yy):
        tt = numpy.array(tt)
        yy = numpy.array(yy)
        ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(numpy.fft.fft(yy))
        guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = numpy.std(yy) * 2.**0.5
        guess_offset = numpy.mean(yy)
        guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])
    
        def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
        
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2.*numpy.pi)
        fitfunc = lambda t: A * numpy.sin(w*t + p) + c
        
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}


    res = fit_sin(x, wv_signal)
    print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )
    
    plt.figure()
    plt.plot(x, res["fitfunc"](x), "r-", linewidth=2)
    plt.plot(x, wv_signal)

    # Testing purposes stop


'''