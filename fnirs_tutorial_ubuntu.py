# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mne
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

d = pd.read_csv('/home/bk/Desktop/nirssignal.TXT', sep = '\t', engine='python', encoding = 'cp949')
df_sample = d.loc[:, :]
df_sample.columns = ['time', 'task', 'mark', 'count', 'oxy1', 'deoxy1', 'total1', 'oxy2', 'deoxy2', 'total2', 'oxy3', 'deoxy3', 'total3']

ex_data = df_sample.drop(['time', 'mark', 'count', 'total1', 'total2', 'total3'], axis = 1)
ex_data = ex_data[['oxy1', 'deoxy1', 'oxy2', 'deoxy2', 'oxy3', 'deoxy3', 'task']]
ex_data.head()

raw_ndarray = ex_data.values
raw_ndarray.shape


raw_ndarray = np.transpose(ex_data.values)
raw_ndarray.shape

#%%
# Create_info
nChannels = 3 # number of physical channels
sampling_frequency = 15.625

channel_names_fnirs = ['HbO '+"%.2d" % i for i in range(1,nChannels+1)] + ['HbR '+"%.2d" % i for i in range(1,nChannels+1)]
channel_names = channel_names_fnirs + ['Events']

channel_types = ['hbo' for i in range(nChannels)] + ['hbr' for i in range(nChannels)] + ['stim']

info = mne.create_info(ch_names=channel_names, sfreq=sampling_frequency, ch_types=channel_types, montage=None)
info['lowpass'] = 0.1
info['highpass'] = 0.1

# Import fNIRS data
raw_data = mne.io.RawArray(data=raw_ndarray, info=info, first_samp=0, verbose=None)

#%%

# Events   
events = mne.find_events(raw_data, stim_channel='Events', shortest_event=1)
event_id = {'open': 1, 'close': 2, 'fix' : 3, 'left' : 4}
color = {1:'green', 2:'magenta', 3:'cyan', 4:'yellow'}
fig = mne.viz.plot_events(events, raw_data.info['sfreq'], raw_data.first_samp, color=color, event_id=event_id)

#%%
# Bandpass filter raw data
l_freq = 0.02 # high-pass filter cutoff ( __/¯¯¯ )
h_freq = 0.2  #  low-pass filter cutoff ( ¯¯¯\__ )
    
raw_data.filter(l_freq, h_freq)

tmp_data = raw_data.get_data()
np.max(np.abs(tmp_data))


#%%
scalings = dict(hbo=1e-1, hbr=1e-1, stim=1)
fig_title = 'fNIRS Raw Bandpass filtered [' + str(l_freq) + ' Hz, ' + str(h_freq) + ' Hz]'
plot_colors = dict(hbo='r', hbr='b', stim='k')
fig = raw_data.plot(title=fig_title, events=events, start=0.0, color=plot_colors, event_color=color, 
                    duration=np.max(raw_data.times), scalings=scalings, order=None, n_channels=len(channel_names), remove_dc=False, highpass=None, lowpass=None)


#%%
# Plot Raw data (default)
scalings = dict(hbo=10e-2, hbr=10e-2, stim=1)
fig_title = 'fNIRS Raw Bandpass filtered [' + str(l_freq) + ' Hz, ' + str(h_freq) + ' Hz]'
plot_colors = dict(hbo='r', hbr='b', stim='k')
fig = raw_data.plot(title=fig_title, events=events, start=10.0, color=plot_colors, event_color=color, 
                    scalings=scalings, order=None, duration=np.max(raw_data.times), 
                    remove_dc=False, highpass=True, lowpass=True)

#%%
#%% Epochs
tmin = -2
tmax = 10
reject = dict(hbo=1, hbr=1)
scalings = dict(hbo=1e-1, hbr=1e-1, stim=1)

epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, proj=True, baseline=(None, 0), preload=True, reject=reject)    

fig_title = 'fNIRS Epochs'
fig = epochs.plot(title=fig_title, scalings=scalings)

#%%
evoked_1 = epochs['left'].average()
fig = evoked_1.plot()

#%%
epochs['left'].plot_image(combine='mean', vmin=-100, vmax=100,
                             ts_args=dict(ylim=dict(hbo=[-1000, 1000],
                                                    hbr=[-1000, 1000])))


fig = raw_data.plot_psd(average=True)
fig.suptitle('Before filtering', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)
raw_haemo = raw_data.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                             l_trans_bandwidth=0.02)
fig = raw_data.plot_psd(average=True)
fig.suptitle('After filtering', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)