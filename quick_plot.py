###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import read_data_results3 as rd
import scipy.fftpack as spf

#Step 1 get the data and the x position
# file='%s'%(sys.argv[1]) #this is the data
file = r"C:\Users\zhaow\OneDrive - Imperial College London\Labs\Y2\Interferometry\Data\interferometry_data_20251020_092214.txt"
results = rd.read_data3(file)

y1 = np.array(results[0])
y2 = np.array(results[1])

x=np.array(results[5])

plt.figure("Detector 1")
plt.plot(x,y1,'o-')
# plt.xlim(-5.2e6,-4.8e6)
plt.xlabel("Position $\mu$steps")
plt.ylabel("Signal 1")
# plt.savefig("figures/quick_plot_detector_1.png")

plt.figure("Detector 2")
plt.plot(x,y2,'o-')
plt.xlabel("Position $\mu$steps")
plt.ylabel("Signal 2")
# plt.savefig("figures/quick_plot_detector_2.png")

plt.show()