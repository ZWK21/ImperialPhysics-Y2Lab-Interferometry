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
file = r'C:\Users\zhaow\OneDrive - Imperial College London\Labs\Y2\Interferometry\Data\BestDataSoFarDONTTOUCH.txt'
results = rd.read_data3(file)

y1 = np.array(results[0])
y2 = np.array(results[1])

x=np.array(results[5])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# --- Top Plot (Detector 1) ---
ax1.plot(x, y1, 'o-', label="Detector 1")
ax1.set_ylabel("Signal 1")
ax1.set_title("Detector Signals (Linked X-Axis)")

# --- Bottom Plot (Detector 2) ---
ax2.plot(x, y2, 'o-', color='C1', label="Detector 2")
ax2.set_xlabel("Position $\mu$steps")
ax2.set_ylabel("Signal 2")

plt.tight_layout()
plt.show()