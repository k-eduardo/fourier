import pandas as pd
from fbprophet import Prophet
import numpy as np
from numpy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt

def fplot(df, column=False):
    if not column:
        df.plot(subplots=True)

print("Fourier Series for Certain Examples")
print("Dataloop SA de CV")
print("Pandas version: "+pd.__version__)

datasets = [["ESR.csv","Electric Seizure Recognition"],["HPC.txt","Household Power Consumption"],["HHED.csv","Household Energy Data"],["example_yosemite_temps.csv","Yosemite Temperatures"]]

dfs = {}

for fname,name in datasets:
    print("\n       ---\n")
    vname = fname.split(".")[0]
    dfs[vname] = pd.read_csv("data/"+fname, header=0)
    print("Just loaded dataset "+name+" with "+str(dfs[vname].shape[0])+" rows and "+str(dfs[vname].shape[1])+" columns.")
    print("Head Structure: "+str(dfs[vname].columns[0]))
    print("Variable name: dfs."+vname)
    print("Memory Usage: {:.2f} MB".format(dfs[vname].memory_usage(deep=True).sum()/1000000))

# Points
N = 10000
# N = 18721

# Domain Length (in seconds)
L = 2000
# L = 5616000

# Angular frequency
w = 2.*np.pi/L

x = np.linspace(0,L,N)
y1 = 3.*np.cos(5*w*x)
y2 = 2.*np.cos(10*w*x)
y3 = 1.*np.cos(12*w*x)

y = y1 + y2 + y3
# y = np.array(dfs["example_yosemite_temps"]["y"])
freqs = fftfreq(N)

print(x)
print(y)

mask = freqs > 0

fft_vals = fft(y)
print(fft_vals)

fft_t = 2.*np.abs(fft_vals/N)

plt.figure(1)
plt.title('Original Signal')
plt.plot(x,y,color='xkcd:salmon',label='original')
plt.legend()

# Uncomment for second part of exercise
plt.figure(2)
plt.title("FFT")
plt.scatter(freqs[mask]/w,fft_t[mask],label='fft')
plt.legend()

plt.show()
