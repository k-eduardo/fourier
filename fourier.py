import pandas as pd
import matplotlib.pyplot as plt

def fplot(df, column=False):
    if not column:
        df.plot(subplots=True)

print("Fourier Series for Certain Examples")
print("Dataloop SA de CV")
print("Pandas version: "+pd.__version__)

datasets = [["ESR.csv","Electric Seizure Recognition"],["HPC.txt","Household Power Consumption"],["HHED.csv","Household Energy Data"]]

dfs = {}

for fname,name in datasets:
    print("\n       ---\n")
    vname = fname.split(".")[0]
    dfs[vname] = pd.read_csv(fname, header=0)
    print("Just loaded dataset "+name+" with "+str(dfs[vname].shape[0])+" rows and "+str(dfs[vname].shape[1])+" columns.")
    print("Head Structure: "+str(dfs[vname].columns[0]))
    print("Variable name: dfs."+vname)
    print("Memory Usage: {:.2f} MB".format(dfs[vname].memory_usage(deep=True).sum()/1000000))

dfs["ESR"].plot()
plt.show()
