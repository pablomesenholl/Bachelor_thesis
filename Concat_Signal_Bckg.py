import ROOT

#load and inspect ROOT file into RDataFrame
file = ROOT.TFile.Open("merged.root")
file.ls()

# find TTree called example
tree = file.Get("myTree")
# load TTree into RDataFrame
df = ROOT.RDataFrame(tree)
# get info on dataframe
df.Describe().Print()

# print("Number of entries in the TTree:", df.Count().GetValue())

# list of your three files
files = [
    "Signal_features.root",
    "Spec_Bckg_features.root",
    "Comb_Bckg_features.root",
]

df1 = ROOT.RDataFrame("tree", "Signal_features.root")
df1.Describe().Print()

df2 = ROOT.RDataFrame("tree", "Spec_Bckg_features.root")
df2.Describe().Print()

df3 = ROOT.RDataFrame("tree", "Comb_Bckg_features.root")
df3.Describe().Print()

# build your chained RDataFrame
df = ROOT.RDataFrame("tree", files)

# write out _all_ branches into merged.root
df.Snapshot("myTree", "merged.root")

print("Wrote merged.root")

