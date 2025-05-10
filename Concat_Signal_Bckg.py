import ROOT

#load and inspect ROOT file into RDataFrame
file = ROOT.TFile.Open("merged.root")
file.ls()

#find TTree called example
tree = file.Get("myTree")
#load TTree into RDataFrame
df = ROOT.RDataFrame(tree)
#get info on dataframe
df.Describe().Print()

print("Number of entries in the TTree:", df.Count().GetValue())

# list of your three files
files = [
    "Signal_features.root",
    "Spec_Bckg_features.root",
    "Comb_Bckg_features.root",
]

# build your chained RDataFrame
df = ROOT.RDataFrame("tree", files)

# write out _all_ branches into merged.root
df.Snapshot("myTree", "merged.root")

print("Wrote merged.root")

