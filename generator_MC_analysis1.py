import ROOT
import uproot

#folder to save all pngs
folder = "Plots_generator_MC_analysis1"

#open ROOT file
file = ROOT.TFile.Open("genparticles.root")

#found TDirectory called genAnalyzer
folder = file.Get("genAnalyzer")

#found TTree called GenParticles
tree = folder.Get("GenParticles")

#load TTree into RDataFrame
df = ROOT.RDataFrame(tree)

#get info on dataframe
df.Describe().Print()

#display dataframe
df.Display().Print()

#print number of rows of dataframe
print("Number of rows of df: ", df.Count().GetValue())

#determine max and min of pt, eta, phi and mass
mass_max = df.Define("mass_max", "ROOT::VecOps::Max(mass)").Max("mass_max").GetValue()
print("Mass max: ", mass_max)
pt_max = df.Define("pt_max", "ROOT::VecOps::Max(pt)").Max("pt_max").GetValue()
print("Pt max: ", pt_max)
eta_max = df.Define("eta_max", "ROOT::VecOps::Max(eta)").Max("eta_max").GetValue()
print("Eta max: ", eta_max)
phi_max = df.Define("phi_max", "ROOT::VecOps::Max(phi)").Max("phi_max").GetValue()
print("Phi max: ", phi_max)
mass_min = df.Define("mass_min", "ROOT::VecOps::Min(mass)").Min("mass_min").GetValue()
print("Mass min: ", mass_min)
eta_min = df.Define("eta_min", "ROOT::VecOps::Min(eta)").Min("eta_min").GetValue()
print("Eta min: ", eta_min)
phi_min = df.Define("phi_min", "ROOT::VecOps::Min(phi)").Min("phi_min").GetValue()
print("Phi min: ", phi_min)
pt_min = df.Define("pt_min", "ROOT::VecOps::Min(pt)").Min("pt_min").GetValue()
print("Pt min: ", pt_min)


#create histos of eta, pt, phi and mass
#histmodel = (name, title, nbins, xmin, xmax)
mass_h = df.Histo1D(("mass_histo", "Mass distribution", 100, mass_min, mass_max), "mass")
pt_h = df.Histo1D(("pt_histo", "p_T distribution", 100, pt_min, pt_max), "pt")
eta_h = df.Histo1D(("eta_histo", "eta distribution", 100, eta_min, eta_max), "eta")
phi_h = df.Histo1D(("phi_histo", "phi distribution", 100, phi_min, phi_max), "phi")

#adjust eta column
df = df.Define("good_eta", "eta[(eta > -10 & eta < 10)]")
good_eta_h = df.Histo1D(("good_eta_h", "good_eta distribution", 100, -10, 10), "good_eta")


#save histograms
# c = ROOT.TCanvas()
# c.SetLogy()
# mass_h.Draw()
# c.Update()
# c.SaveAs("mass distribution.png")
#
# c = ROOT.TCanvas()
# c.SetLogy()
# pt_h.Draw()
# c.Update()
# c.SaveAs("pt distribution.png")
#
# c = ROOT.TCanvas()
# good_eta_h.Draw()
# c.Update()
# c.SaveAs("eta distribution.png")
#
# c = ROOT.TCanvas()
# phi_h.Draw()
# c.Update()
# c.SaveAs("phi distribution.png")




#find K^*0 = 313 from B^0 = 511
df_Kstar = df.Define("Kstar_mask", "abs(pdgId) == 313 && abs(motherPdgId) == 511") #define mask to identify Kstar from B^0
numKstar = df.Define("numKstar", "ROOT::VecOps::Sum(abs(pdgId) == 313 && abs(motherPdgId) == 511)").Sum("numKstar").GetValue()
print("Number of Kstar from B0:", numKstar)

df_Kstar = df_Kstar.Define("pdg_mask", "pdgId[Kstar_mask]") #apply mask
df_Kstar = df_Kstar.Define("motherpdg_mask", "motherPdgId[Kstar_mask]")
df_Kstar = df_Kstar.Define("mass_mask", "mass[Kstar_mask]")
df_Kstar = df_Kstar.Define("eta_mask", "eta[Kstar_mask]")
df_Kstar = df_Kstar.Define("phi_mask", "phi[Kstar_mask]")
df_Kstar = df_Kstar.Define("pt_mask", "pt[Kstar_mask]")

#plot mass distibution of Kstar
Kstar_mass_histo = df_Kstar.Histo1D(("Kstar_mass_histo", "Kstar_mass", 100, 0.7, 1.1), "mass_mask")
# c = ROOT.TCanvas()
# Kstar_mass_histo.Draw()
# c.SaveAs("Kstar mass distribution.png")

#plot pt distribution of Kstar
h = df_Kstar.Histo1D(("Kstar pt", "Kstar pt distr", 100, 0, 20), "pt_mask")
# c = ROOT.TCanvas()
# h.Draw()
# c.SaveAs("Kstar pt distribution.png")

#find pions = 211 from tau = 15
df_pion = df.Define("pion_mask", "abs(pdgId) == 211 && abs(motherPdgId) == 15") #define mask to identify pions from tau
numPion = df.Define("numPion", "ROOT::VecOps::Sum(abs(pdgId) == 211 && abs(motherPdgId) == 15)").Sum("numPion").GetValue()
print("Number of Pions from Tau:", numPion)#count pions from tau

df_pion = df_pion.Define("pdg_mask", "pdgId[pion_mask]")
df_pion = df_pion.Define("motherpdg_mask", "motherPdgId[pion_mask]")
df_pion = df_pion.Define("eta_mask", "eta[pion_mask]")
df_pion = df_pion.Define("pt_mask", "pt[pion_mask]")
df_pion = df_pion.Define("phi_mask", "phi[pion_mask]")
df_pion = df_pion.Define("mass_mask", "mass[pion_mask]")

#plot mass distribution of pions
h = df_pion.Histo1D(("pion mass", "pion mass", 100, 0, 0.2), "mass_mask")
# c = ROOT.TCanvas()
# h.Draw()
# c.SaveAs("pion mass distribution.png")

#plot pt distribution of pions
h = df_pion.Histo1D(("pion pt", "pion pt distr", 100, 0, 20), "pt_mask")
# c = ROOT.TCanvas()
# h.Draw()
# c.SaveAs("pion pt distribution.png")

#find muons = 13 from tau = 15
df_tau_muon = df.Define("muon_mask", "abs(pdgId) == 13 && abs(motherPdgId) == 15") #define mask to find muon from tau
numMuon = df.Define("numMuon", "ROOT::VecOps::Sum(abs(pdgId) == 13 && abs(motherPdgId) == 15)").Sum("numMuon").GetValue()
print("Number of Muons from Tau:", numMuon)

df_tau_muon = df_tau_muon.Define("pdg_mask", "pdgId[muon_mask]") #apply mask to columns
df_tau_muon = df_tau_muon.Define("motherpdg_mask", "motherPdgId[muon_mask]")
df_tau_muon = df_tau_muon.Define("eta_mask", "eta[muon_mask]")
df_tau_muon = df_tau_muon.Define("mass_mask", "mass[muon_mask]")
df_tau_muon = df_tau_muon.Define("phi_mask", "phi[muon_mask]")
df_tau_muon = df_tau_muon.Define("pt_mask", "pt[muon_mask]")

#mass distribution of muons from tau
h = df_tau_muon.Histo1D(("muon mass", "muon mass", 100, 0, 0.2), "mass_mask")
# c = ROOT.TCanvas()
# h.Draw()
# c.SaveAs("muon mass distribution.png")

#pT distribution of muons from tau, find cut
pt_muon_max = df_tau_muon.Define("pt_max", "ROOT::VecOps::Max(pt_mask)").Max("pt_max").GetValue()
# print("Maximum pT of muons from tau:", pt_muon_max, "There seems to be a trigger cut for the pT of the muons")
h_pt = df_tau_muon.Histo1D(("muon pt", "muon pt distr", 100, 0, 20), "pt_mask")
# c = ROOT.TCanvas()
# h_pt.Draw()
# c.SaveAs("muon pt distribution.png")


#find tau = 15 from B^0 = 511
df_tau_B = df.Define("tau_B_mask", " abs(pdgId) == 15 && abs(motherPdgId) == 511 ") #define mask for tau from B^0
numTau = df.Define("numTau", "ROOT::VecOps::Sum(abs(pdgId) == 15 && abs(motherPdgId) == 511)").Sum("numTau").GetValue()
print("Number of Tauons from B0:", numTau)#print number of tauons from B0

df_tau_B = df_tau_B.Define("pdg_mask", "pdgId[tau_B_mask]")
df_tau_B = df_tau_B.Define("eta_mask", "eta[tau_B_mask]")
df_tau_B = df_tau_B.Define("mass_mask", "mass[tau_B_mask]")
df_tau_B = df_tau_B.Define("phi_mask", "phi[tau_B_mask]")
df_tau_B = df_tau_B.Define("pt_mask", "pt[tau_B_mask]")

#mass distribution of tauons from B0
h = df_tau_B.Histo1D(("tau_B mass", "tau_B mass", 100, 1.5, 2), "mass_mask")
# c = ROOT.TCanvas()
# h.Draw()
# c.SaveAs("tau mass distribution.png")

#plot pt distribution of tauons
h = df_tau_B.Histo1D(("tau pt", "tau pt distr", 100, 0, 30), "pt_mask")
# c = ROOT.TCanvas()
# h.Draw()
# c.SaveAs("tau pt distribution.png")
