import ROOT
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#folder to save all pngs
folder = "Plots_generator_MC_analysis2"

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

#get number of events that contain B^0
n_events_total = df.Count().GetValue()
print("Number of events", n_events_total)

#look at B0 eta distribution
df = df.Define("B0_mask", "abs(pdgId) == 511")#define mask to find B0
df = df.Define("B0_eta", "eta[B0_mask]")#apply mask on eta

hist_eta_B0 = df.Histo1D(("eta of B0", "eta distr of B0", 100, -10, 10), "B0_eta")
hist_eta = df.Histo1D(("eta distribution", "eta distribution", 100, -10, 10), "eta")
hist_eta_B0.SetLineColor(ROOT.kRed)
hist_eta.SetLineColor(ROOT.kBlue)

max_y = max(hist_eta_B0.GetValue().GetMaximum(), hist_eta.GetValue().GetMaximum())
hist_eta_B0.GetValue().SetMaximum(1.1 * max_y)
hist_eta.GetValue().SetMaximum(1.1 * max_y)

# c = ROOT.TCanvas()
# hist_eta_B0.Draw("HIST")
# hist_eta.Draw("HIST SAME")
# legend = ROOT.TLegend(0.7,0.7,0.9,0.9)
# legend.AddEntry(hist_eta_B0.GetValue(), "eta distr of B0", "l")
# legend.AddEntry(hist_eta.GetValue(), "total eta distr", "l")
# legend.Draw()
# c.SetLogy()
# c.SaveAs("eta of B0 distribution.png")

#sort pions, tauons, B^0, Kstar, muons from events
#define mask to get alls relevant particles

df = df.Define("particle_mask", "abs(pdgId) == 511 || abs(pdgId) == 313 || abs(pdgId) == 211 || abs(pdgId) == 15 || abs(pdgId) == 13")

#apply mask on pdgId and mass
df = df.Define("pdgId_part", "abs(pdgId[particle_mask])")
df = df.Define("mass_part", "mass[particle_mask]")


#count Kstar, pions, muons, neutrinos per event and in total
numKstar = df.Define("numKstar", "ROOT::VecOps::Sum(abs(pdgId) == 313)").Sum("numKstar").GetValue()
df = df.Define("numB0", "ROOT::VecOps::Sum(abs(pdgId) == 511)")
numB0 = df.Sum("numB0").GetValue()
numPion = df.Define("numPion", "ROOT::VecOps::Sum(abs(pdgId) == 211)").Sum("numPion").GetValue()
numTau = df.Define("numTau", "ROOT::VecOps::Sum(abs(pdgId) == 15)").Sum("numTau").GetValue()
numMuon = df.Define("numMuon", "ROOT::VecOps::Sum(abs(pdgId) == 13)").Sum("numMuon").GetValue()
numNMuon = df.Define("numNMuon", "ROOT::VecOps::Sum(abs(pdgId) == 14)").Sum("numNMuon").GetValue()
numNTau = df.Define("numNTau", "ROOT::VecOps::Sum(abs(pdgId) == 16)").Sum("numNTau").GetValue()
numE = df.Define("numE", "ROOT::VecOps::Sum(abs(pdgId) == 11)").Sum("numE").GetValue()
numNE = df.Define("numNE", "ROOT::VecOps::Sum(abs(pdgId) == 12)").Sum("numNE").GetValue()

#create histo of number of B0 per event
histo_numB0 = df.Histo1D(("numB0 per event", "number of B0 per event", 100, 0, 5), "numB0")
c = ROOT.TCanvas()
histo_numB0.Draw()
c.SaveAs("Plots_generator_MC_analysis2/Number of B0 per event.png")


print("Number of Kstar:", numKstar,
      "\nNumber of B0:", numB0,
      "\nNumber of Pions:", numPion,
      "\nNumber of Tauons:", numTau,
      "\nNumber of Muons:", numMuon,
      "\nNumber of muon Neutrinos:", numNMuon,
      "\nNumber of tau Neutrinos:", numNTau,
      "\nNumber of Electrons:", numE,
      "\nNumber of electron Neutrinos:", numNE)

#create histo of B0 mass distribution
df = df.Define("B0_mass", "mass[B0_mask]")
hist_B0_mass = df.Histo1D(("B0 mass histo", "B0 mass distribution", 100, 4, 6), "B0_mass")
# c = ROOT.TCanvas()
# hist_B0_mass.Draw()
# c.SaveAs("B0 mass distribution.png")

#create histo of B0 pt distribution
df = df.Define("B0_pt", "pt[B0_mask]")
h = df.Histo1D(("B0 pt", "B0 pt distr", 100, 0, 70), "B0_pt")
# c = ROOT.TCanvas()
# h.Draw()
# c.SaveAs("B0 pt distribution.png")

#create histo of all muons pt, find trigger cut
df = df.Define("pt_muon_mask", "pt[abs(pdgId) == 13]")
pt_muon_max = df.Define("pt_max", "ROOT::VecOps::Max(pt_muon_mask)").Max("pt_max").GetValue()
print("Maximum pT of all muons:", pt_muon_max, "There seems to be a trigger cut for the pT of the muons") #max at 12.870697975158691
h_pt = df.Histo1D(("muon pt", "muon pt distr", 100, 0, 20), "pt_muon_mask")
# c = ROOT.TCanvas()
# h_pt.Draw()
# c.SaveAs("muon pt distribution.png")


#try to find selection of total decay mode
df = df.Define("numDaughters_B0_mask", "abs(pdgId) == 511 && numDaughters == 3")
df = df.Define("numDaughters_B0", "numDaughters[numDaughters_B0_mask]")#number of Daughters of B0 decays
numRealB0 = df.Define("numRealB0", "ROOT::VecOps::Sum(numDaughters_B0_mask)").Sum("numRealB0").GetValue()
print("Number of real B0:", numRealB0)

#Figure out number of daughters of tau decays coming from B0
df = df.Define("tau_3_mask", " abs(pdgId) == 15 && abs(motherPdgId) == 511 && numDaughters == 3")
df = df.Define("tau_4_mask", " abs(pdgId) == 15 && abs(motherPdgId) == 511 && numDaughters == 4")#tau from B0 & number of Daughters of tau 3prong decay mask
df = df.Define("tau_5_mask", " abs(pdgId) == 15 && abs(motherPdgId) == 511 && numDaughters == 5")
numTau3 = df.Define("numTau3", "ROOT::VecOps::Sum(tau_3_mask)").Sum("numTau3").GetValue()
numTau4 = df.Define("numTau4", "ROOT::VecOps::Sum(tau_4_mask)").Sum("numTau4").GetValue()
numTau5 = df.Define("numTau5", "ROOT::VecOps::Sum(tau_5_mask)").Sum("numTau5").GetValue()
print("Number of real Tau decays with 3 daughters:", numTau3,
      "\nNumber of real Tau decays with 4 daughters:", numTau4,
      "\nNumber of real Tau decays with 5 daughters:", numTau5)

#define all masks and total mask
df = df.Define("Tau_mask", "abs(pdgId) == 15 && abs(motherPdgId) == 511")#Tau mask
df = df.Define("Kstar_mask", "abs(pdgId) == 313 && abs(motherPdgId) == 511")#Kstar mask
df = df.Define("pion_mask", "abs(pdgId) == 211 && abs(motherPdgId) == 15")#pion mask
df = df.Define("muon_mask", "abs(pdgId) == 13 && abs(motherPdgId) == 15")#muon mask
df = df.Define("total_mask", "Tau_mask || Kstar_mask || pion_mask || muon_mask")#total mask

#apply total mask on relevant parameters (mass, eta and pt)
df = df.Define("eta_total", "eta[total_mask]")
df = df.Define("pt_total", "pt[total_mask]")

#create histogram of decay particles
eta_histo_total = df.Histo1D(("eta histo", "total eta distribution (Tau, Kstar, Muons, Pions)", 100, -2, 2), "eta_total")
pt_histo_total = df.Histo1D(("pt histo", "total pt distribution (Tau, Kstar, Muons, Pions)", 100, 0, 30), "pt_total")

# c = ROOT.TCanvas()
# eta_histo_total.Draw()
# c.SaveAs("total eta distribution.png")

# c = ROOT.TCanvas()
# pt_histo_total.Draw()
# c.SaveAs("total pt distribution.png")