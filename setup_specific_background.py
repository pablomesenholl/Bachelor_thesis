import ROOT
from ROOT import Math
import numpy as np
import matplotlib.pyplot as plt

# define most important features: m_TT, m_Kstar, m_B0, VertexChi2, Isolation

# masses from PDG
m_mu = 0.10565 #GeV
m_tau = 1.77693 #GeV
m_C_pion = 0.139570 #GeV
m_pion0 = 0.134977 #GeV
m_Kstar = 0.89555 #GeV

#load and inspect ROOT file into RDataFrame
file = ROOT.TFile.Open("Specific_Background_Smeared.root")
file.ls()

#find TTree called example
tree = file.Get("Events")
#load TTree into RDataFrame
df = ROOT.RDataFrame(tree)
#get info on dataframe
df.Describe().Print()

print("Number of entries in the TTree:", df.Count().GetValue())


# Function to compute invariant mass from 3 FourVectors
ROOT.gInterpreter.Declare("""
double InvariantMass3(const ROOT::Math::PtEtaPhiMVector& v1,
                      const ROOT::Math::PtEtaPhiMVector& v2,
                      const ROOT::Math::PtEtaPhiMVector& v3) {
    return (v1 + v2 + v3).M();
}
""")

#create 4vector for tauons and Kstar
df = df.Define("tauPlus_lv", "ROOT::Math::PtEtaPhiMVector(tauPlus_pt, tauPlus_eta, tauPlus_phi, m_tauPlus)") #adjust pt, eta, phi for correct particle
df = df.Define("tauMinus_lv", "ROOT::Math::PtEtaPhiMVector(tauMinus_pt, tauMinus_eta, tauMinus_phi, m_tauMinus)") #same here

df = df.Define("Kstar_lv", "ROOT::Math::PtEtaPhiMVector(kst_pt, kst_eta, kst_phi, m_kst)") #adjust pt, eta, phi for Kstar

#compute invariant mass of B^0 with 3 tracks from Kstar tau tau
df = df.Define("invMassB0",
    "InvariantMass3(tauPlus_lv, tauMinus_lv, Kstar_lv)"
)

#compute invariant masses of TT, KstarT-, KstarT+ with 2 tracks
df = df.Define("invMassTT", "ROOT::Math::VectorUtil::InvariantMass(tauPlus_lv, tauMinus_lv)") #mTT
df = df.Define("invMassKstarTPlus", "ROOT::Math::VectorUtil::InvariantMass(tauPlus_lv, Kstar_lv)") #mT+Kstar
df = df.Define("invMassKstarTMinus", "ROOT::Math::VectorUtil::InvariantMass(tauMinus_lv, Kstar_lv)") #mT-Kstar

#B^0 candidate kinematics (if not in data yet)
df = df.Define("B_lv", "tauPlus_lv + tauMinus_lv + Kstar_lv") #B0 lorentz vector
df = df.Define("pt_B0", "B_lv.Pt()")
df = df.Define("eta_B0", "B_lv.Eta()")
df = df.Define("phi_B0", "B_lv.Phi()")

#angular separations Delta_R
df = df.Define("dR_TPlusTMinus", "ROOT::Math::VectorUtil::DeltaR(tauPlus_lv, tauMinus_lv)")
df = df.Define("dR_TPlusKstar", "ROOT::Math::VectorUtil::DeltaR(tauPlus_lv, Kstar_lv)")
df = df.Define("dR_TMinusKstar", "ROOT::Math::VectorUtil::DeltaR(tauMinus_lv, Kstar_lv)")

#assume we have VertexChi2 and primary- (PV) and secondary vertices (SV) of B0
#construct flight length features and pointing angle of B0
df = df.Define("Lx",  "SVx  - PVx") \
       .Define("Ly",  "SVy  - PVy") \
       .Define("Lz",  "SVz  - PVz") #gives vector from PV to SV of B0

df = df.Define("flightLength3D",
               "sqrt(Lx*Lx + Ly*Ly + Lz*Lz)") #flight length of B0

df = df.Define("pointingCos",
    " (B_lv.X()*Lx + B_lv.Y()*Ly + B_lv.Z()*Lz) / (B_lv.P() * flightLength3D)") #cos(alpha) = p_B dot L / |p_B| * |L|, alignement of flight and momentum vector, the closer to 1, the better
#implement flight length significance (L/sigma_L), if you have error on flight length
# df = df.Define("flightLengthSig",
#                "flightLength3D / flightLengthError")  # make sure you have flightLengthError

#VertexChi2 should be a standalone feature already implemented

#Transverse Flight Length of the B0
df = df.Define("transFlightLength", "sqrt(Lx*Lx + Ly*Ly)")

#reconstructed proper tau of the B meson in xy plane
df = df.Define("B0_t_xy", "(transFlightLength * invMassB0)/pt_B0")

#NEXT: implement helicity angle

#add the type of signal/background label to the df for the classifier
df = df.Define("label", "1")


all_cols = list(df.GetColumnNames())     # correct feature names
for old in ("etaB", "phiB", "ptB"):
    all_cols.remove(old)

#save the full RDataFrame in a new root file
df.Snapshot("tree", "Spec_Bckg_features.root", all_cols)

#Now do a bunch of histos to check if simulation went correctly
mass_B0_h = df.Histo1D(("mass_B0_histo", "B0 Mass distribution", 100, 1, 5), "invMassB0")
mass_TT_h = df.Histo1D(("mass_TT_histo", "TT Mass distribution", 100, 0, 4), "invMassTT")
eta_B0_h = df.Histo1D(("eta_B0_histo", "B0 eta distribution", 100, -10, 10), "eta_B0")
eta_TPLus_h = df.Histo1D(("eta_TPLus_histo", "TPlus eta distribution", 100, -10, 10), "tauPlus_eta")
phi_B0_h = df.Histo1D(("phi_B0_histo", "B0 phi distribution", 100, -1, 1), "phi_B0")
pt_B0_h = df.Histo1D(("pt_B0_histo", "B0 pt distribution", 100, 0, 11), "pt_B0")
VertexChi2_h = df.Histo1D(("VertexChi2_histo", "VertexChi2 distribution", 100, 0, 4), "vertexChi2")
eta_Kstar_h = df.Histo1D(("eta_Kstar_histo", "Kstar eta distribution", 100, -10, 10), "kst_eta")
pointingCos_h = df.Histo1D(("PointingCos_histo", "PointingCos distribution", 100, -1, 1), "pointingCos")
flightLength_h = df.Histo1D(("flightLength_histo", "flightLength distribution", 100, 0, 40), "flightLength3D")
m_kst_h = df.Histo1D(("Kstar inv mass histo", "Kstar inv mass distribution", 100, 0, 2), "m_kst")
B0_t_h = df.Histo1D(("B0 proper time histo", "B0 proper time distribution", 100, 0, 3), "B0_t")
Lxy_h = df.Histo1D(("B0 transverse flight lenght histo", "B0 trans. flight length distribution", 100, 0, 4), "transFlightLength")
B0_t_xy = df.Histo1D(("B0 proper transverse time histo", "B0 trans. proper time distribution", 100, 0, 3), "B0_t_xy")
mass_Kst_TPlus_h = df.Histo1D(("Kst TPlus inv mass histo", "inv mass Kst TPlus distribution", 100, 0, 5), "invMassKstarTPlus")
mass_Kst_TMinus_h = df.Histo1D(("Kst TMinus inv mass histo", "inv mass Kst TMinus distribution", 100, 0, 5), "invMassKstarTMinus")


#save histograms
c = ROOT.TCanvas()
mass_B0_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/B0 mass distribution.png")

c = ROOT.TCanvas()
mass_TT_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/TT mass distribution.png")

c = ROOT.TCanvas()
VertexChi2_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/VertexChi2 distribution.png")

c = ROOT.TCanvas()
eta_B0_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/B0 eta distribution.png")

c = ROOT.TCanvas()
eta_TPLus_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/TPlus eta distribution.png")

c = ROOT.TCanvas()
phi_B0_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/B0 phi distribution.png")

c = ROOT.TCanvas()
pt_B0_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/B0 pT distribution.png")

c = ROOT.TCanvas()
eta_Kstar_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/Kstar eta distribution.png")

c = ROOT.TCanvas()
pointingCos_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/pointingCos distribution.png")

c = ROOT.TCanvas()
flightLength_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/flightLength distribution.png")

c = ROOT.TCanvas()
m_kst_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/m_kst distribution.png")

c = ROOT.TCanvas()
m_kst_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/m_kst distribution.png")

c = ROOT.TCanvas()
B0_t_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/B0_t distribution.png")

c = ROOT.TCanvas()
Lxy_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/Lxy distribution.png")

c = ROOT.TCanvas()
B0_t_xy.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/B0 t_xy distribution.png")

c = ROOT.TCanvas()
mass_Kst_TPlus_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/Kst TPlus mass distribution.png")

c = ROOT.TCanvas()
mass_Kst_TMinus_h.Draw()
c.Update()
c.SaveAs("Plots_spec_bckg_simulation/Kst TMinus mass distribution.png")