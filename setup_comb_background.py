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
file = ROOT.TFile.Open("Combinatorial_Background.root")
file.ls()

#find TTree called example
tree = file.Get("Events")
#load TTree into RDataFrame
df = ROOT.RDataFrame(tree)
#get info on dataframe
df.Describe().Print()

print("Number of entries in the TTree:", df.Count().GetValue())

#adjust branch names
df = (
    df
    .Define("tauPlus_eta", "tau1_eta")
    .Define("tauPlus_phi", "tau1_phi")
    .Define("tauPlus_pt", "tau1_pt")
    .Define("tauMinus_eta", "tau3_eta")
    .Define("tauMinus_phi", "tau3_phi")
    .Define("tauMinus_pt", "tau3_pt")
)

#make function to create lorentz vectors for different masses
def make_lv_factory(mass):
    def make_lv(pt, eta, phi, mass):
        return Math.PtEtaPhiMVector(pt, eta, phi, mass)
    return make_lv

# Function to define 4 vectors
ROOT.gInterpreter.Declare(r"""
  ROOT::Math::PtEtaPhiMVector makeTauLV(double pt, double eta, double phi) {
    return ROOT::Math::PtEtaPhiMVector(pt,eta,phi,1.77693);
  }
  ROOT::Math::PtEtaPhiMVector makeKstarLV(double pt, double eta, double phi) {
    return ROOT::Math::PtEtaPhiMVector(pt,eta,phi,0.89555);
  }
""")

# Function to compute invariant mass from 3 FourVectors
ROOT.gInterpreter.Declare("""
double InvariantMass3(const ROOT::Math::PtEtaPhiMVector& v1,
                      const ROOT::Math::PtEtaPhiMVector& v2,
                      const ROOT::Math::PtEtaPhiMVector& v3) {
    return (v1 + v2 + v3).M();
}
""")

#create 4vector for tauons and Kstar
make_lv_tau = make_lv_factory(m_tau) #define factory for tau mass
df = df.Define("tauPlus_lv", "makeTauLV(tau1_pt, tau1_eta, tau1_phi)") #adjust pt, eta, phi for correct particle
df = df.Define("tauMinus_lv", "makeTauLV(tau3_pt, tau3_eta, tau3_phi)") #same here

make_lv_Kstar = make_lv_factory(m_Kstar) #define factory for Kstar mass
df = df.Define("Kstar_lv", "makeKstarLV(kst_pt, kst_eta, kst_phi)") #adjust pt, eta, phi for Kstar

#compute invariant mass of B^0 with 3 tracks from Kstar tau tau
df = df.Define("invMassB0",
    "InvariantMass3(tauPlus_lv, tauMinus_lv, Kstar_lv)"
)

#compute invariant masses of TT, KstarT-, KstarT+ with 2 tracks
df = df.Define("invMassTT", "ROOT::Math::VectorUtil::InvariantMass(tauPlus_lv, tauMinus_lv)") #mTT
df = df.Define("invMassKstarTPLus", "ROOT::Math::VectorUtil::InvariantMass(tauPlus_lv, Kstar_lv)") #mT+Kstar
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


#NEXT: implement helicity angle

#add the type of signal/background label to the df for the classifier
df = df.Define("label", "2")

#save the full RDataFrame in a new root file
all_cols = list(df.GetColumnNames())     # correct feature names
for old in ("tau1_eta", "tau1_pt", "tau1_phi", "tau3_eta", "tau3_pt", "tau3_phi"):
    all_cols.remove(old)

df.Snapshot("tree", "Comb_Bckg_features.root", all_cols)


#Now do a bunch of histos to check if simulation went correctly
mass_B0_h = df.Histo1D(("mass_B0_histo", "B0 Mass distribution", 100, 3, 15), "invMassB0")
mass_TT_h = df.Histo1D(("mass_TT_histo", "TT Mass distribution", 100, 3, 15), "invMassTT")
eta_B0_h = df.Histo1D(("eta_B0_histo", "B0 eta distribution", 100, -10, 10), "eta_B0")
eta_TPLus_h = df.Histo1D(("eta_TPLus_histo", "TPlus eta distribution", 100, -10, 10), "tau1_eta")
phi_B0_h = df.Histo1D(("phi_B0_histo", "B0 phi distribution", 100, -np.pi, np.pi), "phi_B0")
pt_B0_h = df.Histo1D(("pt_B0_histo", "B0 pt distribution", 100, 0, 20), "pt_B0")
VertexChi2_h = df.Histo1D(("VertexChi2_histo", "VertexChi2 distribution", 100, 0, 20), "vertexChi2")
eta_Kstar_h = df.Histo1D(("eta_Kstar_histo", "Kstar eta distribution", 100, -10, 10), "kst_eta")
eta_T3_h = df.Histo1D(("eta_T3_histo", "3prong eta distribution", 100, -10, 10), "tau3_eta")
pointingCos_h = df.Histo1D(("PointingCos_histo", "PointingCos distribution", 100, -np.pi, np.pi), "pointingCos")
flightLength_h = df.Histo1D(("flightLength_histo", "flightLength distribution", 100, 0, 100), "flightLength3D")


#save histograms
c = ROOT.TCanvas()
mass_B0_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/B0 mass distribution.png")

c = ROOT.TCanvas()
mass_TT_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/TT mass distribution.png")

c = ROOT.TCanvas()
VertexChi2_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/VertexChi2 distribution.png")

c = ROOT.TCanvas()
eta_B0_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/B0 eta distribution.png")

c = ROOT.TCanvas()
eta_TPLus_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/TPlus eta distribution.png")

c = ROOT.TCanvas()
phi_B0_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/B0 phi distribution.png")

c = ROOT.TCanvas()
pt_B0_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/B0 pT distribution.png")

c = ROOT.TCanvas()
eta_Kstar_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/Kstar eta distribution.png")

c = ROOT.TCanvas()
eta_T3_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/3prong eta distribution.png")

c = ROOT.TCanvas()
pointingCos_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/pointingCos distribution.png")

c = ROOT.TCanvas()
flightLength_h.Draw()
c.Update()
c.SaveAs("Plots_comb_bckg_simulation/flightLength distribution.png")