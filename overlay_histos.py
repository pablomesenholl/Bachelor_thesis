import ROOT
from ROOT import kRed, kBlue, kGreen

file0 = ROOT.TFile.Open("Signal_features.root")
file1 = ROOT.TFile.Open("Spec_Bckg_features.root")
file2 = ROOT.TFile.Open("Comb_Bckg_features.root")

file0.ls()
file1.ls()
file2.ls()

df0 = ROOT.RDataFrame("tree", file0)
df1 = ROOT.RDataFrame("tree", file1)
df2 = ROOT.RDataFrame("tree", file2)

#B0 invariant mass
nbin, nmin, nmax = 100, 0, 10

B0_mass_sign_h = df0.Histo1D(("B0 mass histo", "B0 invariant mass distr", nbin, nmin, nmax),"invMassB0")
B0_mass_spec_h = df1.Histo1D(("B0 mass histo", "B0 mass spec distr", nbin, nmin, nmax),"invMassB0")
B0_mass_comb_h = df2.Histo1D(("B0 mass histo", "B0 mass comb distr", nbin, nmin, nmax),"invMassB0")

B0_mass_sign_h = B0_mass_sign_h.GetValue()
B0_mass_spec_h = B0_mass_spec_h.GetValue()
B0_mass_comb_h = B0_mass_comb_h.GetValue()

for h in (B0_mass_sign_h, B0_mass_spec_h, B0_mass_comb_h):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)

B0_mass_sign_h.SetLineColor(kGreen)
B0_mass_spec_h.SetLineColor(kBlue)
B0_mass_comb_h.SetLineColor(kRed)

c = ROOT.TCanvas()
B0_mass_sign_h.Draw("HIST")
B0_mass_spec_h.Draw("HIST SAME")
B0_mass_comb_h.Draw("HIST SAME")

leg = ROOT.TLegend(0.60, 0.7, 0.45, 0.55)
leg.AddEntry(B0_mass_sign_h, "signal", "l")
leg.AddEntry(B0_mass_spec_h, "spec bckg", "l")
leg.AddEntry(B0_mass_comb_h, "comb bckg", "l")
leg.Draw()

c.Update()
c.SaveAs("Overlay_histos/B0 mass distr overlay.png")

#TT invariant mass
nbin, nmin, nmax = 100, 0, 8

TT_mass_sign_h = df0.Histo1D(("TT mass histo", "TT invariant mass distr", nbin, nmin, nmax),"invMassTT")
TT_mass_spec_h = df1.Histo1D(("TT mass histo", "TT mass spec distr", nbin, nmin, nmax),"invMassTT")
TT_mass_comb_h = df2.Histo1D(("TT mass histo", "TT mass comb distr", nbin, nmin, nmax),"invMassTT")

TT_mass_sign_h = TT_mass_sign_h.GetValue()
TT_mass_spec_h = TT_mass_spec_h.GetValue()
TT_mass_comb_h = TT_mass_comb_h.GetValue()

for h in (TT_mass_sign_h, TT_mass_spec_h, TT_mass_comb_h):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)

TT_mass_sign_h.SetLineColor(kGreen)
TT_mass_spec_h.SetLineColor(kBlue)
TT_mass_comb_h.SetLineColor(kRed)

c = ROOT.TCanvas()
TT_mass_sign_h.Draw("HIST")
TT_mass_spec_h.Draw("HIST SAME")
TT_mass_comb_h.Draw("HIST SAME")

leg = ROOT.TLegend(0.60, 0.7, 0.45, 0.55)
leg.AddEntry(B0_mass_sign_h, "signal", "l")
leg.AddEntry(B0_mass_spec_h, "spec bckg", "l")
leg.AddEntry(B0_mass_comb_h, "comb bckg", "l")
leg.Draw()

# c.SetLogy()
c.Update()
c.SaveAs("Overlay_histos/TT mass distr overlay.png")

#B0 pT
nbin, nmin, nmax = 100, 0, 14

B0_pt_sign_h = df0.Histo1D(("B0 pt histo", "B0 pt signal distr", nbin, nmin, nmax),"pt_B0")
B0_pt_spec_h = df1.Histo1D(("B0 pt histo", "B0 pt spec distr", nbin, nmin, nmax),"pt_B0")
B0_pt_comb_h = df2.Histo1D(("B0 pt histo", "B0 pT distr", nbin, nmin, nmax),"pt_B0")

B0_pt_sign_h = B0_pt_sign_h.GetValue()
B0_pt_spec_h = B0_pt_spec_h.GetValue()
B0_pt_comb_h = B0_pt_comb_h.GetValue()

for h in (B0_pt_sign_h, B0_pt_spec_h, B0_pt_comb_h):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)

B0_pt_sign_h.SetLineColor(kGreen)
B0_pt_spec_h.SetLineColor(kBlue)
B0_pt_comb_h.SetLineColor(kRed)

c = ROOT.TCanvas()
B0_pt_comb_h.Draw("HIST")
B0_pt_sign_h.Draw("HIST SAME")
B0_pt_spec_h.Draw("HIST SAME")

leg = ROOT.TLegend(0.60, 0.7, 0.45, 0.55)
leg.AddEntry(B0_pt_sign_h, "signal", "l")
leg.AddEntry(B0_pt_spec_h, "spec bckg", "l")
leg.AddEntry(B0_pt_comb_h, "comb bckg", "l")
leg.Draw()

c.Update()
c.SaveAs("Overlay_histos/B0 pt distr overlay.png")

# pointing Cos
nbin, nmin, nmax = 100, -1, 1

pointingCos_sign_h = df0.Histo1D(("pointing Cos", "pointing Cosine distr", nbin, nmin, nmax), "pointingCos")
pointingCos_spec_h = df1.Histo1D(("pointing Cos", "pointingCos spec distr", nbin, nmin, nmax), "pointingCos")
pointingCos_comb_h = df2.Histo1D(("pointing Cos", "pointingCos comb distr", nbin, nmin, nmax), "pointingCos")

pointingCos_sign_h = pointingCos_sign_h.GetValue()
pointingCos_spec_h = pointingCos_spec_h.GetValue()
pointingCos_comb_h = pointingCos_comb_h.GetValue()

for h in (pointingCos_sign_h, pointingCos_spec_h, pointingCos_comb_h):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)

pointingCos_sign_h.SetLineColor(kGreen)
pointingCos_spec_h.SetLineColor(kBlue)
pointingCos_comb_h.SetLineColor(kRed)

c = ROOT.TCanvas()
pointingCos_sign_h.Draw("HIST")
pointingCos_spec_h.Draw("HIST SAME")
pointingCos_comb_h.Draw("HIST SAME")

leg = ROOT.TLegend(0.60, 0.7, 0.45, 0.55)
leg.AddEntry(pointingCos_sign_h, "signal", "l")
leg.AddEntry(pointingCos_spec_h, "spec bckg", "l")
leg.AddEntry(pointingCos_comb_h, "comb bckg", "l")
leg.Draw()

c.SetLogy()
c.Update()
c.SaveAs("Overlay_histos/pointingCos distr overlay.png")

#B0 eta
nbin, nmin, nmax = 100, -10, 10

B0_eta_sign_h = df0.Histo1D(("B0 eta histo", "B0 eta signal distr", nbin, nmin, nmax),"eta_B0")
B0_eta_spec_h = df1.Histo1D(("B0 eta histo", "B0 eta distr", nbin, nmin, nmax),"eta_B0")
B0_eta_comb_h = df2.Histo1D(("B0 eta histo", "B0 eta comb distr", nbin, nmin, nmax),"eta_B0")

B0_eta_sign_h = B0_eta_sign_h.GetValue()
B0_eta_spec_h = B0_eta_spec_h.GetValue()
B0_eta_comb_h = B0_eta_comb_h.GetValue()

for h in (B0_eta_sign_h, B0_eta_spec_h, B0_eta_comb_h):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)

B0_eta_sign_h.SetLineColor(kGreen)
B0_eta_spec_h.SetLineColor(kBlue)
B0_eta_comb_h.SetLineColor(kRed)

c = ROOT.TCanvas()
B0_eta_spec_h.Draw("HIST")
B0_eta_sign_h.Draw("HIST SAME")
B0_eta_comb_h.Draw("HIST SAME")


leg = ROOT.TLegend(0.15, 0.75, 0.3, 0.65)
leg.AddEntry(B0_eta_sign_h, "signal", "l")
leg.AddEntry(B0_eta_spec_h, "spec bckg", "l")
leg.AddEntry(B0_eta_comb_h, "comb bckg", "l")
leg.Draw()

c.Update()
c.SaveAs("Overlay_histos/B0 eta distr overlay.png")

# FlightLength3D
nbin, nmin, nmax = 100, 0, 7

flightLength3D_sign_h = df0.Histo1D(("flightLength3D", "flightLength3D sign distr", nbin, nmin, nmax), "flightLength3D")
flightLength3D_spec_h = df1.Histo1D(("flightLength3D", "flightLength3D spec distr", nbin, nmin, nmax), "flightLength3D")
flightLength3D_comb_h = df2.Histo1D(("flightLength3D", "B0 flight length distr", nbin, nmin, nmax), "flightLength3D")

flightLength3D_sign_h = flightLength3D_sign_h.GetValue()
flightLength3D_spec_h = flightLength3D_spec_h.GetValue()
flightLength3D_comb_h = flightLength3D_comb_h.GetValue()

for h in (flightLength3D_sign_h, flightLength3D_spec_h, flightLength3D_comb_h):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)

flightLength3D_sign_h.SetLineColor(kGreen)
flightLength3D_spec_h.SetLineColor(kBlue)
flightLength3D_comb_h.SetLineColor(kRed)

c = ROOT.TCanvas()
flightLength3D_comb_h.Draw("HIST")
flightLength3D_sign_h.Draw("HIST SAME")
flightLength3D_spec_h.Draw("HIST SAME")


leg = ROOT.TLegend(0.60, 0.7, 0.45, 0.55)
leg.AddEntry(flightLength3D_sign_h, "signal", "l")
leg.AddEntry(flightLength3D_spec_h, "spec bckg", "l")
leg.AddEntry(flightLength3D_comb_h, "comb bckg", "l")
leg.Draw()

c.Update()
c.SaveAs("Overlay_histos/flightLength3D distr overlay.png")

# Angular separation tau tau
nbin, nmin, nmax = 100, 0, 12

dR_TPlusTMinus_sign_h = df0.Histo1D(("dR_TPlusTMinus", "angular seperation of TT distr", nbin, nmin, nmax), "dR_TPlusTMinus")
dR_TPlusTMinus_spec_h = df1.Histo1D(("dR_TPlusTMinus", "dR_TPlusTMinus spec distr", nbin, nmin, nmax), "dR_TPlusTMinus")
dR_TPlusTMinus_comb_h = df2.Histo1D(("dR_TPlusTMinus", "dR_TPlusTMinus comb distr", nbin, nmin, nmax), "dR_TPlusTMinus")

dR_TPlusTMinus_sign_h = dR_TPlusTMinus_sign_h.GetValue()
dR_TPlusTMinus_spec_h = dR_TPlusTMinus_spec_h.GetValue()
dR_TPlusTMinus_comb_h = dR_TPlusTMinus_comb_h.GetValue()

for h in (dR_TPlusTMinus_sign_h, dR_TPlusTMinus_spec_h, dR_TPlusTMinus_comb_h):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)

dR_TPlusTMinus_sign_h.SetLineColor(kGreen)
dR_TPlusTMinus_spec_h.SetLineColor(kBlue)
dR_TPlusTMinus_comb_h.SetLineColor(kRed)

c = ROOT.TCanvas()
dR_TPlusTMinus_sign_h.Draw("HIST")
dR_TPlusTMinus_comb_h.Draw("HIST SAME")
dR_TPlusTMinus_spec_h.Draw("HIST SAME")


leg = ROOT.TLegend(0.60, 0.7, 0.45, 0.55)
leg.AddEntry(dR_TPlusTMinus_sign_h, "signal", "l")
leg.AddEntry(dR_TPlusTMinus_spec_h, "spec bckg", "l")
leg.AddEntry(dR_TPlusTMinus_comb_h, "comb bckg", "l")
leg.Draw()

c.Update()
c.SaveAs("Overlay_histos/dR_TPlusTMinus distr overlay.png")

#Vertex Chi 2
nbin, nmin, nmax = 100, 0, 6

vertexChi2_sign_h = df0.Histo1D(("vertexChi2", "vertexChi2 distr", nbin, nmin, nmax), "vertexChi2")
vertexChi2_spec_h = df1.Histo1D(("vertexChi2", "vertexChi2 spec distr", nbin, nmin, nmax), "vertexChi2")
vertexChi2_comb_h = df2.Histo1D(("vertexChi2", "vertexChi2 comb distr", nbin, nmin, nmax), "vertexChi2")

vertexChi2_sign_h = vertexChi2_sign_h.GetValue()
vertexChi2_spec_h = vertexChi2_spec_h.GetValue()
vertexChi2_comb_h = vertexChi2_comb_h.GetValue()

for h in (vertexChi2_sign_h, vertexChi2_spec_h, vertexChi2_comb_h):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)

vertexChi2_sign_h.SetLineColor(kGreen)
vertexChi2_spec_h.SetLineColor(kBlue)
vertexChi2_comb_h.SetLineColor(kRed)

c = ROOT.TCanvas()
vertexChi2_sign_h.Draw("HIST")
vertexChi2_comb_h.Draw("HIST SAME")
vertexChi2_spec_h.Draw("HIST SAME")


leg = ROOT.TLegend(0.60, 0.7, 0.45, 0.55)
leg.AddEntry(vertexChi2_sign_h, "signal", "l")
leg.AddEntry(vertexChi2_spec_h, "spec bckg", "l")
leg.AddEntry(vertexChi2_comb_h, "comb bckg", "l")
leg.Draw()

c.Update()
c.SaveAs("Overlay_histos/vertexChi2 distr overlay.png")