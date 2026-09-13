"""Complete verification of the Proof of Theorem 3.1 (Appendix A, p.40).

Every component checked independently. Nothing assumed from the paper except
the model primitives of Sections 3.1-3.4.
"""
import numpy as np
from scipy.stats import norm

PASS = lambda ok: "PASS" if ok else "**FAIL**"
res = []

# ---------------------------------------------------------------- primitives
nu, k = 2.3, 1.7
ALPHA, SIG, BETA, KAP, RHO, GAM = 0.69, 0.30, 1.50, 0.60, 0.70, 1.9
XI = KAP/nu
P_, ETA1, ETA2, LAM = 0.44, 50.50, 25.86, 10.23

print("="*76); print("PROOF OF THEOREM 3.1 - COMPONENT-BY-COMPONENT VERIFICATION")
print("="*76)

# ---- 1. the five partial derivatives ------------------------------------
h = 1e-5
g = lambda L,I: I**nu/(k*L)
L0,I0 = 1.4,2.1; S0 = g(L0,I0)
num = dict(
 gL =(g(L0+h,I0)-g(L0-h,I0))/(2*h),
 gI =(g(L0,I0+h)-g(L0,I0-h))/(2*h),
 gLL=(g(L0+h,I0)-2*g(L0,I0)+g(L0-h,I0))/h**2,
 gII=(g(L0,I0+h)-2*g(L0,I0)+g(L0,I0-h))/h**2,
 gLI=(g(L0+h,I0+h)-g(L0+h,I0-h)-g(L0-h,I0+h)+g(L0-h,I0-h))/(4*h**2))
paper = dict(gL=-S0/L0, gI=nu*S0/I0, gLL=2*S0/L0**2,
             gII=nu*(nu-1)*S0/I0**2, gLI=+nu*S0/(L0*I0))
print("\n1. THE FIVE PARTIAL DERIVATIVES (closed forms as printed)")
for kk in num:
    ok = abs(num[kk]-paper[kk]) < 1e-4*max(1,abs(num[kk]))
    res.append(ok)
    print("   %-5s numerical %+11.6f   printed %+11.6f   %s"
          % (kk, num[kk], paper[kk], PASS(ok)))

# ---- 2. the general IFT formulas ----------------------------------------
D = lambda L,I,S: I**nu/(L*S)
Pn=[L0,I0,S0]; IXm={'L':0,'I':1,'S':2}
def d1(v):
    i=IXm[v]; a=list(Pn); b=list(Pn); a[i]+=h; b[i]-=h; return (D(*a)-D(*b))/(2*h)
def d2(v,w):
    i,j=IXm[v],IXm[w]; pp,pm,mp,mm=(list(Pn) for _ in range(4))
    pp[i]+=h;pp[j]+=h;pm[i]+=h;pm[j]-=h;mp[i]-=h;mp[j]+=h;mm[i]-=h;mm[j]-=h
    return (D(*pp)-D(*pm)-D(*mp)+D(*mm))/(4*h*h)
DL,DI,DS=d1('L'),d1('I'),d1('S')
DLI,DLS,DIS,DSS,DLL,DII=d2('L','I'),d2('L','S'),d2('I','S'),d2('S','S'),d2('L','L'),d2('I','I')
ift = dict(
  gLI=(DL*DS*DIS + DI*DS*DLS - DL*DI*DSS - DS**2*DLI)/DS**3,
  gLL=-(DS**2*DLL - 2*DL*DLS*DS + DL**2*DSS)/DS**3,
  gII=-(DS**2*DII - 2*DI*DIS*DS + DI**2*DSS)/DS**3)
print("\n2. THE GENERAL IFT FORMULAS (as printed, in terms of D's partials)")
for kk in ift:
    ok = abs(ift[kk]-num[kk]) < 1e-3*max(1,abs(num[kk]))
    res.append(ok); print("   %-5s IFT formula %+11.6f   direct %+11.6f   %s"
                          % (kk, ift[kk], num[kk], PASS(ok)))

# ---- 3. the dL/L line, continuous part ----------------------------------
rng = np.random.default_rng(1); n, dt = 4_000_000, 1/2520.
Psi0 = 0.35
dZ = rng.standard_normal(n)*np.sqrt(dt)
dPsi = -ALPHA*Psi0*dt + SIG*BETA*dZ
dLoverL = np.exp(-(Psi0+dPsi))/np.exp(-Psi0) - 1.0
pred_drift = (ALPHA*Psi0 + 0.5*(SIG*BETA)**2)*dt
ok = abs(dLoverL.mean() - pred_drift) < 4*dLoverL.std()/np.sqrt(n)
res.append(ok)
print("\n3. dL/L CONTINUOUS PART:  [alpha*Psi + (sigma*beta)^2/2]dt - sigma*beta dZ")
print("   simulated drift %+.8f   printed %+.8f   %s"
      % (dLoverL.mean(), pred_drift, PASS(ok)))

# ---- 4. the dL/L jump term ----------------------------------------------
Y = 0.05
print("\n4. dL/L JUMP TERM")
print("   Psi jumps by gamma*Y (corrected 3.2), so L -> L*exp(-gamma*Y):")
print("     correct   dL/L = exp(-gamma Y) - 1 = %+.8f" % (np.exp(-GAM*Y)-1))
print("     printed  -d(e^{gamma Y} - 1)      = %+.8f" % (-(np.exp(GAM*Y)-1)))
ok = abs((np.exp(-GAM*Y)-1) - (-(np.exp(GAM*Y)-1))) < 1e-10
res.append(ok); print("   %s  (printed is the first-order approximation only)" % PASS(ok))

# ---- 5. jump in dS/S -----------------------------------------------------
print("\n5. JUMP IN dS/S, from S = I^nu/(kL) and L -> L exp(-gamma Y)")
lhs = np.exp(GAM*Y)-1
rhs = 1/np.exp(-GAM*Y) - 1
ok = abs(lhs-rhs) < 1e-12; res.append(ok)
print("   Theorem 3.1 states e^{gamma Y} - 1 = %+.8f ; derived %+.8f   %s"
      % (lhs, rhs, PASS(ok)))

# ---- 6. the MGF line -----------------------------------------------------
rng2 = np.random.default_rng(2); m = 8_000_000
up = rng2.random(m) < P_
Ys = np.where(up, rng2.exponential(1/ETA1,m), -rng2.exponential(1/ETA2,m))
mc = np.exp(GAM*Ys).mean()
closed = P_*ETA1/(ETA1-GAM) + (1-P_)*ETA2/(ETA2+GAM)
ok = abs(mc-closed) < 5e-5; res.append(ok)
print("\n6. E^P[e^{gamma_i Y}] = p eta1/(eta1-gamma) + (1-p) eta2/(eta2+gamma)")
print("   Monte Carlo %.8f   closed form %.8f   %s" % (mc, closed, PASS(ok)))

# ---- 7. the assembled drift, vs Monte Carlo -----------------------------
print("\n7. THE ASSEMBLED DRIFT m_i  (alpha=0, no jumps, so E[S_T]/S_0 = e^{mT})")
T=1.0; npaths=4_000_000
rng3=np.random.default_rng(3); tot=0.0; B=200_000
for _ in range(npaths//B):
    z=rng3.standard_normal(B); w=RHO*z+np.sqrt(1-RHO**2)*rng3.standard_normal(B)
    logI=nu*(-0.5*XI**2*T + XI*np.sqrt(T)*w)     # nu * log I
    psi =SIG*BETA*np.sqrt(T)*z                    # Psi
    tot+=np.exp(logI+psi).sum()
mc_m=np.log(tot/(npaths//B*B))/T
m_plus  = nu*0.0 + 0.5*nu*(nu-1)*XI**2 + 0.5*(SIG*BETA)**2 + SIG*BETA*KAP*RHO
m_minus = nu*0.0 + 0.5*nu*(nu-1)*XI**2 + 0.5*(SIG*BETA)**2 - SIG*BETA*KAP*RHO
ok = abs(mc_m-m_plus) < 3e-3; res.append(ok)
print("   Monte Carlo          %+.6f" % mc_m)
print("   derived  (+ cross)   %+.6f   %s" % (m_plus, PASS(ok)))
print("   printed  (- cross)   %+.6f   %s" % (m_minus, PASS(abs(mc_m-m_minus)<3e-3)))

# ---- 8. the quadratic variations (NEVER DISPLAYED IN THE PROOF) ---------
print("\n8. THE QUADRATIC VARIATIONS  (the proof shows none of these)")
rng4 = np.random.default_rng(4); n4, dt4 = 6_000_000, 1/2520.
z = rng4.standard_normal(n4)*np.sqrt(dt4)
w = RHO*z + np.sqrt(1-RHO**2)*rng4.standard_normal(n4)*np.sqrt(dt4)
dLoL = -SIG*BETA*z            # martingale part of dL/L
dIoI =  XI*w                  # martingale part of dI/I
qv = dict(
  LL =(dLoL*dLoL).sum()/ (n4*dt4),
  II =(dIoI*dIoI).sum()/ (n4*dt4),
  LI =(dLoL*dIoI).sum()/ (n4*dt4))
closed = dict(LL=(SIG*BETA)**2, II=XI**2, LI=-SIG*BETA*XI*RHO)
for kk in qv:
    ok = abs(qv[kk]-closed[kk]) < 5e-3*max(1,abs(closed[kk]))
    res.append(ok)
    print("   d<%s>/(%s dt)  simulated %+10.6f   closed %+10.6f   %s"
          % (kk, {"LL":"L^2","II":"I^2","LI":"L I "}[kk], qv[kk], closed[kk], PASS(ok)))
print("   -> d<L,I> = -L I sigma beta xi rho dt is THE ONLY PLACE rho enters.")

# ---- 9. Ito assembly, term by term --------------------------------------
print("\n9. ITO ASSEMBLY OF dS/S   (each Ito term divided by S)")
L1,I1 = L0,I0; S1 = g(L1,I1)
terms = {
 "g_L dL            ": paper['gL']*L1*(ALPHA*Psi0+0.5*(SIG*BETA)**2)/S1,
 "g_I dI            ": paper['gI']*I1*0.0/S1,
 "(1/2) g_LL d<L>   ": 0.5*paper['gLL']*L1**2*closed['LL']/S1,
 "(1/2) g_II d<I>   ": 0.5*paper['gII']*I1**2*closed['II']/S1,
}
cross_correct = (-nu*S1/(L1*I1))*L1*I1*closed['LI']/S1
cross_printed = (+nu*S1/(L1*I1))*L1*I1*closed['LI']/S1
for kk,v in terms.items(): print("   %s %+12.8f" % (kk, v))
print("   g_LI d<L,I>        %+12.8f   (corrected g_LI = -nu S/(L I))" % cross_correct)
print("   g_LI d<L,I>        %+12.8f   (printed   g_LI = +nu S/(L I))" % cross_printed)
ok = abs(cross_correct - SIG*BETA*KAP*RHO) < 1e-9; res.append(ok)
print("   corrected cross == +sigma beta kappa rho = %+.8f   %s"
      % (SIG*BETA*KAP*RHO, PASS(ok)))
ok2 = abs(cross_printed - SIG*BETA*KAP*RHO) < 1e-9; res.append(ok2)
print("   printed   cross == +sigma beta kappa rho ?                  %s" % PASS(ok2))

print("\n" + "="*76)
print("%d of %d components PASS" % (sum(res), len(res)))
print("="*76)
