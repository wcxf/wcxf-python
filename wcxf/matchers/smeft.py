import numpy as np
from math import sqrt, pi
import smeftrunner
import wcxf
from wcxf.parameters import p

# Based on arXiv:1709.04486

# CONSTANTS

MW = p['m_W']
MZ = p['m_Z']
GF = p['GF']
alpha_e = p['alpha_e']
Nc = 3

# AUXILIARY FUNCTIONS

# Eq. (6.4)
def vT(C):
    GFx = GF - sqrt(2)/4 * ( -C['ll'][1,0,0,1] - C['ll'][0,1,1,0] + 2*C['phil3'][1,1] + 2*C['phil3'][0,0] )
    return sqrt(1/sqrt(2)/abs(GFx))

# Eq. (2.22)
def eps(C):
    return C["phiWB"] * vT(C)**2

# Eq. (2.24)
def g2b(C):
    return 2*MW/vT(C)

def g1b(C):
    eb = sqrt(4*pi*alpha_e)
    return eb*g2b(C)/sqrt(g2b(C)**2-eb**2) + eb**2*g2b(C)/(g2b(C)**2-eb**2) * eps(C)

# Eq. (2.23)
def sb(C):
    return g1b(C)/sqrt(g1b(C)**2+g2b(C)**2) * (1 + eps(C)/2. * g2b(C)/g1b(C) * ((g2b(C)**2-g1b(C)**2)/(g1b(C)**2+g2b(C)**2)))
def cb(C):
    return g2b(C)/sqrt(g1b(C)**2+g2b(C)**2) * (1 - eps(C)/2. * g1b(C)/g2b(C) * ((g2b(C)**2-g1b(C)**2)/(g1b(C)**2+g2b(C)**2)))

# Eq. (2.26)
def eb(C):
    return g2b(C)*sb(C) - 1/2.*cb(C)*g2b(C)* vT(C)**2*C["phiWB"]
def gzb(C):
    return eb(C)/(sb(C)*cb(C)) * (1 + (g1b(C)**2+g2b(C)**2)/(2*g1b(C)*g2b(C))*vT(C)**2*C["phiWB"])

# Eq. (2.30)
def wl(C):
    return np.eye(3)+vT(C)**2*C["phil3"]
def wq(C):
    return np.eye(3)+vT(C)**2*C["phiq3"]
def wr(C):
    return 1/2.*vT(C)**2*C["phiud"]
def znu(C):
    return np.eye(3)*1/2.-1/2.*vT(C)**2*C["phil1"]+1/2.*vT(C)**2*C["phil3"]
def zel(C):
    return np.eye(3)*(-1/2.+sb(C)**2)-1/2.*vT(C)**2*C["phil1"]-1/2.*vT(C)**2*C["phil3"]
def zer(C):
    return np.eye(3)*sb(C)**2-1/2.*vT(C)**2*C["phie"]
def zul(C):
    return np.eye(3)*(1/2.-2./3*sb(C)**2)-1/2.*vT(C)**2*C["phiq1"]+1/2.*vT(C)**2*C["phiq3"]
def zur(C):
    return np.eye(3)*(-2./3)*sb(C)**2-1/2.*vT(C)**2*C["phiu"]
def zdl(C):
    return np.eye(3)*(-1/2.+1/3.*sb(C)**2)-1/2.*vT(C)**2*C["phiq1"]-1/2.*vT(C)**2*C["phiq3"]
def zdr(C):
    return np.eye(3)*(1/3.)*sb(C)**2-1/2.*vT(C)**2*C["phid"]

# MATCHING CONDITIONS

# initialize empty dict that will become a dict of functions
C = {}

# Table 9
C["nu"] = lambda C: 1/2.*C["llphiphi"]* vT(C)**2

# Table 10
C["nugamma"] = lambda C: np.zeros((3,3))

# Table 11
C["egamma"] = lambda C: 1/sqrt(2) * (-C["eW"] * sb(C) + C["eB"] * cb(C)) * vT(C)

C["ugamma"] = lambda C: 1/sqrt(2) * (C["uW"] * sb(C) + C["uB"] * cb(C)) * vT(C)
C["dgamma"] = lambda C: 1/sqrt(2) * (-C["dW"] * sb(C) + C["dB"] * cb(C)) * vT(C)
C["uG"] = lambda C: 1/sqrt(2) * C["uG"] * vT(C)
C["dG"] = lambda C: 1/sqrt(2) * C["dG"] * vT(C)

#Table 12
C["G"] = lambda C: C["G"]
C["Gtilde"] = lambda C: C["Gtilde"]

# Table 13
C["VnunuLL"] = lambda C: C["ll"]-gzb(C)**2/(4*MZ**2)*np.einsum('pr,st',znu(C),znu(C))-gzb(C)**2/(4*MZ**2)*np.einsum('pt,sr',znu(C),znu(C))
C["VeeLL"] = lambda C: C["ll"]-gzb(C)**2/(4*MZ**2)*np.einsum('pr,st',zel(C),zel(C))-gzb(C)**2/(4*MZ**2)*np.einsum('pt,sr',zel(C),zel(C))
C["VnueLL"] = lambda C: C["ll"]+np.einsum('stpr',C["ll"])-g2b(C)**2/(2*MW**2)*np.einsum('pt,rs',wl(C),wl(C).conjugate())-gzb(C)**2/(MZ**2)*np.einsum('pr,st',znu(C),zel(C))

C["VnuuLL"] = lambda C: C["lq1"]+C["lq3"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',znu(C),zul(C))
C["VnudLL"] = lambda C: C["lq1"]-C["lq3"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',znu(C),zdl(C))
C["VeuLL"] = lambda C: C["lq1"]-C["lq3"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zel(C),zul(C))
C["VedLL"] = lambda C: C["lq1"]+C["lq3"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zel(C),zdl(C))
C["VnueduLL"] = lambda C: 2*C["lq3"]-g2b(C)**2/(2*MW**2)*np.einsum('pr,ts',wl(C),wq(C).conjugate())
# + h.c.

C["VuuLL"] = lambda C: C["qq1"]+C["qq3"]-gzb(C)**2/(2*MZ**2)*np.einsum('pr,st',zul(C),zul(C))
C["VddLL"] = lambda C: C["qq1"]+C["qq3"]-gzb(C)**2/(2*MZ**2)*np.einsum('pr,st',zdl(C),zdl(C))
C["V1udLL"] = lambda C: C["qq1"]+np.einsum('stpr',C["qq1"])-C["qq3"]-np.einsum('stpr',C["qq3"])+2/Nc*np.einsum('ptsr',C["qq3"])+2/Nc*np.einsum('srpt',C["qq3"])-g2b(C)**2/(2*MW**2)*np.einsum('pt,rs',wq(C),wq(C).conjugate())/Nc-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zul(C),zdl(C))
C["V8udLL"] = lambda C: 4*np.einsum('ptsr',C["qq3"])+4*np.einsum('srpt',C["qq3"])-g2b(C)**2/(MW**2)*np.einsum('pt,rs',wq(C),wq(C).conjugate())


# Table 14
C["VeeRR"] = lambda C: C["ee"]-gzb(C)**2/(4*MZ**2)*np.einsum('pr,st',zer(C),zer(C))-gzb(C)**2/(4*MZ**2)*np.einsum('pt,sr',zer(C),zer(C))

C["VeuRR"] = lambda C: C["eu"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zer(C),zur(C))
C["VedRR"] = lambda C: C["ed"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zer(C),zdr(C))

C["VuuRR"] = lambda C: C["uu"]-gzb(C)**2/(2*MZ**2)*np.einsum('pr,st',zur(C),zur(C))
C["VddRR"] = lambda C: C["dd"]-gzb(C)**2/(2*MZ**2)*np.einsum('pr,st',zdr(C),zdr(C))
C["V1udRR"] = lambda C: C["ud1"]-g2b(C)**2/(2*MW**2)*np.einsum('pt,rs',wr(C),wr(C).conjugate())/Nc-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zur(C),zdr(C))
C["V8udRR"] = lambda C: C["ud8"]-g2b(C)**2/(MW**2)*np.einsum('pt,rs',wr(C),wr(C).conjugate())


# Table 15
C["VnueLR"] = lambda C: C["le"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',znu(C),zer(C))
C["VeeLR"] = lambda C: C["le"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zel(C),zer(C))

C["VnuuLR"] = lambda C: C["lu"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',znu(C),zur(C))
C["VnudLR"] = lambda C: C["ld"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',znu(C),zdr(C))
C["VeuLR"] = lambda C: C["lu"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zel(C),zur(C))
C["VedLR"] = lambda C: C["ld"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zel(C),zdr(C))
C["VueLR"] = lambda C: C["qe"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zul(C),zer(C))
C["VdeLR"] = lambda C: C["qe"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zdl(C),zer(C))
C["VnueduLR"] = lambda C: -g2b(C)**2/(2*MW**2)*np.einsum('pr,ts',wl(C),wr(C).conjugate())
#+ h.c.

C["V1uuLR"] = lambda C: C["qu1"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zul(C),zur(C))
C["V8uuLR"] = lambda C: C["qu8"]
C["V1udLR"] = lambda C: C["qd1"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zul(C),zdr(C))
C["V8udLR"] = lambda C: C["qd8"]
C["V1duLR"] = lambda C: C["qu1"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zdl(C),zur(C))
C["V8duLR"] = lambda C: C["qu8"]
C["V1ddLR"] = lambda C: C["qd1"]-gzb(C)**2/(MZ**2)*np.einsum('pr,st',zdl(C),zdr(C))
C["V8ddLR"] = lambda C: C["qd8"]
C["V1udduLR"] = lambda C: -g2b(C)**2/(2*MW**2)*np.einsum('pr,ts',wq(C),wr(C).conjugate())
C["V8udduLR"] = lambda C: np.zeros((3,3,3,3))


# Table 16
C["SeuRL"] = lambda C: np.zeros((3,3,3,3))
C["SedRL"] = lambda C: C["ledq"]
C["SnueduRL"] = lambda C: C["ledq"]


# Table 17
C["SeeRR"] = lambda C: np.zeros((3,3,3,3))

C["SeuRR"] = lambda C: -C["lequ1"]
C["TeuRR"] = lambda C: -C["lequ3"]
C["SedRR"] = lambda C: np.zeros((3,3,3,3))
C["TedRR"] = lambda C: np.zeros((3,3,3,3))
C["SnueduRR"] = lambda C: C["lequ1"]
C["TnueduRR"] = lambda C: C["lequ3"]

C["S1uuRR"] = lambda C: np.zeros((3,3,3,3))
C["S8uuRR"] = lambda C: np.zeros((3,3,3,3))
C["S1udRR"] = lambda C: C["quqd1"]
C["S8udRR"] = lambda C: C["quqd8"]
C["S1ddRR"] = lambda C: np.zeros((3,3,3,3))
C["S8ddRR"] = lambda C: np.zeros((3,3,3,3))
C["S1udduRR"] = lambda C: -np.einsum('stpr',C["quqd1"])
C["S8udduRR"] = lambda C: -np.einsum('stpr',C["quqd8"])

# Table 18
C["SnunuLL"] = lambda C: np.zeros((3,3,3,3))

# Table 19
C["SnueLL"] = lambda C: np.zeros((3,3,3,3))
C["TnueLL"] = lambda C: np.zeros((3,3,3,3))
C["SnueLR"] = lambda C: np.zeros((3,3,3,3))

C["SnuuLL"] = lambda C: np.zeros((3,3,3,3))
C["TnuuLL"] = lambda C: np.zeros((3,3,3,3))
C["SnuuLR"] = lambda C: np.zeros((3,3,3,3))
C["SnudLL"] = lambda C: np.zeros((3,3,3,3))
C["TnudLL"] = lambda C: np.zeros((3,3,3,3))
C["SnudLR"] = lambda C: np.zeros((3,3,3,3))
C["SnueduLL"] = lambda C: np.zeros((3,3,3,3))
C["TnueduLL"] = lambda C: np.zeros((3,3,3,3))
C["SnueduLR"] = lambda C: np.zeros((3,3,3,3))
C["VnueduRL"] = lambda C: np.zeros((3,3,3,3))
C["VnueduRR"] = lambda C: np.zeros((3,3,3,3))

# Table 20
C["SuddLL"] = lambda C: -C["qqql"]-np.einsum('rpst',C["qqql"])
C["SduuLL"] = lambda C: -C["qqql"]-np.einsum('rpst',C["qqql"])
C["SuudLR"] = lambda C: np.zeros((3,3,3,3))
C["SduuLR"] = lambda C: -C["qque"]-np.einsum('rpst',C["qque"])
C["SuudRL"] = lambda C: np.zeros((3,3,3,3))
C["SduuRL"] = lambda C: C["duql"]
C["SdudRL"] = lambda C: -C["duql"]
C["SdduRL"] = lambda C: np.zeros((3,3,3,3))
C["SduuRR"] = lambda C: C["duue"]

# Table 21
C["SdddLL"] = lambda C: np.zeros((3,3,3,3))
C["SuddLR"] = lambda C: np.zeros((3,3,3,3))
C["SdduLR"] = lambda C: np.zeros((3,3,3,3))
C["SdddLR"] = lambda C: np.zeros((3,3,3,3))
C["SdddRL"] = lambda C: np.zeros((3,3,3,3))
C["SuddRR"] = lambda C: np.zeros((3,3,3,3))
C["SdddRR"] = lambda C: np.zeros((3,3,3,3))

def match_all_array(C_SMEFT):
    return {k: f(C_SMEFT) for k, f in C.items()}

def match_all(d_SMEFT):
    C = smeftrunner.io.wcxf2arrays(d_SMEFT)
    C = smeftrunner.definitions.symmetrize(C)
    C['vT'] = 246.22
    C_WET = match_all_array(C)
    d_WET = smeftrunner.io.arrays2wcxf(C_WET)
    basis = wcxf.Basis['WET', 'JMS']
    d_WET = {k: v for k, v in d_WET.items() if k in basis.all_wcs}
    return d_WET
