from math import pi, sqrt
import numpy as np
from wcxf.parameters import p


def _scalar2array(d):
    """Convert a dictionary with scalar elements and string indices '_1234'
    to a dictionary of arrays. Unspecified entries are np.nan."""
    da = {}
    for k, v in d.items():
        if '_' not in k:
            da[k] = v
        else:
            name = ''.join(k.split('_')[:-1])
            ind = k.split('_')[-1]
            dim = len(ind)
            if name not in da:
                shape = tuple(3 for i in range(dim))
                da[name] = np.empty(shape, dtype=complex)
                da[name][:] = np.nan
            da[name][tuple(int(i)-1 for i in ind)] = v
    return da

def _symm_herm(C):
    """To get rid of NaNs produced by _scalar2array, symmetrize operators
    where C_ijkl = C_jilk*"""
    nans = np.isnan(C)
    C[nans] = np.einsum('jilk', C)[nans].conj()
    return C

def _symm_current(C):
    """To get rid of NaNs produced by _scalar2array, symmetrize operators
    where C_ijkl = C_klij"""
    nans = np.isnan(C)
    C[nans] = np.einsum('klij', C)[nans]
    return C


# CONSTANTS

Nc = 3.
Qu = 2/3.
Qd = -1/3.
alpha_e = p['alpha_e']
alpha_s = p['alpha_s']
e = sqrt(4*pi*alpha_e)
gs = sqrt(4*pi*alpha_s)
mb = 4.2
ms = 0.095

# WET with b,c,s,d,u

## Class I ##

# sbsb


def SUSYsbsb (C):
    return {
'1sbsb' : C["VddLL"][1, 2, 1, 2],
'2sbsb' : C["S1ddRR"][2, 1, 2, 1].conjugate() - C["S8ddRR"][2, 1, 2, 1].conjugate()/(2*Nc),
'3sbsb' : C["S8ddRR"][2, 1, 2, 1].conjugate()/2,
'4sbsb' : -C["V8ddLR"][1, 2, 1, 2],
'5sbsb' : -2*C["V1ddLR"][1, 2, 1, 2] + C["V8ddLR"][1, 2, 1,2]/Nc,
'1psbsb' : C["VddRR"][1, 2, 1, 2],
'2psbsb' : C["S1ddRR"][1, 2, 1, 2] - C["S8ddRR"][1, 2, 1,2]/(2*Nc),
'3psbsb' : C["S8ddRR"][1, 2, 1, 2]/2}


def Flaviosbsb (SUSYsbsb):
    return {
'CVLL_bsbs' : SUSYsbsb["1sbsb"],
'CSLL_bsbs' : SUSYsbsb["2sbsb"]+1/2.*SUSYsbsb["3sbsb"],
'CTLL_bsbs' : -1/8.*SUSYsbsb["3sbsb"],
'CVLR_bsbs' : -1/2.*SUSYsbsb["5sbsb"],
'CVRR_bsbs' : SUSYsbsb["1psbsb"],
'CSRR_bsbs' : SUSYsbsb["2psbsb"]+1/2.*SUSYsbsb["3psbsb"],
'CTRR_bsbs' : -1/8.*SUSYsbsb["3psbsb"],
'CSLR_bsbs' : SUSYsbsb["4sbsb"]
}

# dbdb


def SUSYdbdb (C):
    return {
"1dbdb" : C['VddLL'][0,2,0,2],
"2dbdb" : C['S1ddRR'][2,0,2,0].conj()-1/(2*Nc)*C['S8ddRR'][2,0,2,0].conj() ,
"3dbdb" :  1/2.*C['S8ddRR'][2,0,2,0].conj(),
"4dbdb" : -C['V8ddLR'][0,2,0,2],
"5dbdb" : -2*C['V1ddLR'][0,2,0,2]+1/Nc*C['V8ddLR'][0,2,0,2],
"1pdbdb" : C['VddRR'][0,2,0,2],
"2pdbdb" : C['S1ddRR'][0,2,0,2]-1/(2*Nc)*C['S8ddRR'][0,2,0,2],
"3pdbdb" :  1/2*C['S8ddRR'][0,2,0,2]
}


def Flaviodbdb (SUSYdbdb):
    return {
'CVLL_bdbd' : SUSYdbdb["1dbdb"],
'CSLL_bdbd' : SUSYdbdb["2dbdb"]+1/2.*SUSYdbdb["3dbdb"],
'CTLL_bdbd' : -1/8.*SUSYdbdb["3dbdb"],
'CVLR_bdbd' : -1/2.*SUSYdbdb["5dbdb"],
'CVRR_bdbd' : SUSYdbdb["1pdbdb"],
'CSRR_bdbd' : SUSYdbdb["2pdbdb"]+1/2.*SUSYdbdb["3pdbdb"],
'CTRR_bdbd' : -1/8.*SUSYdbdb["3pdbdb"],
'CSLR_bdbd' : SUSYdbdb["4dbdb"]
}

## Class II ##

# ublnu

def Bernublnu (C):
    return {
"1ubllp": C["VnueduLL"][:,:,2,0].conjugate(),
"5ubllp" : C["SnueduRL"][:,:,2,0].conjugate(),
"1publlp" : C["VnueduLR"][:,:,2,0].conjugate(),
"5publlp" : C["SnueduRR"][:,:,2,0].conjugate(),
"7publlp" : C["TnueduRR"][:,:,2,0].conjugate(),
}

def ACFGublnu (Bernublnu):
    return {
'CV_bulnu': Bernublnu['1ubllp'],
'CVp_bulnu': Bernublnu['1publlp'],
'CSp_bulnu': Bernublnu['5ubllp'],
'CS_bulnu': Bernublnu['5publlp'],
'CT_bulnu': Bernublnu['7publlp']
}

def Flavioublnu (Bernublnu):
    return {
'CV_bulnu': Bernublnu['1ubllp'],
'CVp_bulnu': Bernublnu['1publlp'],
'CS_bulnu': Bernublnu['5ubllp']/mb,
'CSp_bulnu': Bernublnu['5publlp']/mb,
'CT_bulnu': Bernublnu['7publlp']
}

# cblnu

def Berncblnu (C):
    return {
"1cbllp": C["VnueduLL"][:,:,2,1].conjugate(),
"5cbllp" : C["SnueduRL"][:,:,2,1].conjugate(),
"1pcbllp" : C["VnueduLR"][:,:,2,1].conjugate(),
"5pcbllp" : C["SnueduRR"][:,:,2,1].conjugate(),
"7pcbllp" : C["TnueduRR"][:,:,2,1].conjugate(),
}

def ACFGcblnu (Berncblnu):
    return {
'CV_bclnu': Berncblnu['1cbllp'],
'CVp_bclnu': Berncblnu['1pcbllp'],
'CSp_bclnu': Berncblnu['5cbllp'],
'CS_bclnu': Berncblnu['5pcbllp'],
'CT_bclnu': Berncblnu['7pcbllp']}


def Flaviocblnu (Berncblnu):
    return {
'CV_bclnu': Berncblnu['1cbllp'],
'CVp_bclnu': Berncblnu['1pcbllp'],
'CS_bclnu': Berncblnu['5cbllp']/mb,
'CSp_bclnu': Berncblnu['5pcbllp']/mb,
'CT_bclnu': Berncblnu['7pcbllp']}


## Class III ##

# Fierz basis with sbqq,

# sbuc

def Fsbuc (C):
    return {'Fsbuc1' : C["V1udLL"][0, 1, 1, 2] - C["V8udLL"][0, 1, 1, 2]/(2*Nc),
 'Fsbuc2' : C["V8udLL"][0, 1, 1, 2]/2,
 'Fsbuc3' : C["V1duLR"][1, 2, 0, 1] - C["V8duLR"][1, 2, 0, 1]/(2*Nc),
 'Fsbuc4' : C["V8duLR"][1, 2, 0, 1]/2,
 'Fsbuc5' : C["S1udRR"][0, 1, 1, 2] - C["S8udduRR"][0, 2, 1, 1]/4 - C["S8udRR"][0, 1, 1, 2]/(2*Nc),
 'Fsbuc6' : -C["S1udduRR"][0, 2, 1, 1]/2 + C["S8udduRR"][0, 2, 1, 1]/(4*Nc) + C["S8udRR"][0, 1, 1, 2]/2,
 'Fsbuc7' : -C["V8udduLR"][1, 1, 2, 0].conjugate(),
 'Fsbuc8' : -2*C["V1udduLR"][1, 1, 2, 0].conjugate() + C["V8udduLR"][1, 1, 2, 0].conjugate()/Nc,
 'Fsbuc9' : -C["S8udduRR"][0, 2, 1, 1]/16,
 'Fsbuc10' : -C["S1udduRR"][0, 2, 1, 1]/8 + C["S8udduRR"][0, 2, 1, 1]/(16*Nc)}

def Fpsbuc (C):
    return {'Fsbuc1p' : C["V1udRR"][0, 1, 1, 2] - C["V8udRR"][0, 1, 1, 2]/(2*Nc),
 'Fsbuc2p' : C["V8udRR"][0, 1, 1, 2]/2,
 'Fsbuc3p' : C["V1udLR"][0, 1, 1, 2] - C["V8udLR"][0, 1, 1, 2]/(2*Nc),
 'Fsbuc4p' : C["V8udLR"][0, 1, 1, 2]/2,
 'Fsbuc5p' : C["S1udRR"][1, 0, 2, 1].conjugate() - C["S8udduRR"][1, 1, 2, 0].conjugate()/4 - C["S8udRR"][1, 0, 2, 1].conjugate()/(2*Nc),
 'Fsbuc6p' : -C["S1udduRR"][1, 1, 2, 0].conjugate()/2 + C["S8udduRR"][1, 1, 2, 0].conjugate()/(4*Nc) + C["S8udRR"][1, 0, 2, 1].conjugate()/2,
 'Fsbuc7p' : -C["V8udduLR"][0, 2, 1, 1],
 'Fsbuc8p' : -2*C["V1udduLR"][0, 2, 1, 1] + C["V8udduLR"][0, 2, 1, 1]/Nc,
 'Fsbuc9p' : -C["S8udduRR"][1, 1, 2, 0].conjugate()/16,
 'Fsbuc10p' : -C["S1udduRR"][1, 1, 2, 0].conjugate()/8 + C["S8udduRR"][1, 1, 2, 0].conjugate()/(16*Nc)}

# sbcu

def Fsbcu (C):
    return {'Fsbcu1' : C["V1udLL"][0, 1, 2, 1].conjugate() - C["V8udLL"][0, 1, 2, 1].conjugate()/(2*Nc),
 'Fsbcu2' : C["V8udLL"][0, 1, 2, 1].conjugate()/2,
 'Fsbcu3' : C["V1duLR"][1, 2, 1, 0] - C["V8duLR"][1, 2, 1, 0]/(2*Nc),
 'Fsbcu4' : C["V8duLR"][1, 2, 1, 0]/2,
 'Fsbcu5' : C["S1udRR"][1, 0, 1, 2] - C["S8udduRR"][1, 2, 1, 0]/4 - C["S8udRR"][1, 0, 1, 2]/(2*Nc),
 'Fsbcu6' : -C["S1udduRR"][1, 2, 1, 0]/2 + C["S8udduRR"][1, 2, 1, 0]/(4*Nc) + C["S8udRR"][1, 0, 1, 2]/2,
 'Fsbcu7' : -C["V8udduLR"][0, 1, 2, 1].conjugate(),
 'Fsbcu8' : -2*C["V1udduLR"][0, 1, 2, 1].conjugate() + C["V8udduLR"][0, 1, 2, 1].conjugate()/Nc,
 'Fsbcu9' : -C["S8udduRR"][1, 2, 1, 0]/16,
 'Fsbcu10' : -C["S1udduRR"][1, 2, 1, 0]/8 + C["S8udduRR"][1, 2, 1, 0]/(16*Nc)}


def Fpsbcu (C):
    return {'Fsbcu1p' : C["V1udRR"][0, 1, 2, 1].conjugate() - C["V8udRR"][0, 1, 2, 1].conjugate()/(2*Nc),
 'Fsbcu2p' : C["V8udRR"][0, 1, 2, 1].conjugate()/2,
 'Fsbcu3p' : C["V1udLR"][0, 1, 2, 1].conjugate() - C["V8udLR"][0, 1, 2, 1].conjugate()/(2*Nc),
 'Fsbcu4p' : C["V8udLR"][0, 1, 2, 1].conjugate()/2,
 'Fsbcu5p' : C["S1udRR"][0, 1, 2, 1].conjugate() - C["S8udduRR"][0, 1, 2, 1].conjugate()/4 - C["S8udRR"][0, 1, 2, 1].conjugate()/(2*Nc),
 'Fsbcu6p' : -C["S1udduRR"][0, 1, 2, 1].conjugate()/2 + C["S8udduRR"][0, 1, 2, 1].conjugate()/(4*Nc) + C["S8udRR"][0, 1, 2, 1].conjugate()/2,
 'Fsbcu7p' : -C["V8udduLR"][1, 2, 1, 0],
 'Fsbcu8p' : -2*C["V1udduLR"][1, 2, 1, 0] + C["V8udduLR"][1, 2, 1, 0]/Nc,
 'Fsbcu9p' : -C["S8udduRR"][0, 1, 2, 1].conjugate()/16,
 'Fsbcu10p' : -C["S1udduRR"][0, 1, 2, 1].conjugate()/8 + C["S8udduRR"][0, 1, 2, 1].conjugate()/(16*Nc)}



# Fierz basis

# dbuc

def Fdbuc (C):
    return {'Fdbuc1' : C["V1udLL"][0, 1, 0, 2] - C["V8udLL"][0, 1, 0, 2]/(2*Nc),
 'Fdbuc2' : C["V8udLL"][0, 1, 0, 2]/2,
 'Fdbuc3' : C["V1duLR"][0, 2, 0, 1] - C["V8duLR"][0, 2, 0, 1]/(2*Nc),
 'Fdbuc4' : C["V8duLR"][0, 2, 0, 1]/2,
 'Fdbuc5' : C["S1udRR"][0, 1, 0, 2] - C["S8udduRR"][0, 2, 0, 1]/4 - C["S8udRR"][0, 1, 0, 2]/(2*Nc),
 'Fdbuc6' : -C["S1udduRR"][0, 2, 0, 1]/2 + C["S8udduRR"][0, 2, 0, 1]/(4*Nc) + C["S8udRR"][0, 1, 0, 2]/2,
 'Fdbuc7' : -C["V8udduLR"][1, 0, 2, 0].conjugate(),
 'Fdbuc8' : -2*C["V1udduLR"][1, 0, 2, 0].conjugate() + C["V8udduLR"][1, 0, 2, 0].conjugate()/Nc,
 'Fdbuc9' : -C["S8udduRR"][0, 2, 0, 1]/16,
 'Fdbuc10' : -C["S1udduRR"][0, 2, 0, 1]/8 + C["S8udduRR"][0, 2, 0, 1]/(16*Nc)}


def Fpdbuc (C):
    return {'Fdbuc1p' : C["V1udRR"][0, 1, 0, 2] - C["V8udRR"][0, 1, 0, 2]/(2*Nc),
 'Fdbuc2p' : C["V8udRR"][0, 1, 0, 2]/2,
 'Fdbuc3p' : C["V1udLR"][0, 1, 0, 2] - C["V8udLR"][0, 1, 0, 2]/(2*Nc),
 'Fdbuc4p' : C["V8udLR"][0, 1, 0, 2]/2,
 'Fdbuc5p' : C["S1udRR"][1, 0, 2, 0].conjugate() - C["S8udduRR"][1, 0, 2, 0].conjugate()/4 - C["S8udRR"][1, 0, 2, 0].conjugate()/(2*Nc),
 'Fdbuc6p' : -C["S1udduRR"][1, 0, 2, 0].conjugate()/2 + C["S8udduRR"][1, 0, 2, 0].conjugate()/(4*Nc) + C["S8udRR"][1, 0, 2, 0].conjugate()/2,
 'Fdbuc7p' : -C["V8udduLR"][0, 2, 0, 1],
 'Fdbuc8p' : -2*C["V1udduLR"][0, 2, 0, 1] + C["V8udduLR"][0, 2, 0, 1]/Nc,
 'Fdbuc9p' : -C["S8udduRR"][1, 0, 2, 0].conjugate()/16,
 'Fdbuc10p' : -C["S1udduRR"][1, 0, 2, 0].conjugate()/8 + C["S8udduRR"][1, 0, 2, 0].conjugate()/(16*Nc)}

# dbcu

def Fdbcu (C):
    return {'Fdbcu1' : C["V1udLL"][0, 1, 2, 0].conjugate() - C["V8udLL"][0, 1, 2, 0].conjugate()/(2*Nc),
 'Fdbcu2' : C["V8udLL"][0, 1, 2, 0].conjugate()/2,
 'Fdbcu3' : C["V1duLR"][0, 2, 1, 0] - C["V8duLR"][0, 2, 1, 0]/(2*Nc),
 'Fdbcu4' : C["V8duLR"][0, 2, 1, 0]/2,
 'Fdbcu5' : C["S1udRR"][1, 0, 0, 2] - C["S8udduRR"][1, 2, 0, 0]/4 - C["S8udRR"][1, 0, 0, 2]/(2*Nc),
 'Fdbcu6' : -C["S1udduRR"][1, 2, 0, 0]/2 + C["S8udduRR"][1, 2, 0, 0]/(4*Nc) + C["S8udRR"][1, 0, 0, 2]/2,
 'Fdbcu7' : -C["V8udduLR"][0, 0, 2, 1].conjugate(),
 'Fdbcu8' : -2*C["V1udduLR"][0, 0, 2, 1].conjugate() + C["V8udduLR"][0, 0, 2, 1].conjugate()/Nc,
 'Fdbcu9' : -C["S8udduRR"][1, 2, 0, 0]/16,
 'Fdbcu10' : -C["S1udduRR"][1, 2, 0, 0]/8 + C["S8udduRR"][1, 2, 0, 0]/(16*Nc)}

def Fpdbcu (C):
    return {'Fdbcu1p' : C["V1udRR"][0, 1, 2, 0].conjugate() - C["V8udRR"][0, 1, 2, 0].conjugate()/(2*Nc),
 'Fdbcu2p' : C["V8udRR"][0, 1, 2, 0].conjugate()/2,
 'Fdbcu3p' : C["V1udLR"][0, 1, 2, 0].conjugate() - C["V8udLR"][0, 1, 2, 0].conjugate()/(2*Nc),
 'Fdbcu4p' : C["V8udLR"][0, 1, 2, 0].conjugate()/2,
 'Fdbcu5p' : C["S1udRR"][0, 1, 2, 0].conjugate() - C["S8udduRR"][0, 0, 2, 1].conjugate()/4 - C["S8udRR"][0, 1, 2, 0].conjugate()/(2*Nc),
 'Fdbcu6p' : -C["S1udduRR"][0, 0, 2, 1].conjugate()/2 + C["S8udduRR"][0, 0, 2, 1].conjugate()/(4*Nc) + C["S8udRR"][0, 1, 2, 0].conjugate()/2,
 'Fdbcu7p' : -C["V8udduLR"][1, 2, 0, 0],
 'Fdbcu8p' : -2*C["V1udduLR"][1, 2, 0, 0] + C["V8udduLR"][1, 2, 0, 0]/Nc,
 'Fdbcu9p' : -C["S8udduRR"][0, 0, 2, 1].conjugate()/16,
 'Fdbcu10p' : -C["S1udduRR"][0, 0, 2, 1].conjugate()/8 + C["S8udduRR"][0, 0, 2, 1].conjugate()/(16*Nc)}


# Bern basis

# sbuc

def Bernsbuc (Fsbuc):
    return {'1sbuc' : -Fsbuc['Fsbuc1']/3 + (4*Fsbuc['Fsbuc3'])/3 - Fsbuc['Fsbuc2']/(3*Nc) + (4*Fsbuc['Fsbuc4'])/(3*Nc),
 '2sbuc' : (-2*Fsbuc['Fsbuc2'])/3 + (8*Fsbuc['Fsbuc4'])/3,
 '3sbuc' : Fsbuc['Fsbuc1']/12 - Fsbuc['Fsbuc3']/12 + Fsbuc['Fsbuc2']/(12*Nc) - Fsbuc['Fsbuc4']/(12*Nc),
 '4sbuc' : Fsbuc['Fsbuc2']/6 - Fsbuc['Fsbuc4']/6,
 '5sbuc' : -Fsbuc['Fsbuc5']/3 + (4*Fsbuc['Fsbuc7'])/3 - Fsbuc['Fsbuc6']/(3*Nc) + (4*Fsbuc['Fsbuc8'])/(3*Nc),
 '6sbuc' : (-2*Fsbuc['Fsbuc6'])/3 + (8*Fsbuc['Fsbuc8'])/3,
 '7sbuc' : Fsbuc['Fsbuc5']/3 - Fsbuc['Fsbuc7']/3 + Fsbuc['Fsbuc9'] + Fsbuc['Fsbuc10']/Nc + Fsbuc['Fsbuc6']/(3*Nc) - Fsbuc['Fsbuc8']/(3*Nc),
 '8sbuc' : 2*Fsbuc['Fsbuc10'] + (2*Fsbuc['Fsbuc6'])/3 - (2*Fsbuc['Fsbuc8'])/3,
 '9sbuc' : Fsbuc['Fsbuc5']/48 - Fsbuc['Fsbuc7']/48 + Fsbuc['Fsbuc6']/(48*Nc) - Fsbuc['Fsbuc8']/(48*Nc),
 '10sbuc' : Fsbuc['Fsbuc6']/24 - Fsbuc['Fsbuc8']/24}


def Bernpsbuc (Fpsbuc):
    return {'1psbuc' : -Fpsbuc['Fsbuc1p']/3 + (4*Fpsbuc['Fsbuc3p'])/3 - Fpsbuc['Fsbuc2p']/(3*Nc) + (4*Fpsbuc['Fsbuc4p'])/(3*Nc),
 '2psbuc' : (-2*Fpsbuc['Fsbuc2p'])/3 + (8*Fpsbuc['Fsbuc4p'])/3,
 '3psbuc' : Fpsbuc['Fsbuc1p']/12 - Fpsbuc['Fsbuc3p']/12 + Fpsbuc['Fsbuc2p']/(12*Nc) - Fpsbuc['Fsbuc4p']/(12*Nc),
 '4psbuc' : Fpsbuc['Fsbuc2p']/6 - Fpsbuc['Fsbuc4p']/6,
 '5psbuc' : -Fpsbuc['Fsbuc5p']/3 + (4*Fpsbuc['Fsbuc7p'])/3 - Fpsbuc['Fsbuc6p']/(3*Nc) + (4*Fpsbuc['Fsbuc8p'])/(3*Nc),
 '6psbuc' : (-2*Fpsbuc['Fsbuc6p'])/3 + (8*Fpsbuc['Fsbuc8p'])/3,
 '7psbuc' : Fpsbuc['Fsbuc5p']/3 - Fpsbuc['Fsbuc7p']/3 + Fpsbuc['Fsbuc9p'] + Fpsbuc['Fsbuc10p']/Nc + Fpsbuc['Fsbuc6p']/(3*Nc) - Fpsbuc['Fsbuc8p']/(3*Nc),
 '8psbuc' : 2*Fpsbuc['Fsbuc10p'] + (2*Fpsbuc['Fsbuc6p'])/3 - (2*Fpsbuc['Fsbuc8p'])/3,
 '9psbuc' : Fpsbuc['Fsbuc5p']/48 - Fpsbuc['Fsbuc7p']/48 + Fpsbuc['Fsbuc6p']/(48*Nc) - Fpsbuc['Fsbuc8p']/(48*Nc),
 '10psbuc' : Fpsbuc['Fsbuc6p']/24 - Fpsbuc['Fsbuc8p']/24}

# sbcu

def Bernsbcu (Fsbcu):
    return {'1sbcu' : -Fsbcu['Fsbcu1']/3 + (4*Fsbcu['Fsbcu3'])/3 - Fsbcu['Fsbcu2']/(3*Nc) + (4*Fsbcu['Fsbcu4'])/(3*Nc),
 '2sbcu' : (-2*Fsbcu['Fsbcu2'])/3 + (8*Fsbcu['Fsbcu4'])/3,
 '3sbcu' : Fsbcu['Fsbcu1']/12 - Fsbcu['Fsbcu3']/12 + Fsbcu['Fsbcu2']/(12*Nc) - Fsbcu['Fsbcu4']/(12*Nc),
 '4sbcu' : Fsbcu['Fsbcu2']/6 - Fsbcu['Fsbcu4']/6,
 '5sbcu' : -Fsbcu['Fsbcu5']/3 + (4*Fsbcu['Fsbcu7'])/3 - Fsbcu['Fsbcu6']/(3*Nc) + (4*Fsbcu['Fsbcu8'])/(3*Nc),
 '6sbcu' : (-2*Fsbcu['Fsbcu6'])/3 + (8*Fsbcu['Fsbcu8'])/3,
 '7sbcu' : Fsbcu['Fsbcu5']/3 - Fsbcu['Fsbcu7']/3 + Fsbcu['Fsbcu9'] + Fsbcu['Fsbcu10']/Nc + Fsbcu['Fsbcu6']/(3*Nc) - Fsbcu['Fsbcu8']/(3*Nc),
 '8sbcu' : 2*Fsbcu['Fsbcu10'] + (2*Fsbcu['Fsbcu6'])/3 - (2*Fsbcu['Fsbcu8'])/3,
 '9sbcu' : Fsbcu['Fsbcu5']/48 - Fsbcu['Fsbcu7']/48 + Fsbcu['Fsbcu6']/(48*Nc) - Fsbcu['Fsbcu8']/(48*Nc),
 '10sbcu' : Fsbcu['Fsbcu6']/24 - Fsbcu['Fsbcu8']/24}


def Bernpsbcu (Fpsbcu):
    return {'1psbcu' : -Fpsbcu['Fsbcu1p']/3 + (4*Fpsbcu['Fsbcu3p'])/3 - Fpsbcu['Fsbcu2p']/(3*Nc) + (4*Fpsbcu['Fsbcu4p'])/(3*Nc),
 '2psbcu' : (-2*Fpsbcu['Fsbcu2p'])/3 + (8*Fpsbcu['Fsbcu4p'])/3,
 '3psbcu' : Fpsbcu['Fsbcu1p']/12 - Fpsbcu['Fsbcu3p']/12 + Fpsbcu['Fsbcu2p']/(12*Nc) - Fpsbcu['Fsbcu4p']/(12*Nc),
 '4psbcu' : Fpsbcu['Fsbcu2p']/6 - Fpsbcu['Fsbcu4p']/6,
 '5psbcu' : -Fpsbcu['Fsbcu5p']/3 + (4*Fpsbcu['Fsbcu7p'])/3 - Fpsbcu['Fsbcu6p']/(3*Nc) + (4*Fpsbcu['Fsbcu8p'])/(3*Nc),
 '6psbcu' : (-2*Fpsbcu['Fsbcu6p'])/3 + (8*Fpsbcu['Fsbcu8p'])/3,
 '7psbcu' : Fpsbcu['Fsbcu5p']/3 - Fpsbcu['Fsbcu7p']/3 + Fpsbcu['Fsbcu9p'] + Fpsbcu['Fsbcu10p']/Nc + Fpsbcu['Fsbcu6p']/(3*Nc) - Fpsbcu['Fsbcu8p']/(3*Nc),
 '8psbcu' : 2*Fpsbcu['Fsbcu10p'] + (2*Fpsbcu['Fsbcu6p'])/3 - (2*Fpsbcu['Fsbcu8p'])/3,
 '9psbcu' : Fpsbcu['Fsbcu5p']/48 - Fpsbcu['Fsbcu7p']/48 + Fpsbcu['Fsbcu6p']/(48*Nc) - Fpsbcu['Fsbcu8p']/(48*Nc),
 '10psbcu' : Fpsbcu['Fsbcu6p']/24 - Fpsbcu['Fsbcu8p']/24}



# dbuc

def Berndbuc (Fdbuc):
    return {'1dbuc' : -Fdbuc['Fdbuc1']/3 + (4*Fdbuc['Fdbuc3'])/3 - Fdbuc['Fdbuc2']/(3*Nc) + (4*Fdbuc['Fdbuc4'])/(3*Nc),
 '2dbuc' : (-2*Fdbuc['Fdbuc2'])/3 + (8*Fdbuc['Fdbuc4'])/3,
 '3dbuc' : Fdbuc['Fdbuc1']/12 - Fdbuc['Fdbuc3']/12 + Fdbuc['Fdbuc2']/(12*Nc) - Fdbuc['Fdbuc4']/(12*Nc),
 '4dbuc' : Fdbuc['Fdbuc2']/6 - Fdbuc['Fdbuc4']/6,
 '5dbuc' : -Fdbuc['Fdbuc5']/3 + (4*Fdbuc['Fdbuc7'])/3 - Fdbuc['Fdbuc6']/(3*Nc) + (4*Fdbuc['Fdbuc8'])/(3*Nc),
 '6dbuc' : (-2*Fdbuc['Fdbuc6'])/3 + (8*Fdbuc['Fdbuc8'])/3,
 '7dbuc' : Fdbuc['Fdbuc5']/3 - Fdbuc['Fdbuc7']/3 + Fdbuc['Fdbuc9'] + Fdbuc['Fdbuc10']/Nc + Fdbuc['Fdbuc6']/(3*Nc) - Fdbuc['Fdbuc8']/(3*Nc),
 '8dbuc' : 2*Fdbuc['Fdbuc10'] + (2*Fdbuc['Fdbuc6'])/3 - (2*Fdbuc['Fdbuc8'])/3,
 '9dbuc' : Fdbuc['Fdbuc5']/48 - Fdbuc['Fdbuc7']/48 + Fdbuc['Fdbuc6']/(48*Nc) - Fdbuc['Fdbuc8']/(48*Nc),
 '10dbuc' : Fdbuc['Fdbuc6']/24 - Fdbuc['Fdbuc8']/24}


def Bernpdbuc (Fpdbuc):
    return {'1pdbuc' : -Fpdbuc['Fdbuc1p']/3 + (4*Fpdbuc['Fdbuc3p'])/3 - Fpdbuc['Fdbuc2p']/(3*Nc) + (4*Fpdbuc['Fdbuc4p'])/(3*Nc),
 '2pdbuc' : (-2*Fpdbuc['Fdbuc2p'])/3 + (8*Fpdbuc['Fdbuc4p'])/3,
 '3pdbuc' : Fpdbuc['Fdbuc1p']/12 - Fpdbuc['Fdbuc3p']/12 + Fpdbuc['Fdbuc2p']/(12*Nc) - Fpdbuc['Fdbuc4p']/(12*Nc),
 '4pdbuc' : Fpdbuc['Fdbuc2p']/6 - Fpdbuc['Fdbuc4p']/6,
 '5pdbuc' : -Fpdbuc['Fdbuc5p']/3 + (4*Fpdbuc['Fdbuc7p'])/3 - Fpdbuc['Fdbuc6p']/(3*Nc) + (4*Fpdbuc['Fdbuc8p'])/(3*Nc),
 '6pdbuc' : (-2*Fpdbuc['Fdbuc6p'])/3 + (8*Fpdbuc['Fdbuc8p'])/3,
 '7pdbuc' : Fpdbuc['Fdbuc5p']/3 - Fpdbuc['Fdbuc7p']/3 + Fpdbuc['Fdbuc9p'] + Fpdbuc['Fdbuc10p']/Nc + Fpdbuc['Fdbuc6p']/(3*Nc) - Fpdbuc['Fdbuc8p']/(3*Nc),
 '8pdbuc' : 2*Fpdbuc['Fdbuc10p'] + (2*Fpdbuc['Fdbuc6p'])/3 - (2*Fpdbuc['Fdbuc8p'])/3,
 '9pdbuc' : Fpdbuc['Fdbuc5p']/48 - Fpdbuc['Fdbuc7p']/48 + Fpdbuc['Fdbuc6p']/(48*Nc) - Fpdbuc['Fdbuc8p']/(48*Nc),
 '10pdbuc' : Fpdbuc['Fdbuc6p']/24 - Fpdbuc['Fdbuc8p']/24}

# dbcu

def Berndbcu (Fdbcu):
    return {'1dbcu' : -Fdbcu['Fdbcu1']/3 + (4*Fdbcu['Fdbcu3'])/3 - Fdbcu['Fdbcu2']/(3*Nc) + (4*Fdbcu['Fdbcu4'])/(3*Nc),
 '2dbcu' : (-2*Fdbcu['Fdbcu2'])/3 + (8*Fdbcu['Fdbcu4'])/3,
 '3dbcu' : Fdbcu['Fdbcu1']/12 - Fdbcu['Fdbcu3']/12 + Fdbcu['Fdbcu2']/(12*Nc) - Fdbcu['Fdbcu4']/(12*Nc),
 '4dbcu' : Fdbcu['Fdbcu2']/6 - Fdbcu['Fdbcu4']/6,
 '5dbcu' : -Fdbcu['Fdbcu5']/3 + (4*Fdbcu['Fdbcu7'])/3 - Fdbcu['Fdbcu6']/(3*Nc) + (4*Fdbcu['Fdbcu8'])/(3*Nc),
 '6dbcu' : (-2*Fdbcu['Fdbcu6'])/3 + (8*Fdbcu['Fdbcu8'])/3,
 '7dbcu' : Fdbcu['Fdbcu5']/3 - Fdbcu['Fdbcu7']/3 + Fdbcu['Fdbcu9'] + Fdbcu['Fdbcu10']/Nc + Fdbcu['Fdbcu6']/(3*Nc) - Fdbcu['Fdbcu8']/(3*Nc),
 '8dbcu' : 2*Fdbcu['Fdbcu10'] + (2*Fdbcu['Fdbcu6'])/3 - (2*Fdbcu['Fdbcu8'])/3,
 '9dbcu' : Fdbcu['Fdbcu5']/48 - Fdbcu['Fdbcu7']/48 + Fdbcu['Fdbcu6']/(48*Nc) - Fdbcu['Fdbcu8']/(48*Nc),
 '10dbcu' : Fdbcu['Fdbcu6']/24 - Fdbcu['Fdbcu8']/24}

def Bernpdbcu (Fpdbcu):
    return {'1pdbcu' : -Fpdbcu['Fdbcu1p']/3 + (4*Fpdbcu['Fdbcu3p'])/3 - Fpdbcu['Fdbcu2p']/(3*Nc) + (4*Fpdbcu['Fdbcu4p'])/(3*Nc),
 '2pdbcu' : (-2*Fpdbcu['Fdbcu2p'])/3 + (8*Fpdbcu['Fdbcu4p'])/3,
 '3pdbcu' : Fpdbcu['Fdbcu1p']/12 - Fpdbcu['Fdbcu3p']/12 + Fpdbcu['Fdbcu2p']/(12*Nc) - Fpdbcu['Fdbcu4p']/(12*Nc),
 '4pdbcu' : Fpdbcu['Fdbcu2p']/6 - Fpdbcu['Fdbcu4p']/6,
 '5pdbcu' : -Fpdbcu['Fdbcu5p']/3 + (4*Fpdbcu['Fdbcu7p'])/3 - Fpdbcu['Fdbcu6p']/(3*Nc) + (4*Fpdbcu['Fdbcu8p'])/(3*Nc),
 '6pdbcu' : (-2*Fpdbcu['Fdbcu6p'])/3 + (8*Fpdbcu['Fdbcu8p'])/3,
 '7pdbcu' : Fpdbcu['Fdbcu5p']/3 - Fpdbcu['Fdbcu7p']/3 + Fpdbcu['Fdbcu9p'] + Fpdbcu['Fdbcu10p']/Nc + Fpdbcu['Fdbcu6p']/(3*Nc) - Fpdbcu['Fdbcu8p']/(3*Nc),
 '8pdbcu' : 2*Fpdbcu['Fdbcu10p'] + (2*Fpdbcu['Fdbcu6p'])/3 - (2*Fpdbcu['Fdbcu8p'])/3,
 '9pdbcu' : Fpdbcu['Fdbcu5p']/48 - Fpdbcu['Fdbcu7p']/48 + Fpdbcu['Fdbcu6p']/(48*Nc) - Fpdbcu['Fdbcu8p']/(48*Nc),
 '10pdbcu' : Fpdbcu['Fdbcu6p']/24 - Fpdbcu['Fdbcu8p']/24}

## Class IV ##

# Fierz basis

# sbsd

def Fsbsd (C):
    return {'Fsbsd1' : C["VddLL"][0, 1, 2, 1].conjugate(),
 'Fsbsd2' : C["VddLL"][0, 1, 2, 1].conjugate(),
 'Fsbsd3' : C["V1ddLR"][1, 2, 1, 0] - C["V8ddLR"][1, 2, 1, 0]/(2*Nc),
 'Fsbsd4' : C["V8ddLR"][1, 2, 1, 0]/2,
 'Fsbsd5' : C["S1ddRR"][1, 0, 1, 2] - C["S8ddRR"][1, 0, 1, 2]/(2*Nc) - C["S8ddRR"][0, 1, 2, 1].conjugate()/4,
 'Fsbsd6' : C["S8ddRR"][1, 0, 1, 2]/2 - C["S1ddRR"][0, 1, 2, 1].conjugate()/2 + C["S8ddRR"][0, 1, 2, 1].conjugate()/(4*Nc),
 'Fsbsd7' : -C["V8ddLR"][0, 1, 2, 1].conjugate(),
 'Fsbsd8' : -2*C["V1ddLR"][0, 1, 2, 1].conjugate() + C["V8ddLR"][0, 1, 2, 1].conjugate()/Nc,
 'Fsbsd9' : -C["S8ddRR"][0, 1, 2, 1].conjugate()/16,
 'Fsbsd10' : -C["S1ddRR"][0, 1, 2, 1].conjugate()/8 + C["S8ddRR"][0, 1, 2, 1].conjugate()/(16*Nc)}

def Fpsbsd (C):
    return {'Fsbsd1p' : C["VddRR"][0, 1, 2, 1].conjugate(),
 'Fsbsd2p' : C["VddRR"][0, 1, 2, 1].conjugate(),
 'Fsbsd3p' : C["V1ddLR"][0, 1, 2, 1].conjugate() - C["V8ddLR"][0, 1, 2, 1].conjugate()/(2*Nc),
 'Fsbsd4p' : C["V8ddLR"][0, 1, 2, 1].conjugate()/2,
 'Fsbsd5p' : -C["S8ddRR"][1, 0, 1, 2]/4 + C["S1ddRR"][0, 1, 2, 1].conjugate() - C["S8ddRR"][0, 1, 2, 1].conjugate()/(2*Nc),
 'Fsbsd6p' : -C["S1ddRR"][1, 0, 1, 2]/2 + C["S8ddRR"][1, 0, 1, 2]/(4*Nc) + C["S8ddRR"][0, 1, 2, 1].conjugate()/2,
 'Fsbsd7p' : -C["V8ddLR"][1, 2, 1, 0],
 'Fsbsd8p' : -2*C["V1ddLR"][1, 2, 1, 0] + C["V8ddLR"][1, 2, 1, 0]/Nc,
 'Fsbsd9p' : -C["S8ddRR"][1, 0, 1, 2]/16,
 'Fsbsd10p' : -C["S1ddRR"][1, 0, 1, 2]/8 + C["S8ddRR"][1, 0, 1, 2]/(16*Nc)}

# dbds

def Fdbds (C):
    return {'Fdbds1' : C["VddLL"][0, 1, 0, 2],
 'Fdbds2' : C["VddLL"][0, 1, 0, 2],
 'Fdbds3' : C["V1ddLR"][0, 2, 0, 1] - C["V8ddLR"][0, 2, 0, 1]/(2*Nc),
 'Fdbds4' : C["V8ddLR"][0, 2, 0, 1]/2,
 'Fdbds5' : C["S1ddRR"][0, 1, 0, 2] - C["S8ddRR"][0, 1, 0, 2]/(2*Nc) - C["S8ddRR"][1, 0, 2, 0].conjugate()/4,
 'Fdbds6' : C["S8ddRR"][0, 1, 0, 2]/2 - C["S1ddRR"][1, 0, 2, 0].conjugate()/2 + C["S8ddRR"][1, 0, 2, 0].conjugate()/(4*Nc),
 'Fdbds7' : -C["V8ddLR"][0, 1, 0, 2],
 'Fdbds8' : -2*C["V1ddLR"][0, 1, 0, 2] + C["V8ddLR"][0, 1, 0, 2]/Nc,
 'Fdbds9' : -C["S8ddRR"][1, 0, 2, 0].conjugate()/16,
 'Fdbds10' : -C["S1ddRR"][1, 0, 2, 0].conjugate()/8 + C["S8ddRR"][1, 0, 2, 0].conjugate()/(16*Nc)}

def Fpdbds (C):
    return {'Fdbds1p' : C["VddRR"][0, 1, 0, 2],
 'Fdbds2p' : C["VddRR"][0, 1, 0, 2],
 'Fdbds3p' : C["V1ddLR"][0, 1, 0, 2] - C["V8ddLR"][0, 1, 0, 2]/(2*Nc),
 'Fdbds4p' : C["V8ddLR"][0, 1, 0, 2]/2,
 'Fdbds5p' : -C["S8ddRR"][0, 1, 0, 2]/4 + C["S1ddRR"][1, 0, 2, 0].conjugate() - C["S8ddRR"][1, 0, 2, 0].conjugate()/(2*Nc),
 'Fdbds6p' : -C["S1ddRR"][0, 1, 0, 2]/2 + C["S8ddRR"][0, 1, 0, 2]/(4*Nc) + C["S8ddRR"][1, 0, 2, 0].conjugate()/2,
 'Fdbds7p' : -C["V8ddLR"][0, 2, 0, 1],
 'Fdbds8p' : -2*C["V1ddLR"][0, 2, 0, 1] + C["V8ddLR"][0, 2, 0, 1]/Nc,
 'Fdbds9p' : -C["S8ddRR"][0, 1, 0, 2]/16,
 'Fdbds10p' : -C["S1ddRR"][0, 1, 0, 2]/8 + C["S8ddRR"][0, 1, 0, 2]/(16*Nc)}


# Bern basis
#different choice of basis for class IV! remains to be checked

# sbsd

def Bernsbsd (Fsbsd):
    return {'1sbsd' : -Fsbsd['Fsbsd1']/3 + (4*Fsbsd['Fsbsd3'])/3 - Fsbsd['Fsbsd2']/(3*Nc) + (4*Fsbsd['Fsbsd4'])/(3*Nc),
 '3sbsd' : Fsbsd['Fsbsd1']/12 - Fsbsd['Fsbsd3']/12 + Fsbsd['Fsbsd2']/(12*Nc) - Fsbsd['Fsbsd4']/(12*Nc),
 '5sbsd' : -Fsbsd['Fsbsd5']/3 + (4*Fsbsd['Fsbsd7'])/3 - Fsbsd['Fsbsd6']/(3*Nc) + (4*Fsbsd['Fsbsd8'])/(3*Nc),
 '7sbsd' : Fsbsd['Fsbsd5']/3 - Fsbsd['Fsbsd7']/3 + Fsbsd['Fsbsd9'] + Fsbsd['Fsbsd10']/Nc + Fsbsd['Fsbsd6']/(3*Nc) - Fsbsd['Fsbsd8']/(3*Nc),
 '9sbsd' : Fsbsd['Fsbsd5']/48 - Fsbsd['Fsbsd7']/48 + Fsbsd['Fsbsd6']/(48*Nc) - Fsbsd['Fsbsd8']/(48*Nc)}

def Bernpsbsd (Fpsbsd):
    return {'1psbsd' : -Fpsbsd['Fsbsd1p']/3 + (4*Fpsbsd['Fsbsd3p'])/3 - Fpsbsd['Fsbsd2p']/(3*Nc) + (4*Fpsbsd['Fsbsd4p'])/(3*Nc),
 '3psbsd' : Fpsbsd['Fsbsd1p']/12 - Fpsbsd['Fsbsd3p']/12 + Fpsbsd['Fsbsd2p']/(12*Nc) - Fpsbsd['Fsbsd4p']/(12*Nc),
 '5psbsd' : -Fpsbsd['Fsbsd5p']/3 + (4*Fpsbsd['Fsbsd7p'])/3 - Fpsbsd['Fsbsd6p']/(3*Nc) + (4*Fpsbsd['Fsbsd8p'])/(3*Nc),
 '7psbsd' : Fpsbsd['Fsbsd5p']/3 - Fpsbsd['Fsbsd7p']/3 + Fpsbsd['Fsbsd9p'] + Fpsbsd['Fsbsd10p']/Nc + Fpsbsd['Fsbsd6p']/(3*Nc) - Fpsbsd['Fsbsd8p']/(3*Nc),
 '9psbsd' : Fpsbsd['Fsbsd5p']/48 - Fpsbsd['Fsbsd7p']/48 + Fpsbsd['Fsbsd6p']/(48*Nc) - Fpsbsd['Fsbsd8p']/(48*Nc)}

# dbds
def Berndbds (Fdbds):
    return {'1dbds' : -Fdbds['Fdbds1']/3 + (4*Fdbds['Fdbds3'])/3 - Fdbds['Fdbds2']/(3*Nc) + (4*Fdbds['Fdbds4'])/(3*Nc),
 '3dbds' : Fdbds['Fdbds1']/12 - Fdbds['Fdbds3']/12 + Fdbds['Fdbds2']/(12*Nc) - Fdbds['Fdbds4']/(12*Nc),
 '5dbds' : -Fdbds['Fdbds5']/3 + (4*Fdbds['Fdbds7'])/3 - Fdbds['Fdbds6']/(3*Nc) + (4*Fdbds['Fdbds8'])/(3*Nc),
 '7dbds' : Fdbds['Fdbds5']/3 - Fdbds['Fdbds7']/3 + Fdbds['Fdbds9'] + Fdbds['Fdbds10']/Nc + Fdbds['Fdbds6']/(3*Nc) - Fdbds['Fdbds8']/(3*Nc),
 '9dbds' : Fdbds['Fdbds5']/48 - Fdbds['Fdbds7']/48 + Fdbds['Fdbds6']/(48*Nc) - Fdbds['Fdbds8']/(48*Nc)}

def Bernpdbds (Fpdbds):
    return {'1pdbds' : -Fpdbds['Fdbds1p']/3 + (4*Fpdbds['Fdbds3p'])/3 - Fpdbds['Fdbds2p']/(3*Nc) + (4*Fpdbds['Fdbds4p'])/(3*Nc),
 '3pdbds' : Fpdbds['Fdbds1p']/12 - Fpdbds['Fdbds3p']/12 + Fpdbds['Fdbds2p']/(12*Nc) - Fpdbds['Fdbds4p']/(12*Nc),
 '5pdbds' : -Fpdbds['Fdbds5p']/3 + (4*Fpdbds['Fdbds7p'])/3 - Fpdbds['Fdbds6p']/(3*Nc) + (4*Fpdbds['Fdbds8p'])/(3*Nc),
 '7pdbds' : Fpdbds['Fdbds5p']/3 - Fpdbds['Fdbds7p']/3 + Fpdbds['Fdbds9p'] + Fpdbds['Fdbds10p']/Nc + Fpdbds['Fdbds6p']/(3*Nc) - Fpdbds['Fdbds8p']/(3*Nc),
 '9pdbds' : Fpdbds['Fdbds5p']/48 - Fpdbds['Fdbds7p']/48 + Fpdbds['Fdbds6p']/(48*Nc) - Fpdbds['Fdbds8p']/(48*Nc)}

# dbsb shoudl be added


## Class V ##

# four-quark operators

# Fierz basis

# sbuu
def Fsbuu (C):
    return {'Fsbuu1' : C["V1udLL"][0, 0, 1, 2] - C["V8udLL"][0, 0, 1, 2]/(2*Nc),
 'Fsbuu2' : C["V8udLL"][0, 0, 1, 2]/2,
 'Fsbuu3' : C["V1duLR"][1, 2, 0, 0] - C["V8duLR"][1, 2, 0, 0]/(2*Nc),
 'Fsbuu4' : C["V8duLR"][1, 2, 0, 0]/2,
 'Fsbuu5' : C["S1udRR"][0, 0, 1, 2] - C["S8udduRR"][0, 2, 1, 0]/4 - C["S8udRR"][0, 0, 1, 2]/(2*Nc),
 'Fsbuu6' : -C["S1udduRR"][0, 2, 1, 0]/2 + C["S8udduRR"][0, 2, 1, 0]/(4*Nc) + C["S8udRR"][0, 0, 1, 2]/2,
 'Fsbuu7' : -C["V8udduLR"][0, 1, 2, 0].conjugate(),
 'Fsbuu8' : -2*C["V1udduLR"][0, 1, 2, 0].conjugate() + C["V8udduLR"][0, 1, 2, 0].conjugate()/Nc,
 'Fsbuu9' : -C["S8udduRR"][0, 2, 1, 0]/16,
 'Fsbuu10' : -C["S1udduRR"][0, 2, 1, 0]/8 + C["S8udduRR"][0, 2, 1, 0]/(16*Nc)}

def Fpsbuu (C):
    return {'Fsbuu1p' : C["V1udRR"][0, 0, 1, 2] - C["V8udRR"][0, 0, 1, 2]/(2*Nc),
 'Fsbuu2p' : C["V8udRR"][0, 0, 1, 2]/2,
 'Fsbuu3p' : C["V1udLR"][0, 0, 1, 2] - C["V8udLR"][0, 0, 1, 2]/(2*Nc),
 'Fsbuu4p' : C["V8udLR"][0, 0, 1, 2]/2,
 'Fsbuu5p' : C["S1udRR"][0, 0, 2, 1].conjugate() - C["S8udduRR"][0, 1, 2, 0].conjugate()/4 - C["S8udRR"][0, 0, 2, 1].conjugate()/(2*Nc),
 'Fsbuu6p' : -C["S1udduRR"][0, 1, 2, 0].conjugate()/2 + C["S8udduRR"][0, 1, 2, 0].conjugate()/(4*Nc) + C["S8udRR"][0, 0, 2, 1].conjugate()/2,
 'Fsbuu7p' : -C["V8udduLR"][0, 2, 1, 0],
 'Fsbuu8p' : -2*C["V1udduLR"][0, 2, 1, 0] + C["V8udduLR"][0, 2, 1, 0]/Nc,
 'Fsbuu9p' : -C["S8udduRR"][0, 1, 2, 0].conjugate()/16,
 'Fsbuu10p' : -C["S1udduRR"][0, 1, 2, 0].conjugate()/8 + C["S8udduRR"][0, 1, 2, 0].conjugate()/(16*Nc)}


# dbuu
def Fdbuu (C):
    return {'Fdbuu1' : C["V1udLL"][0, 0, 0, 2] - C["V8udLL"][0, 0, 0, 2]/(2*Nc),
 'Fdbuu2' : C["V8udLL"][0, 0, 0, 2]/2,
 'Fdbuu3' : C["V1duLR"][0, 2, 0, 0] - C["V8duLR"][0, 2, 0, 0]/(2*Nc),
 'Fdbuu4' : C["V8duLR"][0, 2, 0, 0]/2,
 'Fdbuu5' : C["S1udRR"][0, 0, 0, 2] - C["S8udduRR"][0, 2, 0, 0]/4 - C["S8udRR"][0, 0, 0, 2]/(2*Nc),
 'Fdbuu6' : -C["S1udduRR"][0, 2, 0, 0]/2 + C["S8udduRR"][0, 2, 0, 0]/(4*Nc) + C["S8udRR"][0, 0, 0, 2]/2,
 'Fdbuu7' : -C["V8udduLR"][0, 0, 2, 0].conjugate(),
 'Fdbuu8' : -2*C["V1udduLR"][0, 0, 2, 0].conjugate() + C["V8udduLR"][0, 0, 2, 0].conjugate()/Nc,
 'Fdbuu9' : -C["S8udduRR"][0, 2, 0, 0]/16,
 'Fdbuu10' : -C["S1udduRR"][0, 2, 0, 0]/8 + C["S8udduRR"][0, 2, 0, 0]/(16*Nc)}

def Fpdbuu (C):
    return {'Fdbuu1p' : C["V1udRR"][0, 0, 0, 2] - C["V8udRR"][0, 0, 0, 2]/(2*Nc),
 'Fdbuu2p' : C["V8udRR"][0, 0, 0, 2]/2,
 'Fdbuu3p' : C["V1udLR"][0, 0, 0, 2] - C["V8udLR"][0, 0, 0, 2]/(2*Nc),
 'Fdbuu4p' : C["V8udLR"][0, 0, 0, 2]/2,
 'Fdbuu5p' : C["S1udRR"][0, 0, 2, 0].conjugate() - C["S8udduRR"][0, 0, 2, 0].conjugate()/4 - C["S8udRR"][0, 0, 2, 0].conjugate()/(2*Nc),
 'Fdbuu6p' : -C["S1udduRR"][0, 0, 2, 0].conjugate()/2 + C["S8udduRR"][0, 0, 2, 0].conjugate()/(4*Nc) + C["S8udRR"][0, 0, 2, 0].conjugate()/2,
 'Fdbuu7p' : -C["V8udduLR"][0, 2, 0, 0],
 'Fdbuu8p' : -2*C["V1udduLR"][0, 2, 0, 0] + C["V8udduLR"][0, 2, 0, 0]/Nc,
 'Fdbuu9p' : -C["S8udduRR"][0, 0, 2, 0].conjugate()/16,
 'Fdbuu10p' : -C["S1udduRR"][0, 0, 2, 0].conjugate()/8 + C["S8udduRR"][0, 0, 2, 0].conjugate()/(16*Nc)}

# sbcc

def Fsbcc (C):
    return {'Fsbcc1' : C["V1udLL"][1, 1, 1, 2] - C["V8udLL"][1, 1, 1, 2]/(2*Nc),
 'Fsbcc2' : C["V8udLL"][1, 1, 1, 2]/2,
 'Fsbcc3' : C["V1duLR"][1, 2, 1, 1] - C["V8duLR"][1, 2, 1, 1]/(2*Nc),
 'Fsbcc4' : C["V8duLR"][1, 2, 1, 1]/2,
 'Fsbcc5' : C["S1udRR"][1, 1, 1, 2] - C["S8udduRR"][1, 2, 1, 1]/4 - C["S8udRR"][1, 1, 1, 2]/(2*Nc),
 'Fsbcc6' : -C["S1udduRR"][1, 2, 1, 1]/2 + C["S8udduRR"][1, 2, 1, 1]/(4*Nc) + C["S8udRR"][1, 1, 1, 2]/2,
 'Fsbcc7' : -C["V8udduLR"][1, 1, 2, 1].conjugate(),
 'Fsbcc8' : -2*C["V1udduLR"][1, 1, 2, 1].conjugate() + C["V8udduLR"][1, 1, 2, 1].conjugate()/Nc,
 'Fsbcc9' : -C["S8udduRR"][1, 2, 1, 1]/16,
 'Fsbcc10' : -C["S1udduRR"][1, 2, 1, 1]/8 + C["S8udduRR"][1, 2, 1, 1]/(16*Nc)}

def Fpsbcc (C):
    return {'Fsbcc1p' : C["V1udRR"][1, 1, 1, 2] - C["V8udRR"][1, 1, 1, 2]/(2*Nc),
 'Fsbcc2p' : C["V8udRR"][1, 1, 1, 2]/2,
 'Fsbcc3p' : C["V1udLR"][1, 1, 1, 2] - C["V8udLR"][1, 1, 1, 2]/(2*Nc),
 'Fsbcc4p' : C["V8udLR"][1, 1, 1, 2]/2,
 'Fsbcc5p' : C["S1udRR"][1, 1, 2, 1].conjugate() - C["S8udduRR"][1, 1, 2, 1].conjugate()/4 - C["S8udRR"][1, 1, 2, 1].conjugate()/(2*Nc),
 'Fsbcc6p' : -C["S1udduRR"][1, 1, 2, 1].conjugate()/2 + C["S8udduRR"][1, 1, 2, 1].conjugate()/(4*Nc) + C["S8udRR"][1, 1, 2, 1].conjugate()/2,
 'Fsbcc7p' : -C["V8udduLR"][1, 2, 1, 1],
 'Fsbcc8p' : -2*C["V1udduLR"][1, 2, 1, 1] + C["V8udduLR"][1, 2, 1, 1]/Nc,
 'Fsbcc9p' : -C["S8udduRR"][1, 1, 2, 1].conjugate()/16,
 'Fsbcc10p' : -C["S1udduRR"][1, 1, 2, 1].conjugate()/8 + C["S8udduRR"][1, 1, 2, 1].conjugate()/(16*Nc)}

 # dbcc

def Fdbcc (C):
    return {'Fdbcc1' : C["V1udLL"][1, 1, 0, 2] - C["V8udLL"][1, 1, 0, 2]/(2*Nc),
 'Fdbcc2' : C["V8udLL"][1, 1, 0, 2]/2,
 'Fdbcc3' : C["V1duLR"][0, 2, 1, 1] - C["V8duLR"][0, 2, 1, 1]/(2*Nc),
 'Fdbcc4' : C["V8duLR"][0, 2, 1, 1]/2,
 'Fdbcc5' : C["S1udRR"][1, 1, 0, 2] - C["S8udduRR"][1, 2, 0, 1]/4 - C["S8udRR"][1, 1, 0, 2]/(2*Nc),
 'Fdbcc6' : -C["S1udduRR"][1, 2, 0, 1]/2 + C["S8udduRR"][1, 2, 0, 1]/(4*Nc) + C["S8udRR"][1, 1, 0, 2]/2,
 'Fdbcc7' : -C["V8udduLR"][1, 0, 2, 1].conjugate(),
 'Fdbcc8' : -2*C["V1udduLR"][1, 0, 2, 1].conjugate() + C["V8udduLR"][1, 0, 2, 1].conjugate()/Nc,
 'Fdbcc9' : -C["S8udduRR"][1, 2, 0, 1]/16,
 'Fdbcc10' : -C["S1udduRR"][1, 2, 0, 1]/8 + C["S8udduRR"][1, 2, 0, 1]/(16*Nc)}

def Fpdbcc (C):
    return {'Fdbcc1p' : C["V1udRR"][1, 1, 0, 2] - C["V8udRR"][1, 1, 0, 2]/(2*Nc),
 'Fdbcc2p' : C["V8udRR"][1, 1, 0, 2]/2,
 'Fdbcc3p' : C["V1udLR"][1, 1, 0, 2] - C["V8udLR"][1, 1, 0, 2]/(2*Nc),
 'Fdbcc4p' : C["V8udLR"][1, 1, 0, 2]/2,
 'Fdbcc5p' : C["S1udRR"][1, 1, 2, 0].conjugate() - C["S8udduRR"][1, 0, 2, 1].conjugate()/4 - C["S8udRR"][1, 1, 2, 0].conjugate()/(2*Nc),
 'Fdbcc6p' : -C["S1udduRR"][1, 0, 2, 1].conjugate()/2 + C["S8udduRR"][1, 0, 2, 1].conjugate()/(4*Nc) + C["S8udRR"][1, 1, 2, 0].conjugate()/2,
 'Fdbcc7p' : -C["V8udduLR"][1, 2, 0, 1],
 'Fdbcc8p' : -2*C["V1udduLR"][1, 2, 0, 1] + C["V8udduLR"][1, 2, 0, 1]/Nc,
 'Fdbcc9p' : -C["S8udduRR"][1, 0, 2, 1].conjugate()/16,
 'Fdbcc10p' : -C["S1udduRR"][1, 0, 2, 1].conjugate()/8 + C["S8udduRR"][1, 0, 2, 1].conjugate()/(16*Nc)}


 # sbdd

def Fsbdd (C):
    return {'Fsbdd1' : C["VddLL"][0, 0, 1, 2],
 'Fsbdd2' : C["VddLL"][0, 1, 2, 0].conjugate(),
 'Fsbdd3' : C["V1ddLR"][1, 2, 0, 0] - C["V8ddLR"][1, 2, 0, 0]/(2*Nc),
 'Fsbdd4' : C["V8ddLR"][1, 2, 0, 0]/2,
 'Fsbdd5' : C["S1ddRR"][0, 0, 1, 2] - C["S8ddRR"][0, 0, 1, 2]/(2*Nc) - C["S8ddRR"][0, 2, 1, 0]/4,
 'Fsbdd6' : -C["S1ddRR"][0, 2, 1, 0]/2 + C["S8ddRR"][0, 0, 1, 2]/2 + C["S8ddRR"][0, 2, 1, 0]/(4*Nc),
 'Fsbdd7' : -C["V8ddLR"][0, 1, 2, 0].conjugate(),
 'Fsbdd8' : -2*C["V1ddLR"][0, 1, 2, 0].conjugate() + C["V8ddLR"][0, 1, 2, 0].conjugate()/Nc,
 'Fsbdd9' : -C["S8ddRR"][0, 2, 1, 0]/16,
 'Fsbdd10' : -C["S1ddRR"][0, 2, 1, 0]/8 + C["S8ddRR"][0, 2, 1, 0]/(16*Nc)}

def Fpsbdd (C):
    return {'Fsbdd1p' : C["VddRR"][0, 0, 1, 2],
 'Fsbdd2p' : C["VddRR"][0, 1, 2, 0].conjugate(),
 'Fsbdd3p' : C["V1ddLR"][0, 0, 1, 2] - C["V8ddLR"][0, 0, 1, 2]/(2*Nc),
 'Fsbdd4p' : C["V8ddLR"][0, 0, 1, 2]/2,
 'Fsbdd5p' : C["S1ddRR"][0, 0, 2, 1].conjugate() - C["S8ddRR"][0, 0, 2, 1].conjugate()/(2*Nc) - C["S8ddRR"][0, 1, 2, 0].conjugate()/4,
 'Fsbdd6p' : -C["S1ddRR"][0, 1, 2, 0].conjugate()/2 + C["S8ddRR"][0, 0, 2, 1].conjugate()/2 + C["S8ddRR"][0, 1, 2, 0].conjugate()/(4*Nc),
 'Fsbdd7p' : -C["V8ddLR"][0, 2, 1, 0],
 'Fsbdd8p' : -2*C["V1ddLR"][0, 2, 1, 0] + C["V8ddLR"][0, 2, 1, 0]/Nc,
 'Fsbdd9p' : -C["S8ddRR"][0, 1, 2, 0].conjugate()/16,
 'Fsbdd10p' : -C["S1ddRR"][0, 1, 2, 0].conjugate()/8 + C["S8ddRR"][0, 1, 2, 0].conjugate()/(16*Nc)}

# dbdd

def Fdbdd (C):
    return {'Fdbdd1' : C["VddLL"][0, 0, 0, 2],
 'Fdbdd2' : C["VddLL"][0, 0, 0, 2],
 'Fdbdd3' : C["V1ddLR"][0, 2, 0, 0] - C["V8ddLR"][0, 2, 0, 0]/(2*Nc),
 'Fdbdd4' : C["V8ddLR"][0, 2, 0, 0]/2,
 'Fdbdd5' : C["S1ddRR"][0, 0, 0, 2] - C["S8ddRR"][0, 0, 0, 2]/4 - C["S8ddRR"][0, 0, 0, 2]/(2*Nc),
 'Fdbdd6' : -C["S1ddRR"][0, 0, 0, 2]/2 + C["S8ddRR"][0, 0, 0, 2]/2 + C["S8ddRR"][0, 0, 0, 2]/(4*Nc),
 'Fdbdd7' : -C["V8ddLR"][0, 0, 0, 2],
 'Fdbdd8' : -2*C["V1ddLR"][0, 0, 0, 2] + C["V8ddLR"][0, 0, 0, 2]/Nc,
 'Fdbdd9' : -C["S8ddRR"][0, 0, 0, 2]/16,
 'Fdbdd10' : -C["S1ddRR"][0, 0, 0, 2]/8 + C["S8ddRR"][0, 0, 0, 2]/(16*Nc)}

def Fpdbdd (C):
    return {'Fdbdd1p' : C["VddRR"][0, 0, 0, 2],
 'Fdbdd2p' : C["VddRR"][0, 0, 0, 2],
 'Fdbdd3p' : C["V1ddLR"][0, 0, 0, 2] - C["V8ddLR"][0, 0, 0, 2]/(2*Nc),
 'Fdbdd4p' : C["V8ddLR"][0, 0, 0, 2]/2,
 'Fdbdd5p' : C["S1ddRR"][0, 0, 2, 0].conjugate() - C["S8ddRR"][0, 0, 2, 0].conjugate()/4 - C["S8ddRR"][0, 0, 2, 0].conjugate()/(2*Nc),
 'Fdbdd6p' : -C["S1ddRR"][0, 0, 2, 0].conjugate()/2 + C["S8ddRR"][0, 0, 2, 0].conjugate()/2 + C["S8ddRR"][0, 0, 2, 0].conjugate()/(4*Nc),
 'Fdbdd7p' : -C["V8ddLR"][0, 2, 0, 0],
 'Fdbdd8p' : -2*C["V1ddLR"][0, 2, 0, 0] + C["V8ddLR"][0, 2, 0, 0]/Nc,
 'Fdbdd9p' : -C["S8ddRR"][0, 0, 2, 0].conjugate()/16,
 'Fdbdd10p' : -C["S1ddRR"][0, 0, 2, 0].conjugate()/8 + C["S8ddRR"][0, 0, 2, 0].conjugate()/(16*Nc)}


 # sbss

def Fsbss (C):
    return {'Fsbss1' : C["VddLL"][1, 1, 1, 2],
 'Fsbss2' : C["VddLL"][1, 1, 1, 2],
 'Fsbss3' : C["V1ddLR"][1, 2, 1, 1] - C["V8ddLR"][1, 2, 1, 1]/(2*Nc),
 'Fsbss4' : C["V8ddLR"][1, 2, 1, 1]/2,
 'Fsbss5' : C["S1ddRR"][1, 1, 1, 2] - C["S8ddRR"][1, 1, 1, 2]/4 - C["S8ddRR"][1, 1, 1, 2]/(2*Nc),
 'Fsbss6' : -C["S1ddRR"][1, 1, 1, 2]/2 + C["S8ddRR"][1, 1, 1, 2]/2 + C["S8ddRR"][1, 1, 1, 2]/(4*Nc),
 'Fsbss7' : -C["V8ddLR"][1, 1, 1, 2],
 'Fsbss8' : -2*C["V1ddLR"][1, 1, 1, 2] + C["V8ddLR"][1, 1, 1, 2]/Nc,
 'Fsbss9' : -C["S8ddRR"][1, 1, 1, 2]/16,
 'Fsbss10' : -C["S1ddRR"][1, 1, 1, 2]/8 + C["S8ddRR"][1, 1, 1, 2]/(16*Nc)}


def Fpsbss (C):
    return {'Fsbss1p' : C["VddRR"][1, 1, 1, 2],
 'Fsbss2p' : C["VddRR"][1, 1, 1, 2],
 'Fsbss3p' : C["V1ddLR"][1, 1, 1, 2] - C["V8ddLR"][1, 1, 1, 2]/(2*Nc),
 'Fsbss4p' : C["V8ddLR"][1, 1, 1, 2]/2,
 'Fsbss5p' : C["S1ddRR"][1, 1, 2, 1].conjugate() - C["S8ddRR"][1, 1, 2, 1].conjugate()/4 - C["S8ddRR"][1, 1, 2, 1].conjugate()/(2*Nc),
 'Fsbss6p' : -C["S1ddRR"][1, 1, 2, 1].conjugate()/2 + C["S8ddRR"][1, 1, 2, 1].conjugate()/2 + C["S8ddRR"][1, 1, 2, 1].conjugate()/(4*Nc),
 'Fsbss7p' : -C["V8ddLR"][1, 2, 1, 1],
 'Fsbss8p' : -2*C["V1ddLR"][1, 2, 1, 1] + C["V8ddLR"][1, 2, 1, 1]/Nc,
 'Fsbss9p' : -C["S8ddRR"][1, 1, 2, 1].conjugate()/16,
 'Fsbss10p' : -C["S1ddRR"][1, 1, 2, 1].conjugate()/8 + C["S8ddRR"][1, 1, 2, 1].conjugate()/(16*Nc)}


# dbss

def Fdbss (C):
    return {'Fdbss1' : C["VddLL"][0, 2, 1, 1],
 'Fdbss2' : C["VddLL"][0, 1, 1, 2],
 'Fdbss3' : C["V1ddLR"][0, 2, 1, 1] - C["V8ddLR"][0, 2, 1, 1]/(2*Nc),
 'Fdbss4' : C["V8ddLR"][0, 2, 1, 1]/2,
 'Fdbss5' : C["S1ddRR"][0, 2, 1, 1] - C["S8ddRR"][0, 1, 1, 2]/4 - C["S8ddRR"][0, 2, 1, 1]/(2*Nc),
 'Fdbss6' : -C["S1ddRR"][0, 1, 1, 2]/2 + C["S8ddRR"][0, 1, 1, 2]/(4*Nc) + C["S8ddRR"][0, 2, 1, 1]/2,
 'Fdbss7' : -C["V8ddLR"][0, 1, 1, 2],
 'Fdbss8' : -2*C["V1ddLR"][0, 1, 1, 2] + C["V8ddLR"][0, 1, 1, 2]/Nc,
 'Fdbss9' : -C["S8ddRR"][0, 1, 1, 2]/16,
 'Fdbss10' : -C["S1ddRR"][0, 1, 1, 2]/8 + C["S8ddRR"][0, 1, 1, 2]/(16*Nc)}

def Fpdbss (C):
    return {'Fdbss1p' : C["VddRR"][0, 2, 1, 1],
 'Fdbss2p' : C["VddRR"][0, 1, 1, 2],
 'Fdbss3p' : C["V1ddLR"][1, 1, 0, 2] - C["V8ddLR"][1, 1, 0, 2]/(2*Nc),
 'Fdbss4p' : C["V8ddLR"][1, 1, 0, 2]/2,
 'Fdbss5p' : C["S1ddRR"][1, 1, 2, 0].conjugate() - C["S8ddRR"][1, 0, 2, 1].conjugate()/4 - C["S8ddRR"][1, 1, 2, 0].conjugate()/(2*Nc),
 'Fdbss6p' : -C["S1ddRR"][1, 0, 2, 1].conjugate()/2 + C["S8ddRR"][1, 0, 2, 1].conjugate()/(4*Nc) + C["S8ddRR"][1, 1, 2, 0].conjugate()/2,
 'Fdbss7p' : -C["V8ddLR"][1, 2, 0, 1],
 'Fdbss8p' : -2*C["V1ddLR"][1, 2, 0, 1] + C["V8ddLR"][1, 2, 0, 1]/Nc,
 'Fdbss9p' : -C["S8ddRR"][1, 0, 2, 1].conjugate()/16,
 'Fdbss10p' : -C["S1ddRR"][1, 0, 2, 1].conjugate()/8 + C["S8ddRR"][1, 0, 2, 1].conjugate()/(16*Nc)}

 # sbbb

def Fsbbb (C):
    return {'Fsbbb1' : C["VddLL"][1, 2, 2, 2],
 'Fsbbb2' : C["VddLL"][1, 2, 2, 2],
 'Fsbbb3' : C["V1ddLR"][1, 2, 2, 2] - C["V8ddLR"][1, 2, 2, 2]/(2*Nc),
 'Fsbbb4' : C["V8ddLR"][1, 2, 2, 2]/2,
 'Fsbbb5' : C["S1ddRR"][1, 2, 2, 2] - C["S8ddRR"][1, 2, 2, 2]/4 - C["S8ddRR"][1, 2, 2, 2]/(2*Nc),
 'Fsbbb6' : -C["S1ddRR"][1, 2, 2, 2]/2 + C["S8ddRR"][1, 2, 2, 2]/2 + C["S8ddRR"][1, 2, 2, 2]/(4*Nc),
 'Fsbbb7' : -C["V8ddLR"][1, 2, 2, 2],
 'Fsbbb8' : -2*C["V1ddLR"][1, 2, 2, 2] + C["V8ddLR"][1, 2, 2, 2]/Nc,
 'Fsbbb9' : -C["S8ddRR"][1, 2, 2, 2]/16,
 'Fsbbb10' : -C["S1ddRR"][1, 2, 2, 2]/8 + C["S8ddRR"][1, 2, 2, 2]/(16*Nc)}


def Fpsbbb (C):
    return {'Fsbbb1p' : C["VddRR"][1, 2, 2, 2],
 'Fsbbb2p' : C["VddRR"][1, 2, 2, 2],
 'Fsbbb3p' : C["V1ddLR"][2, 2, 1, 2] - C["V8ddLR"][2, 2, 1, 2]/(2*Nc),
 'Fsbbb4p' : C["V8ddLR"][2, 2, 1, 2]/2,
 'Fsbbb5p' : C["S1ddRR"][2, 1, 2, 2].conjugate() - C["S8ddRR"][2, 1, 2, 2].conjugate()/4 - C["S8ddRR"][2, 1, 2, 2].conjugate()/(2*Nc),
 'Fsbbb6p' : -C["S1ddRR"][2, 1, 2, 2].conjugate()/2 + C["S8ddRR"][2, 1, 2, 2].conjugate()/2 + C["S8ddRR"][2, 1, 2, 2].conjugate()/(4*Nc),
 'Fsbbb7p' : -C["V8ddLR"][2, 2, 1, 2],
 'Fsbbb8p' : -2*C["V1ddLR"][2, 2, 1, 2] + C["V8ddLR"][2, 2, 1, 2]/Nc,
 'Fsbbb9p' : -C["S8ddRR"][2, 1, 2, 2].conjugate()/16,
 'Fsbbb10p' : -C["S1ddRR"][2, 1, 2, 2].conjugate()/8 + C["S8ddRR"][2, 1, 2, 2].conjugate()/(16*Nc)}

# dbbb

def Fdbbb (C):
    return {'Fdbbb1' : C["VddLL"][0, 2, 2, 2],
 'Fdbbb2' : C["VddLL"][0, 2, 2, 2],
 'Fdbbb3' : C["V1ddLR"][0, 2, 2, 2] - C["V8ddLR"][0, 2, 2, 2]/(2*Nc),
 'Fdbbb4' : C["V8ddLR"][0, 2, 2, 2]/2,
 'Fdbbb5' : C["S1ddRR"][0, 2, 2, 2] - C["S8ddRR"][0, 2, 2, 2]/4 - C["S8ddRR"][0, 2, 2, 2]/(2*Nc),
 'Fdbbb6' : -C["S1ddRR"][0, 2, 2, 2]/2 + C["S8ddRR"][0, 2, 2, 2]/2 + C["S8ddRR"][0, 2, 2, 2]/(4*Nc),
 'Fdbbb7' : -C["V8ddLR"][0, 2, 2, 2],
 'Fdbbb8' : -2*C["V1ddLR"][0, 2, 2, 2] + C["V8ddLR"][0, 2, 2, 2]/Nc,
 'Fdbbb9' : -C["S8ddRR"][0, 2, 2, 2]/16,
 'Fdbbb10' : -C["S1ddRR"][0, 2, 2, 2]/8 + C["S8ddRR"][0, 2, 2, 2]/(16*Nc)}


def Fpdbbb (C):
    return {'Fdbbb1p' : C["VddRR"][0, 2, 2, 2],
 'Fdbbb2p' : C["VddRR"][0, 2, 2, 2],
 'Fdbbb3p' : C["V1ddLR"][2, 2, 0, 2] - C["V8ddLR"][2, 2, 0, 2]/(2*Nc),
 'Fdbbb4p' : C["V8ddLR"][2, 2, 0, 2]/2,
 'Fdbbb5p' : C["S1ddRR"][2, 0, 2, 2].conjugate() - C["S8ddRR"][2, 0, 2, 2].conjugate()/4 - C["S8ddRR"][2, 0, 2, 2].conjugate()/(2*Nc),
 'Fdbbb6p' : -C["S1ddRR"][2, 0, 2, 2].conjugate()/2 + C["S8ddRR"][2, 0, 2, 2].conjugate()/2 + C["S8ddRR"][2, 0, 2, 2].conjugate()/(4*Nc),
 'Fdbbb7p' : -C["V8ddLR"][2, 2, 0, 2],
 'Fdbbb8p' : -2*C["V1ddLR"][2, 2, 0, 2] + C["V8ddLR"][2, 2, 0, 2]/Nc,
 'Fdbbb9p' : -C["S8ddRR"][2, 0, 2, 2].conjugate()/16,
 'Fdbbb10p' : -C["S1ddRR"][2, 0, 2, 2].conjugate()/8 + C["S8ddRR"][2, 0, 2, 2].conjugate()/(16*Nc)}


# Bern basis

# sbuu

def Bernsbuu (Fsbuu):
    return {'1sbuu' : -Fsbuu['Fsbuu1']/3 + (4*Fsbuu['Fsbuu3'])/3 - Fsbuu['Fsbuu2']/(3*Nc) + (4*Fsbuu['Fsbuu4'])/(3*Nc),
 '2sbuu' : (-2*Fsbuu['Fsbuu2'])/3 + (8*Fsbuu['Fsbuu4'])/3,
 '3sbuu' : Fsbuu['Fsbuu1']/12 - Fsbuu['Fsbuu3']/12 + Fsbuu['Fsbuu2']/(12*Nc) - Fsbuu['Fsbuu4']/(12*Nc),
 '4sbuu' : Fsbuu['Fsbuu2']/6 - Fsbuu['Fsbuu4']/6,
 '5sbuu' : -Fsbuu['Fsbuu5']/3 + (4*Fsbuu['Fsbuu7'])/3 - Fsbuu['Fsbuu6']/(3*Nc) + (4*Fsbuu['Fsbuu8'])/(3*Nc),
 '6sbuu' : (-2*Fsbuu['Fsbuu6'])/3 + (8*Fsbuu['Fsbuu8'])/3,
 '7sbuu' : Fsbuu['Fsbuu5']/3 - Fsbuu['Fsbuu7']/3 + Fsbuu['Fsbuu9'] + Fsbuu['Fsbuu10']/Nc + Fsbuu['Fsbuu6']/(3*Nc) - Fsbuu['Fsbuu8']/(3*Nc),
 '8sbuu' : 2*Fsbuu['Fsbuu10'] + (2*Fsbuu['Fsbuu6'])/3 - (2*Fsbuu['Fsbuu8'])/3,
 '9sbuu' : Fsbuu['Fsbuu5']/48 - Fsbuu['Fsbuu7']/48 + Fsbuu['Fsbuu6']/(48*Nc) - Fsbuu['Fsbuu8']/(48*Nc),
 '10sbuu' : Fsbuu['Fsbuu6']/24 - Fsbuu['Fsbuu8']/24}

def Bernpsbuu (Fpsbuu):
    return {'1psbuu' : -Fpsbuu['Fsbuu1p']/3 + (4*Fpsbuu['Fsbuu3p'])/3 - Fpsbuu['Fsbuu2p']/(3*Nc) + (4*Fpsbuu['Fsbuu4p'])/(3*Nc),
 '2psbuu' : (-2*Fpsbuu['Fsbuu2p'])/3 + (8*Fpsbuu['Fsbuu4p'])/3,
 '3psbuu' : Fpsbuu['Fsbuu1p']/12 - Fpsbuu['Fsbuu3p']/12 + Fpsbuu['Fsbuu2p']/(12*Nc) - Fpsbuu['Fsbuu4p']/(12*Nc),
 '4psbuu' : Fpsbuu['Fsbuu2p']/6 - Fpsbuu['Fsbuu4p']/6,
 '5psbuu' : -Fpsbuu['Fsbuu5p']/3 + (4*Fpsbuu['Fsbuu7p'])/3 - Fpsbuu['Fsbuu6p']/(3*Nc) + (4*Fpsbuu['Fsbuu8p'])/(3*Nc),
 '6psbuu' : (-2*Fpsbuu['Fsbuu6p'])/3 + (8*Fpsbuu['Fsbuu8p'])/3,
 '7psbuu' : Fpsbuu['Fsbuu5p']/3 - Fpsbuu['Fsbuu7p']/3 + Fpsbuu['Fsbuu9p'] + Fpsbuu['Fsbuu10p']/Nc + Fpsbuu['Fsbuu6p']/(3*Nc) - Fpsbuu['Fsbuu8p']/(3*Nc),
 '8psbuu' : 2*Fpsbuu['Fsbuu10p'] + (2*Fpsbuu['Fsbuu6p'])/3 - (2*Fpsbuu['Fsbuu8p'])/3,
 '9psbuu' : Fpsbuu['Fsbuu5p']/48 - Fpsbuu['Fsbuu7p']/48 + Fpsbuu['Fsbuu6p']/(48*Nc) - Fpsbuu['Fsbuu8p']/(48*Nc),
 '10psbuu' : Fpsbuu['Fsbuu6p']/24 - Fpsbuu['Fsbuu8p']/24}

 # dbuu

def Berndbuu (Fdbuu):
    return {'1dbuu' : -Fdbuu['Fdbuu1']/3 + (4*Fdbuu['Fdbuu3'])/3 - Fdbuu['Fdbuu2']/(3*Nc) + (4*Fdbuu['Fdbuu4'])/(3*Nc),
 '2dbuu' : (-2*Fdbuu['Fdbuu2'])/3 + (8*Fdbuu['Fdbuu4'])/3,
 '3dbuu' : Fdbuu['Fdbuu1']/12 - Fdbuu['Fdbuu3']/12 + Fdbuu['Fdbuu2']/(12*Nc) - Fdbuu['Fdbuu4']/(12*Nc),
 '4dbuu' : Fdbuu['Fdbuu2']/6 - Fdbuu['Fdbuu4']/6,
 '5dbuu' : -Fdbuu['Fdbuu5']/3 + (4*Fdbuu['Fdbuu7'])/3 - Fdbuu['Fdbuu6']/(3*Nc) + (4*Fdbuu['Fdbuu8'])/(3*Nc),
 '6dbuu' : (-2*Fdbuu['Fdbuu6'])/3 + (8*Fdbuu['Fdbuu8'])/3,
 '7dbuu' : Fdbuu['Fdbuu5']/3 - Fdbuu['Fdbuu7']/3 + Fdbuu['Fdbuu9'] + Fdbuu['Fdbuu10']/Nc + Fdbuu['Fdbuu6']/(3*Nc) - Fdbuu['Fdbuu8']/(3*Nc),
 '8dbuu' : 2*Fdbuu['Fdbuu10'] + (2*Fdbuu['Fdbuu6'])/3 - (2*Fdbuu['Fdbuu8'])/3,
 '9dbuu' : Fdbuu['Fdbuu5']/48 - Fdbuu['Fdbuu7']/48 + Fdbuu['Fdbuu6']/(48*Nc) - Fdbuu['Fdbuu8']/(48*Nc),
 '10dbuu' : Fdbuu['Fdbuu6']/24 - Fdbuu['Fdbuu8']/24}

def Bernpdbuu (Fpdbuu):
    return {'1pdbuu' : -Fpdbuu['Fdbuu1p']/3 + (4*Fpdbuu['Fdbuu3p'])/3 - Fpdbuu['Fdbuu2p']/(3*Nc) + (4*Fpdbuu['Fdbuu4p'])/(3*Nc),
 '2pdbuu' : (-2*Fpdbuu['Fdbuu2p'])/3 + (8*Fpdbuu['Fdbuu4p'])/3,
 '3pdbuu' : Fpdbuu['Fdbuu1p']/12 - Fpdbuu['Fdbuu3p']/12 + Fpdbuu['Fdbuu2p']/(12*Nc) - Fpdbuu['Fdbuu4p']/(12*Nc),
 '4pdbuu' : Fpdbuu['Fdbuu2p']/6 - Fpdbuu['Fdbuu4p']/6,
 '5pdbuu' : -Fpdbuu['Fdbuu5p']/3 + (4*Fpdbuu['Fdbuu7p'])/3 - Fpdbuu['Fdbuu6p']/(3*Nc) + (4*Fpdbuu['Fdbuu8p'])/(3*Nc),
 '6pdbuu' : (-2*Fpdbuu['Fdbuu6p'])/3 + (8*Fpdbuu['Fdbuu8p'])/3,
 '7pdbuu' : Fpdbuu['Fdbuu5p']/3 - Fpdbuu['Fdbuu7p']/3 + Fpdbuu['Fdbuu9p'] + Fpdbuu['Fdbuu10p']/Nc + Fpdbuu['Fdbuu6p']/(3*Nc) - Fpdbuu['Fdbuu8p']/(3*Nc),
 '8pdbuu' : 2*Fpdbuu['Fdbuu10p'] + (2*Fpdbuu['Fdbuu6p'])/3 - (2*Fpdbuu['Fdbuu8p'])/3,
 '9pdbuu' : Fpdbuu['Fdbuu5p']/48 - Fpdbuu['Fdbuu7p']/48 + Fpdbuu['Fdbuu6p']/(48*Nc) - Fpdbuu['Fdbuu8p']/(48*Nc),
 '10pdbuu' : Fpdbuu['Fdbuu6p']/24 - Fpdbuu['Fdbuu8p']/24}


# sbcc

def Bernsbcc (Fsbcc):
    return {'1sbcc' : -Fsbcc['Fsbcc1']/3 + (4*Fsbcc['Fsbcc3'])/3 - Fsbcc['Fsbcc2']/(3*Nc) + (4*Fsbcc['Fsbcc4'])/(3*Nc),
 '2sbcc' : (-2*Fsbcc['Fsbcc2'])/3 + (8*Fsbcc['Fsbcc4'])/3,
 '3sbcc' : Fsbcc['Fsbcc1']/12 - Fsbcc['Fsbcc3']/12 + Fsbcc['Fsbcc2']/(12*Nc) - Fsbcc['Fsbcc4']/(12*Nc),
 '4sbcc' : Fsbcc['Fsbcc2']/6 - Fsbcc['Fsbcc4']/6,
 '5sbcc' : -Fsbcc['Fsbcc5']/3 + (4*Fsbcc['Fsbcc7'])/3 - Fsbcc['Fsbcc6']/(3*Nc) + (4*Fsbcc['Fsbcc8'])/(3*Nc),
 '6sbcc' : (-2*Fsbcc['Fsbcc6'])/3 + (8*Fsbcc['Fsbcc8'])/3,
 '7sbcc' : Fsbcc['Fsbcc5']/3 - Fsbcc['Fsbcc7']/3 + Fsbcc['Fsbcc9'] + Fsbcc['Fsbcc10']/Nc + Fsbcc['Fsbcc6']/(3*Nc) - Fsbcc['Fsbcc8']/(3*Nc),
 '8sbcc' : 2*Fsbcc['Fsbcc10'] + (2*Fsbcc['Fsbcc6'])/3 - (2*Fsbcc['Fsbcc8'])/3,
 '9sbcc' : Fsbcc['Fsbcc5']/48 - Fsbcc['Fsbcc7']/48 + Fsbcc['Fsbcc6']/(48*Nc) - Fsbcc['Fsbcc8']/(48*Nc),
 '10sbcc' : Fsbcc['Fsbcc6']/24 - Fsbcc['Fsbcc8']/24}

def Bernpsbcc (Fpsbcc):
    return {'1psbcc' : -Fpsbcc['Fsbcc1p']/3 + (4*Fpsbcc['Fsbcc3p'])/3 - Fpsbcc['Fsbcc2p']/(3*Nc) + (4*Fpsbcc['Fsbcc4p'])/(3*Nc),
 '2psbcc' : (-2*Fpsbcc['Fsbcc2p'])/3 + (8*Fpsbcc['Fsbcc4p'])/3,
 '3psbcc' : Fpsbcc['Fsbcc1p']/12 - Fpsbcc['Fsbcc3p']/12 + Fpsbcc['Fsbcc2p']/(12*Nc) - Fpsbcc['Fsbcc4p']/(12*Nc),
 '4psbcc' : Fpsbcc['Fsbcc2p']/6 - Fpsbcc['Fsbcc4p']/6,
 '5psbcc' : -Fpsbcc['Fsbcc5p']/3 + (4*Fpsbcc['Fsbcc7p'])/3 - Fpsbcc['Fsbcc6p']/(3*Nc) + (4*Fpsbcc['Fsbcc8p'])/(3*Nc),
 '6psbcc' : (-2*Fpsbcc['Fsbcc6p'])/3 + (8*Fpsbcc['Fsbcc8p'])/3,
 '7psbcc' : Fpsbcc['Fsbcc5p']/3 - Fpsbcc['Fsbcc7p']/3 + Fpsbcc['Fsbcc9p'] + Fpsbcc['Fsbcc10p']/Nc + Fpsbcc['Fsbcc6p']/(3*Nc) - Fpsbcc['Fsbcc8p']/(3*Nc),
 '8psbcc' : 2*Fpsbcc['Fsbcc10p'] + (2*Fpsbcc['Fsbcc6p'])/3 - (2*Fpsbcc['Fsbcc8p'])/3,
 '9psbcc' : Fpsbcc['Fsbcc5p']/48 - Fpsbcc['Fsbcc7p']/48 + Fpsbcc['Fsbcc6p']/(48*Nc) - Fpsbcc['Fsbcc8p']/(48*Nc),
 '10psbcc' : Fpsbcc['Fsbcc6p']/24 - Fpsbcc['Fsbcc8p']/24}

# dbcc

def Berndbcc (Fdbcc):
    return {'1dbcc' : -Fdbcc['Fdbcc1']/3 + (4*Fdbcc['Fdbcc3'])/3 - Fdbcc['Fdbcc2']/(3*Nc) + (4*Fdbcc['Fdbcc4'])/(3*Nc),
 '2dbcc' : (-2*Fdbcc['Fdbcc2'])/3 + (8*Fdbcc['Fdbcc4'])/3,
 '3dbcc' : Fdbcc['Fdbcc1']/12 - Fdbcc['Fdbcc3']/12 + Fdbcc['Fdbcc2']/(12*Nc) - Fdbcc['Fdbcc4']/(12*Nc),
 '4dbcc' : Fdbcc['Fdbcc2']/6 - Fdbcc['Fdbcc4']/6,
 '5dbcc' : -Fdbcc['Fdbcc5']/3 + (4*Fdbcc['Fdbcc7'])/3 - Fdbcc['Fdbcc6']/(3*Nc) + (4*Fdbcc['Fdbcc8'])/(3*Nc),
 '6dbcc' : (-2*Fdbcc['Fdbcc6'])/3 + (8*Fdbcc['Fdbcc8'])/3,
 '7dbcc' : Fdbcc['Fdbcc5']/3 - Fdbcc['Fdbcc7']/3 + Fdbcc['Fdbcc9'] + Fdbcc['Fdbcc10']/Nc + Fdbcc['Fdbcc6']/(3*Nc) - Fdbcc['Fdbcc8']/(3*Nc),
 '8dbcc' : 2*Fdbcc['Fdbcc10'] + (2*Fdbcc['Fdbcc6'])/3 - (2*Fdbcc['Fdbcc8'])/3,
 '9dbcc' : Fdbcc['Fdbcc5']/48 - Fdbcc['Fdbcc7']/48 + Fdbcc['Fdbcc6']/(48*Nc) - Fdbcc['Fdbcc8']/(48*Nc),
 '10dbcc' : Fdbcc['Fdbcc6']/24 - Fdbcc['Fdbcc8']/24}


def Bernpdbcc (Fpdbcc):
    return {'1pdbcc' : -Fpdbcc['Fdbcc1p']/3 + (4*Fpdbcc['Fdbcc3p'])/3 - Fpdbcc['Fdbcc2p']/(3*Nc) + (4*Fpdbcc['Fdbcc4p'])/(3*Nc),
 '2pdbcc' : (-2*Fpdbcc['Fdbcc2p'])/3 + (8*Fpdbcc['Fdbcc4p'])/3,
 '3pdbcc' : Fpdbcc['Fdbcc1p']/12 - Fpdbcc['Fdbcc3p']/12 + Fpdbcc['Fdbcc2p']/(12*Nc) - Fpdbcc['Fdbcc4p']/(12*Nc),
 '4pdbcc' : Fpdbcc['Fdbcc2p']/6 - Fpdbcc['Fdbcc4p']/6,
 '5pdbcc' : -Fpdbcc['Fdbcc5p']/3 + (4*Fpdbcc['Fdbcc7p'])/3 - Fpdbcc['Fdbcc6p']/(3*Nc) + (4*Fpdbcc['Fdbcc8p'])/(3*Nc),
 '6pdbcc' : (-2*Fpdbcc['Fdbcc6p'])/3 + (8*Fpdbcc['Fdbcc8p'])/3,
 '7pdbcc' : Fpdbcc['Fdbcc5p']/3 - Fpdbcc['Fdbcc7p']/3 + Fpdbcc['Fdbcc9p'] + Fpdbcc['Fdbcc10p']/Nc + Fpdbcc['Fdbcc6p']/(3*Nc) - Fpdbcc['Fdbcc8p']/(3*Nc),
 '8pdbcc' : 2*Fpdbcc['Fdbcc10p'] + (2*Fpdbcc['Fdbcc6p'])/3 - (2*Fpdbcc['Fdbcc8p'])/3,
 '9pdbcc' : Fpdbcc['Fdbcc5p']/48 - Fpdbcc['Fdbcc7p']/48 + Fpdbcc['Fdbcc6p']/(48*Nc) - Fpdbcc['Fdbcc8p']/(48*Nc),
 '10pdbcc' : Fpdbcc['Fdbcc6p']/24 - Fpdbcc['Fdbcc8p']/24}


# sbdd

def Bernsbdd (Fsbdd):
    return {'1sbdd' : -Fsbdd['Fsbdd1']/3 + (4*Fsbdd['Fsbdd3'])/3 - Fsbdd['Fsbdd2']/(3*Nc) + (4*Fsbdd['Fsbdd4'])/(3*Nc),
 '2sbdd' : (-2*Fsbdd['Fsbdd2'])/3 + (8*Fsbdd['Fsbdd4'])/3,
 '3sbdd' : Fsbdd['Fsbdd1']/12 - Fsbdd['Fsbdd3']/12 + Fsbdd['Fsbdd2']/(12*Nc) - Fsbdd['Fsbdd4']/(12*Nc),
 '4sbdd' : Fsbdd['Fsbdd2']/6 - Fsbdd['Fsbdd4']/6,
 '5sbdd' : -Fsbdd['Fsbdd5']/3 + (4*Fsbdd['Fsbdd7'])/3 - Fsbdd['Fsbdd6']/(3*Nc) + (4*Fsbdd['Fsbdd8'])/(3*Nc),
 '6sbdd' : (-2*Fsbdd['Fsbdd6'])/3 + (8*Fsbdd['Fsbdd8'])/3,
 '7sbdd' : Fsbdd['Fsbdd5']/3 - Fsbdd['Fsbdd7']/3 + Fsbdd['Fsbdd9'] + Fsbdd['Fsbdd10']/Nc + Fsbdd['Fsbdd6']/(3*Nc) - Fsbdd['Fsbdd8']/(3*Nc),
 '8sbdd' : 2*Fsbdd['Fsbdd10'] + (2*Fsbdd['Fsbdd6'])/3 - (2*Fsbdd['Fsbdd8'])/3,
 '9sbdd' : Fsbdd['Fsbdd5']/48 - Fsbdd['Fsbdd7']/48 + Fsbdd['Fsbdd6']/(48*Nc) - Fsbdd['Fsbdd8']/(48*Nc),
 '10sbdd' : Fsbdd['Fsbdd6']/24 - Fsbdd['Fsbdd8']/24}


def Bernpsbdd (Fpsbdd):
    return {'1psbdd' : -Fpsbdd['Fsbdd1p']/3 + (4*Fpsbdd['Fsbdd3p'])/3 - Fpsbdd['Fsbdd2p']/(3*Nc) + (4*Fpsbdd['Fsbdd4p'])/(3*Nc),
 '2psbdd' : (-2*Fpsbdd['Fsbdd2p'])/3 + (8*Fpsbdd['Fsbdd4p'])/3,
 '3psbdd' : Fpsbdd['Fsbdd1p']/12 - Fpsbdd['Fsbdd3p']/12 + Fpsbdd['Fsbdd2p']/(12*Nc) - Fpsbdd['Fsbdd4p']/(12*Nc),
 '4psbdd' : Fpsbdd['Fsbdd2p']/6 - Fpsbdd['Fsbdd4p']/6,
 '5psbdd' : -Fpsbdd['Fsbdd5p']/3 + (4*Fpsbdd['Fsbdd7p'])/3 - Fpsbdd['Fsbdd6p']/(3*Nc) + (4*Fpsbdd['Fsbdd8p'])/(3*Nc),
 '6psbdd' : (-2*Fpsbdd['Fsbdd6p'])/3 + (8*Fpsbdd['Fsbdd8p'])/3,
 '7psbdd' : Fpsbdd['Fsbdd5p']/3 - Fpsbdd['Fsbdd7p']/3 + Fpsbdd['Fsbdd9p'] + Fpsbdd['Fsbdd10p']/Nc + Fpsbdd['Fsbdd6p']/(3*Nc) - Fpsbdd['Fsbdd8p']/(3*Nc),
 '8psbdd' : 2*Fpsbdd['Fsbdd10p'] + (2*Fpsbdd['Fsbdd6p'])/3 - (2*Fpsbdd['Fsbdd8p'])/3,
 '9psbdd' : Fpsbdd['Fsbdd5p']/48 - Fpsbdd['Fsbdd7p']/48 + Fpsbdd['Fsbdd6p']/(48*Nc) - Fpsbdd['Fsbdd8p']/(48*Nc),
 '10psbdd' : Fpsbdd['Fsbdd6p']/24 - Fpsbdd['Fsbdd8p']/24}


# dbdd

def Berndbdd (Fdbdd):
    return {'1dbdd' : -Fdbdd['Fdbdd1']/3 + (4*Fdbdd['Fdbdd3'])/3 - Fdbdd['Fdbdd2']/(3*Nc) + (4*Fdbdd['Fdbdd4'])/(3*Nc),
 '2dbdd' : (-2*Fdbdd['Fdbdd2'])/3 + (8*Fdbdd['Fdbdd4'])/3,
 '3dbdd' : Fdbdd['Fdbdd1']/12 - Fdbdd['Fdbdd3']/12 + Fdbdd['Fdbdd2']/(12*Nc) - Fdbdd['Fdbdd4']/(12*Nc),
 '4dbdd' : Fdbdd['Fdbdd2']/6 - Fdbdd['Fdbdd4']/6,
 '5dbdd' : -Fdbdd['Fdbdd5']/3 + (4*Fdbdd['Fdbdd7'])/3 - Fdbdd['Fdbdd6']/(3*Nc) + (4*Fdbdd['Fdbdd8'])/(3*Nc),
 '6dbdd' : (-2*Fdbdd['Fdbdd6'])/3 + (8*Fdbdd['Fdbdd8'])/3,
 '7dbdd' : Fdbdd['Fdbdd5']/3 - Fdbdd['Fdbdd7']/3 + Fdbdd['Fdbdd9'] + Fdbdd['Fdbdd10']/Nc + Fdbdd['Fdbdd6']/(3*Nc) - Fdbdd['Fdbdd8']/(3*Nc),
 '8dbdd' : 2*Fdbdd['Fdbdd10'] + (2*Fdbdd['Fdbdd6'])/3 - (2*Fdbdd['Fdbdd8'])/3,
 '9dbdd' : Fdbdd['Fdbdd5']/48 - Fdbdd['Fdbdd7']/48 + Fdbdd['Fdbdd6']/(48*Nc) - Fdbdd['Fdbdd8']/(48*Nc),
 '10dbdd' : Fdbdd['Fdbdd6']/24 - Fdbdd['Fdbdd8']/24}

def Bernpdbdd (Fpdbdd):
    return {'1pdbdd' : -Fpdbdd['Fdbdd1p']/3 + (4*Fpdbdd['Fdbdd3p'])/3 - Fpdbdd['Fdbdd2p']/(3*Nc) + (4*Fpdbdd['Fdbdd4p'])/(3*Nc),
 '2pdbdd' : (-2*Fpdbdd['Fdbdd2p'])/3 + (8*Fpdbdd['Fdbdd4p'])/3,
 '3pdbdd' : Fpdbdd['Fdbdd1p']/12 - Fpdbdd['Fdbdd3p']/12 + Fpdbdd['Fdbdd2p']/(12*Nc) - Fpdbdd['Fdbdd4p']/(12*Nc),
 '4pdbdd' : Fpdbdd['Fdbdd2p']/6 - Fpdbdd['Fdbdd4p']/6,
 '5pdbdd' : -Fpdbdd['Fdbdd5p']/3 + (4*Fpdbdd['Fdbdd7p'])/3 - Fpdbdd['Fdbdd6p']/(3*Nc) + (4*Fpdbdd['Fdbdd8p'])/(3*Nc),
 '6pdbdd' : (-2*Fpdbdd['Fdbdd6p'])/3 + (8*Fpdbdd['Fdbdd8p'])/3,
 '7pdbdd' : Fpdbdd['Fdbdd5p']/3 - Fpdbdd['Fdbdd7p']/3 + Fpdbdd['Fdbdd9p'] + Fpdbdd['Fdbdd10p']/Nc + Fpdbdd['Fdbdd6p']/(3*Nc) - Fpdbdd['Fdbdd8p']/(3*Nc),
 '8pdbdd' : 2*Fpdbdd['Fdbdd10p'] + (2*Fpdbdd['Fdbdd6p'])/3 - (2*Fpdbdd['Fdbdd8p'])/3,
 '9pdbdd' : Fpdbdd['Fdbdd5p']/48 - Fpdbdd['Fdbdd7p']/48 + Fpdbdd['Fdbdd6p']/(48*Nc) - Fpdbdd['Fdbdd8p']/(48*Nc),
 '10pdbdd' : Fpdbdd['Fdbdd6p']/24 - Fpdbdd['Fdbdd8p']/24}


# sbss

def Bernsbss (Fsbss):
    return {'1sbss' : -Fsbss['Fsbss1']/3 + (4*Fsbss['Fsbss3'])/3 - Fsbss['Fsbss2']/(3*Nc) + (4*Fsbss['Fsbss4'])/(3*Nc),
 '3sbss' : Fsbss['Fsbss1']/12 - Fsbss['Fsbss3']/12 + Fsbss['Fsbss2']/(12*Nc) - Fsbss['Fsbss4']/(12*Nc),
 '5sbss' : -Fsbss['Fsbss5']/3 + (4*Fsbss['Fsbss7'])/3 - Fsbss['Fsbss6']/(3*Nc) + (4*Fsbss['Fsbss8'])/(3*Nc),
 '7sbss' : Fsbss['Fsbss5']/3 - Fsbss['Fsbss7']/3 + Fsbss['Fsbss9'] + Fsbss['Fsbss10']/Nc + Fsbss['Fsbss6']/(3*Nc) - Fsbss['Fsbss8']/(3*Nc),
 '9sbss' : Fsbss['Fsbss5']/48 - Fsbss['Fsbss7']/48 + Fsbss['Fsbss6']/(48*Nc) - Fsbss['Fsbss8']/(48*Nc)}


def Bernpsbss (Fpsbss):
    return {'1psbss' : -Fpsbss['Fsbss1p']/3 + (4*Fpsbss['Fsbss3p'])/3 - Fpsbss['Fsbss2p']/(3*Nc) + (4*Fpsbss['Fsbss4p'])/(3*Nc),
 '3psbss' : Fpsbss['Fsbss1p']/12 - Fpsbss['Fsbss3p']/12 + Fpsbss['Fsbss2p']/(12*Nc) - Fpsbss['Fsbss4p']/(12*Nc),
 '5psbss' : -Fpsbss['Fsbss5p']/3 + (4*Fpsbss['Fsbss7p'])/3 - Fpsbss['Fsbss6p']/(3*Nc) + (4*Fpsbss['Fsbss8p'])/(3*Nc),
 '7psbss' : Fpsbss['Fsbss5p']/3 - Fpsbss['Fsbss7p']/3 + Fpsbss['Fsbss9p'] + Fpsbss['Fsbss10p']/Nc + Fpsbss['Fsbss6p']/(3*Nc) - Fpsbss['Fsbss8p']/(3*Nc),
 '9psbss' : Fpsbss['Fsbss5p']/48 - Fpsbss['Fsbss7p']/48 + Fpsbss['Fsbss6p']/(48*Nc) - Fpsbss['Fsbss8p']/(48*Nc)}

 # dbss

def Berndbss (Fdbss):
    return {'1dbss' : -Fdbss['Fdbss1']/3 + (4*Fdbss['Fdbss3'])/3 - Fdbss['Fdbss2']/(3*Nc) + (4*Fdbss['Fdbss4'])/(3*Nc),
 '2dbss' : (-2*Fdbss['Fdbss2'])/3 + (8*Fdbss['Fdbss4'])/3,
 '3dbss' : Fdbss['Fdbss1']/12 - Fdbss['Fdbss3']/12 + Fdbss['Fdbss2']/(12*Nc) - Fdbss['Fdbss4']/(12*Nc),
 '4dbss' : Fdbss['Fdbss2']/6 - Fdbss['Fdbss4']/6,
 '5dbss' : -Fdbss['Fdbss5']/3 + (4*Fdbss['Fdbss7'])/3 - Fdbss['Fdbss6']/(3*Nc) + (4*Fdbss['Fdbss8'])/(3*Nc),
 '6dbss' : (-2*Fdbss['Fdbss6'])/3 + (8*Fdbss['Fdbss8'])/3,
 '7dbss' : Fdbss['Fdbss5']/3 - Fdbss['Fdbss7']/3 + Fdbss['Fdbss9'] + Fdbss['Fdbss10']/Nc + Fdbss['Fdbss6']/(3*Nc) - Fdbss['Fdbss8']/(3*Nc),
 '8dbss' : 2*Fdbss['Fdbss10'] + (2*Fdbss['Fdbss6'])/3 - (2*Fdbss['Fdbss8'])/3,
 '9dbss' : Fdbss['Fdbss5']/48 - Fdbss['Fdbss7']/48 + Fdbss['Fdbss6']/(48*Nc) - Fdbss['Fdbss8']/(48*Nc),
 '10dbss' : Fdbss['Fdbss6']/24 - Fdbss['Fdbss8']/24}


def Bernpdbss (Fpdbss):
    return {'1pdbss' : -Fpdbss['Fdbss1p']/3 + (4*Fpdbss['Fdbss3p'])/3 - Fpdbss['Fdbss2p']/(3*Nc) + (4*Fpdbss['Fdbss4p'])/(3*Nc),
 '2pdbss' : (-2*Fpdbss['Fdbss2p'])/3 + (8*Fpdbss['Fdbss4p'])/3,
 '3pdbss' : Fpdbss['Fdbss1p']/12 - Fpdbss['Fdbss3p']/12 + Fpdbss['Fdbss2p']/(12*Nc) - Fpdbss['Fdbss4p']/(12*Nc),
 '4pdbss' : Fpdbss['Fdbss2p']/6 - Fpdbss['Fdbss4p']/6,
 '5pdbss' : -Fpdbss['Fdbss5p']/3 + (4*Fpdbss['Fdbss7p'])/3 - Fpdbss['Fdbss6p']/(3*Nc) + (4*Fpdbss['Fdbss8p'])/(3*Nc),
 '6pdbss' : (-2*Fpdbss['Fdbss6p'])/3 + (8*Fpdbss['Fdbss8p'])/3,
 '7pdbss' : Fpdbss['Fdbss5p']/3 - Fpdbss['Fdbss7p']/3 + Fpdbss['Fdbss9p'] + Fpdbss['Fdbss10p']/Nc + Fpdbss['Fdbss6p']/(3*Nc) - Fpdbss['Fdbss8p']/(3*Nc),
 '8pdbss' : 2*Fpdbss['Fdbss10p'] + (2*Fpdbss['Fdbss6p'])/3 - (2*Fpdbss['Fdbss8p'])/3,
 '9pdbss' : Fpdbss['Fdbss5p']/48 - Fpdbss['Fdbss7p']/48 + Fpdbss['Fdbss6p']/(48*Nc) - Fpdbss['Fdbss8p']/(48*Nc),
 '10pdbss' : Fpdbss['Fdbss6p']/24 - Fpdbss['Fdbss8p']/24}


 # sbbb

def Bernsbbb (Fsbbb):
    return {'1sbbb' : -Fsbbb['Fsbbb1']/3 + (4*Fsbbb['Fsbbb3'])/3 - Fsbbb['Fsbbb2']/(3*Nc) + (4*Fsbbb['Fsbbb4'])/(3*Nc),
 '3sbbb' : Fsbbb['Fsbbb1']/12 - Fsbbb['Fsbbb3']/12 + Fsbbb['Fsbbb2']/(12*Nc) - Fsbbb['Fsbbb4']/(12*Nc),
 '5sbbb' : -Fsbbb['Fsbbb5']/3 + (4*Fsbbb['Fsbbb7'])/3 - Fsbbb['Fsbbb6']/(3*Nc) + (4*Fsbbb['Fsbbb8'])/(3*Nc),
 '7sbbb' : Fsbbb['Fsbbb5']/3 - Fsbbb['Fsbbb7']/3 + Fsbbb['Fsbbb9'] + Fsbbb['Fsbbb10']/Nc + Fsbbb['Fsbbb6']/(3*Nc) - Fsbbb['Fsbbb8']/(3*Nc),
 '9sbbb' : Fsbbb['Fsbbb5']/48 - Fsbbb['Fsbbb7']/48 + Fsbbb['Fsbbb6']/(48*Nc) - Fsbbb['Fsbbb8']/(48*Nc)}

def Bernpsbbb (Fpsbbb):
    return {'1psbbb' : -Fpsbbb['Fsbbb1p']/3 + (4*Fpsbbb['Fsbbb3p'])/3 - Fpsbbb['Fsbbb2p']/(3*Nc) + (4*Fpsbbb['Fsbbb4p'])/(3*Nc),
 '3psbbb' : Fpsbbb['Fsbbb1p']/12 - Fpsbbb['Fsbbb3p']/12 + Fpsbbb['Fsbbb2p']/(12*Nc) - Fpsbbb['Fsbbb4p']/(12*Nc),
 '5psbbb' : -Fpsbbb['Fsbbb5p']/3 + (4*Fpsbbb['Fsbbb7p'])/3 - Fpsbbb['Fsbbb6p']/(3*Nc) + (4*Fpsbbb['Fsbbb8p'])/(3*Nc),
 '7psbbb' : Fpsbbb['Fsbbb5p']/3 - Fpsbbb['Fsbbb7p']/3 + Fpsbbb['Fsbbb9p'] + Fpsbbb['Fsbbb10p']/Nc + Fpsbbb['Fsbbb6p']/(3*Nc) - Fpsbbb['Fsbbb8p']/(3*Nc),
 '9psbbb' : Fpsbbb['Fsbbb5p']/48 - Fpsbbb['Fsbbb7p']/48 + Fpsbbb['Fsbbb6p']/(48*Nc) - Fpsbbb['Fsbbb8p']/(48*Nc)}

# dbbb

def Berndbbb (Fdbbb):
    return {'1dbbb' : -Fdbbb['Fdbbb1']/3 + (4*Fdbbb['Fdbbb3'])/3 - Fdbbb['Fdbbb2']/(3*Nc) + (4*Fdbbb['Fdbbb4'])/(3*Nc),
'3dbbb' : Fdbbb['Fdbbb1']/12 - Fdbbb['Fdbbb3']/12 + Fdbbb['Fdbbb2']/(12*Nc) - Fdbbb['Fdbbb4']/(12*Nc),
'5dbbb' : -Fdbbb['Fdbbb5']/3 + (4*Fdbbb['Fdbbb7'])/3 - Fdbbb['Fdbbb6']/(3*Nc) + (4*Fdbbb['Fdbbb8'])/(3*Nc),
'7dbbb' : Fdbbb['Fdbbb5']/3 - Fdbbb['Fdbbb7']/3 + Fdbbb['Fdbbb9'] + Fdbbb['Fdbbb10']/Nc + Fdbbb['Fdbbb6']/(3*Nc) - Fdbbb['Fdbbb8']/(3*Nc),
'9dbbb' : Fdbbb['Fdbbb5']/48 - Fdbbb['Fdbbb7']/48 + Fdbbb['Fdbbb6']/(48*Nc) - Fdbbb['Fdbbb8']/(48*Nc)}

def Bernpdbbb (Fpdbbb):
    return {'1pdbbb' : -Fpdbbb['Fdbbb1p']/3 + (4*Fpdbbb['Fdbbb3p'])/3 - Fpdbbb['Fdbbb2p']/(3*Nc) + (4*Fpdbbb['Fdbbb4p'])/(3*Nc),
 '3pdbbb' : Fpdbbb['Fdbbb1p']/12 - Fpdbbb['Fdbbb3p']/12 + Fpdbbb['Fdbbb2p']/(12*Nc) - Fpdbbb['Fdbbb4p']/(12*Nc),
 '5pdbbb' : -Fpdbbb['Fdbbb5p']/3 + (4*Fpdbbb['Fdbbb7p'])/3 - Fpdbbb['Fdbbb6p']/(3*Nc) + (4*Fpdbbb['Fdbbb8p'])/(3*Nc),
 '7pdbbb' : Fpdbbb['Fdbbb5p']/3 - Fpdbbb['Fdbbb7p']/3 + Fpdbbb['Fdbbb9p'] + Fpdbbb['Fdbbb10p']/Nc + Fpdbbb['Fdbbb6p']/(3*Nc) - Fpdbbb['Fdbbb8p']/(3*Nc),
 '9pdbbb' : Fpdbbb['Fdbbb5p']/48 - Fpdbbb['Fdbbb7p']/48 + Fpdbbb['Fdbbb6p']/(48*Nc) - Fpdbbb['Fdbbb8p']/(48*Nc)}



# Buras basis

#sbqq

def Burassbqq (Fsbuu,Fsbdd,Fsbcc,Fsbss,Fsbbb):
    return {'Burassbqq1' : 2*Fsbcc['Fsbcc1'] - 2*Fsbuu['Fsbuu1'],
 'Burassbqq2' : Fsbcc['Fsbcc1']/3 + Fsbcc['Fsbcc2'] - Fsbuu['Fsbuu1']/3 - Fsbuu['Fsbuu2'],
 'Burassbqq3' : (-2*Fsbbb['Fsbbb1'])/27 - (2*Fsbbb['Fsbbb2'])/81 + (8*Fsbbb['Fsbbb3'])/27 + (8*Fsbbb['Fsbbb4'])/81 + (2*Fsbcc['Fsbcc3'])/9 + (2*Fsbcc['Fsbcc4'])/27 - (2*Fsbdd['Fsbdd1'])/27 - (2*Fsbdd['Fsbdd2'])/81 + (8*Fsbdd['Fsbdd3'])/27 + (8*Fsbdd['Fsbdd4'])/81 - (2*Fsbss['Fsbss1'])/27 - (2*Fsbss['Fsbss2'])/81 + (8*Fsbss['Fsbss3'])/27 + (8*Fsbss['Fsbss4'])/81 - Fsbuu['Fsbuu1']/9 - Fsbuu['Fsbuu2']/27 + (2*Fsbuu['Fsbuu3'])/9 + (2*Fsbuu['Fsbuu4'])/27,
 'Burassbqq4' : (-4*Fsbbb['Fsbbb2'])/27 + (16*Fsbbb['Fsbbb4'])/27 + (4*Fsbcc['Fsbcc4'])/9 - (4*Fsbdd['Fsbdd2'])/27 + (16*Fsbdd['Fsbdd4'])/27 - (4*Fsbss['Fsbss2'])/27 + (16*Fsbss['Fsbss4'])/27 - (2*Fsbuu['Fsbuu2'])/9 + (4*Fsbuu['Fsbuu4'])/9,
 'Burassbqq5' : Fsbbb['Fsbbb1']/54 + Fsbbb['Fsbbb2']/162 - Fsbbb['Fsbbb3']/54 - Fsbbb['Fsbbb4']/162 - Fsbcc['Fsbcc3']/72 - Fsbcc['Fsbcc4']/216 + Fsbdd['Fsbdd1']/54 + Fsbdd['Fsbdd2']/162 - Fsbdd['Fsbdd3']/54 - Fsbdd['Fsbdd4']/162 + Fsbss['Fsbss1']/54 + Fsbss['Fsbss2']/162 - Fsbss['Fsbss3']/54 - Fsbss['Fsbss4']/162 + Fsbuu['Fsbuu1']/36 + Fsbuu['Fsbuu2']/108 - Fsbuu['Fsbuu3']/72 - Fsbuu['Fsbuu4']/216,
 'Burassbqq6' : Fsbbb['Fsbbb2']/27 - Fsbbb['Fsbbb4']/27 - Fsbcc['Fsbcc4']/36 + Fsbdd['Fsbdd2']/27 - Fsbdd['Fsbdd4']/27 + Fsbss['Fsbss2']/27 - Fsbss['Fsbss4']/27 + Fsbuu['Fsbuu2']/18 - Fsbuu['Fsbuu4']/36,
 'Burassbqq7' : Fsbbb['Fsbbb1']/9 + Fsbbb['Fsbbb2']/27 - (4*Fsbbb['Fsbbb3'])/9 - (4*Fsbbb['Fsbbb4'])/27 + (2*Fsbcc['Fsbcc3'])/3 + (2*Fsbcc['Fsbcc4'])/9 + Fsbdd['Fsbdd1']/9 + Fsbdd['Fsbdd2']/27 - (4*Fsbdd['Fsbdd3'])/9 - (4*Fsbdd['Fsbdd4'])/27 + Fsbss['Fsbss1']/9 + Fsbss['Fsbss2']/27 - (4*Fsbss['Fsbss3'])/9 - (4*Fsbss['Fsbss4'])/27 - Fsbuu['Fsbuu1']/3 - Fsbuu['Fsbuu2']/9 + (2*Fsbuu['Fsbuu3'])/3 + (2*Fsbuu['Fsbuu4'])/9,
 'Burassbqq8' : (2*Fsbbb['Fsbbb2'])/9 - (8*Fsbbb['Fsbbb4'])/9 + (4*Fsbcc['Fsbcc4'])/3 + (2*Fsbdd['Fsbdd2'])/9 - (8*Fsbdd['Fsbdd4'])/9 + (2*Fsbss['Fsbss2'])/9 - (8*Fsbss['Fsbss4'])/9 - (2*Fsbuu['Fsbuu2'])/3 + (4*Fsbuu['Fsbuu4'])/3,
 'Burassbqq9' : -Fsbbb['Fsbbb1']/36 - Fsbbb['Fsbbb2']/108 + Fsbbb['Fsbbb3']/36 + Fsbbb['Fsbbb4']/108 - Fsbcc['Fsbcc3']/24 - Fsbcc['Fsbcc4']/72 - Fsbdd['Fsbdd1']/36 - Fsbdd['Fsbdd2']/108 + Fsbdd['Fsbdd3']/36 + Fsbdd['Fsbdd4']/108 - Fsbss['Fsbss1']/36 - Fsbss['Fsbss2']/108 + Fsbss['Fsbss3']/36 + Fsbss['Fsbss4']/108 + Fsbuu['Fsbuu1']/12 + Fsbuu['Fsbuu2']/36 - Fsbuu['Fsbuu3']/24 - Fsbuu['Fsbuu4']/72,
 'Burassbqq10' : -Fsbbb['Fsbbb2']/18 + Fsbbb['Fsbbb4']/18 - Fsbcc['Fsbcc4']/12 - Fsbdd['Fsbdd2']/18 + Fsbdd['Fsbdd4']/18 - Fsbss['Fsbss2']/18 + Fsbss['Fsbss4']/18 + Fsbuu['Fsbuu2']/6 - Fsbuu['Fsbuu4']/12}

def Buraspsbqq (Fpsbuu,Fpsbdd,Fpsbcc,Fpsbss,Fpsbbb):
    return {'Burassbqq1p' : 2*Fpsbcc['Fsbcc1p'] - 2*Fpsbuu['Fsbuu1p'],
 'Burassbqq2p' : Fpsbcc['Fsbcc1p']/3 + Fpsbcc['Fsbcc2p'] - Fpsbuu['Fsbuu1p']/3 - Fpsbuu['Fsbuu2p'],
 'Burassbqq3p' : (-2*Fpsbbb['Fsbbb1p'])/27 - (2*Fpsbbb['Fsbbb2p'])/81 + (8*Fpsbbb['Fsbbb3p'])/27 + (8*Fpsbbb['Fsbbb4p'])/81 + (2*Fpsbcc['Fsbcc3p'])/9 + (2*Fpsbcc['Fsbcc4p'])/27 - (2*Fpsbdd['Fsbdd1p'])/27 - (2*Fpsbdd['Fsbdd2p'])/81 + (8*Fpsbdd['Fsbdd3p'])/27 + (8*Fpsbdd['Fsbdd4p'])/81 - (2*Fpsbss['Fsbss1p'])/27 - (2*Fpsbss['Fsbss2p'])/81 + (8*Fpsbss['Fsbss3p'])/27 + (8*Fpsbss['Fsbss4p'])/81 - Fpsbuu['Fsbuu1p']/9 - Fpsbuu['Fsbuu2p']/27 + (2*Fpsbuu['Fsbuu3p'])/9 + (2*Fpsbuu['Fsbuu4p'])/27,
 'Burassbqq4p' : (-4*Fpsbbb['Fsbbb2p'])/27 + (16*Fpsbbb['Fsbbb4p'])/27 + (4*Fpsbcc['Fsbcc4p'])/9 - (4*Fpsbdd['Fsbdd2p'])/27 + (16*Fpsbdd['Fsbdd4p'])/27 - (4*Fpsbss['Fsbss2p'])/27 + (16*Fpsbss['Fsbss4p'])/27 - (2*Fpsbuu['Fsbuu2p'])/9 + (4*Fpsbuu['Fsbuu4p'])/9,
 'Burassbqq5p' : Fpsbbb['Fsbbb1p']/54 + Fpsbbb['Fsbbb2p']/162 - Fpsbbb['Fsbbb3p']/54 - Fpsbbb['Fsbbb4p']/162 - Fpsbcc['Fsbcc3p']/72 - Fpsbcc['Fsbcc4p']/216 + Fpsbdd['Fsbdd1p']/54 + Fpsbdd['Fsbdd2p']/162 - Fpsbdd['Fsbdd3p']/54 - Fpsbdd['Fsbdd4p']/162 + Fpsbss['Fsbss1p']/54 + Fpsbss['Fsbss2p']/162 - Fpsbss['Fsbss3p']/54 - Fpsbss['Fsbss4p']/162 + Fpsbuu['Fsbuu1p']/36 + Fpsbuu['Fsbuu2p']/108 - Fpsbuu['Fsbuu3p']/72 - Fpsbuu['Fsbuu4p']/216,
 'Burassbqq6p' : Fpsbbb['Fsbbb2p']/27 - Fpsbbb['Fsbbb4p']/27 - Fpsbcc['Fsbcc4p']/36 + Fpsbdd['Fsbdd2p']/27 - Fpsbdd['Fsbdd4p']/27 + Fpsbss['Fsbss2p']/27 - Fpsbss['Fsbss4p']/27 + Fpsbuu['Fsbuu2p']/18 - Fpsbuu['Fsbuu4p']/36,
 'Burassbqq7p' : Fpsbbb['Fsbbb1p']/9 + Fpsbbb['Fsbbb2p']/27 - (4*Fpsbbb['Fsbbb3p'])/9 - (4*Fpsbbb['Fsbbb4p'])/27 + (2*Fpsbcc['Fsbcc3p'])/3 + (2*Fpsbcc['Fsbcc4p'])/9 + Fpsbdd['Fsbdd1p']/9 + Fpsbdd['Fsbdd2p']/27 - (4*Fpsbdd['Fsbdd3p'])/9 - (4*Fpsbdd['Fsbdd4p'])/27 + Fpsbss['Fsbss1p']/9 + Fpsbss['Fsbss2p']/27 - (4*Fpsbss['Fsbss3p'])/9 - (4*Fpsbss['Fsbss4p'])/27 - Fpsbuu['Fsbuu1p']/3 - Fpsbuu['Fsbuu2p']/9 + (2*Fpsbuu['Fsbuu3p'])/3 + (2*Fpsbuu['Fsbuu4p'])/9,
 'Burassbqq8p' : (2*Fpsbbb['Fsbbb2p'])/9 - (8*Fpsbbb['Fsbbb4p'])/9 + (4*Fpsbcc['Fsbcc4p'])/3 + (2*Fpsbdd['Fsbdd2p'])/9 - (8*Fpsbdd['Fsbdd4p'])/9 + (2*Fpsbss['Fsbss2p'])/9 - (8*Fpsbss['Fsbss4p'])/9 - (2*Fpsbuu['Fsbuu2p'])/3 + (4*Fpsbuu['Fsbuu4p'])/3,
 'Burassbqq9p' : -Fpsbbb['Fsbbb1p']/36 - Fpsbbb['Fsbbb2p']/108 + Fpsbbb['Fsbbb3p']/36 + Fpsbbb['Fsbbb4p']/108 - Fpsbcc['Fsbcc3p']/24 - Fpsbcc['Fsbcc4p']/72 - Fpsbdd['Fsbdd1p']/36 - Fpsbdd['Fsbdd2p']/108 + Fpsbdd['Fsbdd3p']/36 + Fpsbdd['Fsbdd4p']/108 - Fpsbss['Fsbss1p']/36 - Fpsbss['Fsbss2p']/108 + Fpsbss['Fsbss3p']/36 + Fpsbss['Fsbss4p']/108 + Fpsbuu['Fsbuu1p']/12 + Fpsbuu['Fsbuu2p']/36 - Fpsbuu['Fsbuu3p']/24 - Fpsbuu['Fsbuu4p']/72,
 'Burassbqq10p' : -Fpsbbb['Fsbbb2p']/18 + Fpsbbb['Fsbbb4p']/18 - Fpsbcc['Fsbcc4p']/12 - Fpsbdd['Fsbdd2p']/18 + Fpsbdd['Fsbdd4p']/18 - Fpsbss['Fsbss2p']/18 + Fpsbss['Fsbss4p']/18 + Fpsbuu['Fsbuu2p']/6 - Fpsbuu['Fsbuu4p']/12}


# dbqq

def Burasdbqq (Fdbuu,Fdbdd,Fdbcc,Fdbss,Fdbbb):
    return {'Burasdbqq1' : 2*Fdbcc['Fdbcc1'] - 2*Fdbuu['Fdbuu1'],
 'Burasdbqq2' : Fdbcc['Fdbcc1']/3 + Fdbcc['Fdbcc2'] - Fdbuu['Fdbuu1']/3 - Fdbuu['Fdbuu2'],
 'Burasdbqq3' : (-2*Fdbbb['Fdbbb1'])/27 - (2*Fdbbb['Fdbbb2'])/81 + (8*Fdbbb['Fdbbb3'])/27 + (8*Fdbbb['Fdbbb4'])/81 + (2*Fdbcc['Fdbcc3'])/9 + (2*Fdbcc['Fdbcc4'])/27 - (2*Fdbdd['Fdbdd1'])/27 - (2*Fdbdd['Fdbdd2'])/81 + (8*Fdbdd['Fdbdd3'])/27 + (8*Fdbdd['Fdbdd4'])/81 - (2*Fdbss['Fdbss1'])/27 - (2*Fdbss['Fdbss2'])/81 + (8*Fdbss['Fdbss3'])/27 + (8*Fdbss['Fdbss4'])/81 - Fdbuu['Fdbuu1']/9 - Fdbuu['Fdbuu2']/27 + (2*Fdbuu['Fdbuu3'])/9 + (2*Fdbuu['Fdbuu4'])/27,
 'Burasdbqq4' : (-4*Fdbbb['Fdbbb2'])/27 + (16*Fdbbb['Fdbbb4'])/27 + (4*Fdbcc['Fdbcc4'])/9 - (4*Fdbdd['Fdbdd2'])/27 + (16*Fdbdd['Fdbdd4'])/27 - (4*Fdbss['Fdbss2'])/27 + (16*Fdbss['Fdbss4'])/27 - (2*Fdbuu['Fdbuu2'])/9 + (4*Fdbuu['Fdbuu4'])/9,
 'Burasdbqq5' : Fdbbb['Fdbbb1']/54 + Fdbbb['Fdbbb2']/162 - Fdbbb['Fdbbb3']/54 - Fdbbb['Fdbbb4']/162 - Fdbcc['Fdbcc3']/72 - Fdbcc['Fdbcc4']/216 + Fdbdd['Fdbdd1']/54 + Fdbdd['Fdbdd2']/162 - Fdbdd['Fdbdd3']/54 - Fdbdd['Fdbdd4']/162 + Fdbss['Fdbss1']/54 + Fdbss['Fdbss2']/162 - Fdbss['Fdbss3']/54 - Fdbss['Fdbss4']/162 + Fdbuu['Fdbuu1']/36 + Fdbuu['Fdbuu2']/108 - Fdbuu['Fdbuu3']/72 - Fdbuu['Fdbuu4']/216,
 'Burasdbqq6' : Fdbbb['Fdbbb2']/27 - Fdbbb['Fdbbb4']/27 - Fdbcc['Fdbcc4']/36 + Fdbdd['Fdbdd2']/27 - Fdbdd['Fdbdd4']/27 + Fdbss['Fdbss2']/27 - Fdbss['Fdbss4']/27 + Fdbuu['Fdbuu2']/18 - Fdbuu['Fdbuu4']/36,
 'Burasdbqq7' : Fdbbb['Fdbbb1']/9 + Fdbbb['Fdbbb2']/27 - (4*Fdbbb['Fdbbb3'])/9 - (4*Fdbbb['Fdbbb4'])/27 + (2*Fdbcc['Fdbcc3'])/3 + (2*Fdbcc['Fdbcc4'])/9 + Fdbdd['Fdbdd1']/9 + Fdbdd['Fdbdd2']/27 - (4*Fdbdd['Fdbdd3'])/9 - (4*Fdbdd['Fdbdd4'])/27 + Fdbss['Fdbss1']/9 + Fdbss['Fdbss2']/27 - (4*Fdbss['Fdbss3'])/9 - (4*Fdbss['Fdbss4'])/27 - Fdbuu['Fdbuu1']/3 - Fdbuu['Fdbuu2']/9 + (2*Fdbuu['Fdbuu3'])/3 + (2*Fdbuu['Fdbuu4'])/9,
 'Burasdbqq8' : (2*Fdbbb['Fdbbb2'])/9 - (8*Fdbbb['Fdbbb4'])/9 + (4*Fdbcc['Fdbcc4'])/3 + (2*Fdbdd['Fdbdd2'])/9 - (8*Fdbdd['Fdbdd4'])/9 + (2*Fdbss['Fdbss2'])/9 - (8*Fdbss['Fdbss4'])/9 - (2*Fdbuu['Fdbuu2'])/3 + (4*Fdbuu['Fdbuu4'])/3,
 'Burasdbqq9' : -Fdbbb['Fdbbb1']/36 - Fdbbb['Fdbbb2']/108 + Fdbbb['Fdbbb3']/36 + Fdbbb['Fdbbb4']/108 - Fdbcc['Fdbcc3']/24 - Fdbcc['Fdbcc4']/72 - Fdbdd['Fdbdd1']/36 - Fdbdd['Fdbdd2']/108 + Fdbdd['Fdbdd3']/36 + Fdbdd['Fdbdd4']/108 - Fdbss['Fdbss1']/36 - Fdbss['Fdbss2']/108 + Fdbss['Fdbss3']/36 + Fdbss['Fdbss4']/108 + Fdbuu['Fdbuu1']/12 + Fdbuu['Fdbuu2']/36 - Fdbuu['Fdbuu3']/24 - Fdbuu['Fdbuu4']/72,
 'Burasdbqq10' : -Fdbbb['Fdbbb2']/18 + Fdbbb['Fdbbb4']/18 - Fdbcc['Fdbcc4']/12 - Fdbdd['Fdbdd2']/18 + Fdbdd['Fdbdd4']/18 - Fdbss['Fdbss2']/18 + Fdbss['Fdbss4']/18 + Fdbuu['Fdbuu2']/6 - Fdbuu['Fdbuu4']/12}


def Buraspdbqq (Fpdbuu,Fpdbdd,Fpdbcc,Fpdbss,Fpdbbb):
    return {'Burasdbqq1p' : 2*Fpdbcc['Fdbcc1p'] - 2*Fpdbuu['Fdbuu1p'],
 'Burasdbqq2p' : Fpdbcc['Fdbcc1p']/3 + Fpdbcc['Fdbcc2p'] - Fpdbuu['Fdbuu1p']/3 - Fpdbuu['Fdbuu2p'],
 'Burasdbqq3p' : (-2*Fpdbbb['Fdbbb1p'])/27 - (2*Fpdbbb['Fdbbb2p'])/81 + (8*Fpdbbb['Fdbbb3p'])/27 + (8*Fpdbbb['Fdbbb4p'])/81 + (2*Fpdbcc['Fdbcc3p'])/9 + (2*Fpdbcc['Fdbcc4p'])/27 - (2*Fpdbdd['Fdbdd1p'])/27 - (2*Fpdbdd['Fdbdd2p'])/81 + (8*Fpdbdd['Fdbdd3p'])/27 + (8*Fpdbdd['Fdbdd4p'])/81 - (2*Fpdbss['Fdbss1p'])/27 - (2*Fpdbss['Fdbss2p'])/81 + (8*Fpdbss['Fdbss3p'])/27 + (8*Fpdbss['Fdbss4p'])/81 - Fpdbuu['Fdbuu1p']/9 - Fpdbuu['Fdbuu2p']/27 + (2*Fpdbuu['Fdbuu3p'])/9 + (2*Fpdbuu['Fdbuu4p'])/27,
 'Burasdbqq4p' : (-4*Fpdbbb['Fdbbb2p'])/27 + (16*Fpdbbb['Fdbbb4p'])/27 + (4*Fpdbcc['Fdbcc4p'])/9 - (4*Fpdbdd['Fdbdd2p'])/27 + (16*Fpdbdd['Fdbdd4p'])/27 - (4*Fpdbss['Fdbss2p'])/27 + (16*Fpdbss['Fdbss4p'])/27 - (2*Fpdbuu['Fdbuu2p'])/9 + (4*Fpdbuu['Fdbuu4p'])/9,
 'Burasdbqq5p' : Fpdbbb['Fdbbb1p']/54 + Fpdbbb['Fdbbb2p']/162 - Fpdbbb['Fdbbb3p']/54 - Fpdbbb['Fdbbb4p']/162 - Fpdbcc['Fdbcc3p']/72 - Fpdbcc['Fdbcc4p']/216 + Fpdbdd['Fdbdd1p']/54 + Fpdbdd['Fdbdd2p']/162 - Fpdbdd['Fdbdd3p']/54 - Fpdbdd['Fdbdd4p']/162 + Fpdbss['Fdbss1p']/54 + Fpdbss['Fdbss2p']/162 - Fpdbss['Fdbss3p']/54 - Fpdbss['Fdbss4p']/162 + Fpdbuu['Fdbuu1p']/36 + Fpdbuu['Fdbuu2p']/108 - Fpdbuu['Fdbuu3p']/72 - Fpdbuu['Fdbuu4p']/216,
 'Burasdbqq6p' : Fpdbbb['Fdbbb2p']/27 - Fpdbbb['Fdbbb4p']/27 - Fpdbcc['Fdbcc4p']/36 + Fpdbdd['Fdbdd2p']/27 - Fpdbdd['Fdbdd4p']/27 + Fpdbss['Fdbss2p']/27 - Fpdbss['Fdbss4p']/27 + Fpdbuu['Fdbuu2p']/18 - Fpdbuu['Fdbuu4p']/36,
 'Burasdbqq7p' : Fpdbbb['Fdbbb1p']/9 + Fpdbbb['Fdbbb2p']/27 - (4*Fpdbbb['Fdbbb3p'])/9 - (4*Fpdbbb['Fdbbb4p'])/27 + (2*Fpdbcc['Fdbcc3p'])/3 + (2*Fpdbcc['Fdbcc4p'])/9 + Fpdbdd['Fdbdd1p']/9 + Fpdbdd['Fdbdd2p']/27 - (4*Fpdbdd['Fdbdd3p'])/9 - (4*Fpdbdd['Fdbdd4p'])/27 + Fpdbss['Fdbss1p']/9 + Fpdbss['Fdbss2p']/27 - (4*Fpdbss['Fdbss3p'])/9 - (4*Fpdbss['Fdbss4p'])/27 - Fpdbuu['Fdbuu1p']/3 - Fpdbuu['Fdbuu2p']/9 + (2*Fpdbuu['Fdbuu3p'])/3 + (2*Fpdbuu['Fdbuu4p'])/9,
 'Burasdbqq8p' : (2*Fpdbbb['Fdbbb2p'])/9 - (8*Fpdbbb['Fdbbb4p'])/9 + (4*Fpdbcc['Fdbcc4p'])/3 + (2*Fpdbdd['Fdbdd2p'])/9 - (8*Fpdbdd['Fdbdd4p'])/9 + (2*Fpdbss['Fdbss2p'])/9 - (8*Fpdbss['Fdbss4p'])/9 - (2*Fpdbuu['Fdbuu2p'])/3 + (4*Fpdbuu['Fdbuu4p'])/3,
 'Burasdbqq9p' : -Fpdbbb['Fdbbb1p']/36 - Fpdbbb['Fdbbb2p']/108 + Fpdbbb['Fdbbb3p']/36 + Fpdbbb['Fdbbb4p']/108 - Fpdbcc['Fdbcc3p']/24 - Fpdbcc['Fdbcc4p']/72 - Fpdbdd['Fdbdd1p']/36 - Fpdbdd['Fdbdd2p']/108 + Fpdbdd['Fdbdd3p']/36 + Fpdbdd['Fdbdd4p']/108 - Fpdbss['Fdbss1p']/36 - Fpdbss['Fdbss2p']/108 + Fpdbss['Fdbss3p']/36 + Fpdbss['Fdbss4p']/108 + Fpdbuu['Fdbuu1p']/12 + Fpdbuu['Fdbuu2p']/36 - Fpdbuu['Fdbuu3p']/24 - Fpdbuu['Fdbuu4p']/72,
 'Burasdbqq10p' : -Fpdbbb['Fdbbb2p']/18 + Fpdbbb['Fdbbb4p']/18 - Fpdbcc['Fdbcc4p']/12 - Fpdbdd['Fdbdd2p']/18 + Fpdbdd['Fdbdd4p']/18 - Fpdbss['Fdbss2p']/18 + Fpdbss['Fdbss4p']/18 + Fpdbuu['Fdbuu2p']/6 - Fpdbuu['Fdbuu4p']/12}



# BGHW basis

#sbuu

def BGHWsbuu (Fsbuu):
    return {'BGHWsbuu3' : -Fsbuu['Fsbuu1']/3 + (4*Fsbuu['Fsbuu3'])/3 - Fsbuu['Fsbuu2']/(3*Nc) + (4*Fsbuu['Fsbuu4'])/(3*Nc),
 'BGHWsbuu4' : (-2*Fsbuu['Fsbuu2'])/3 + (8*Fsbuu['Fsbuu4'])/3,
 'BGHWsbuu5' : Fsbuu['Fsbuu1']/12 - Fsbuu['Fsbuu3']/12 + Fsbuu['Fsbuu2']/(12*Nc) - Fsbuu['Fsbuu4']/(12*Nc),
 'BGHWsbuu6' : Fsbuu['Fsbuu2']/6 - Fsbuu['Fsbuu4']/6,
 'BGHWsbuu15' : Fsbuu['Fsbuu5'],
 'BGHWsbuu16' : Fsbuu['Fsbuu6'],
 'BGHWsbuu17' : Fsbuu['Fsbuu7'],
 'BGHWsbuu18' : Fsbuu['Fsbuu8'],
 'BGHWsbuu19' : Fsbuu['Fsbuu9'],
 'BGHWsbuu20' : Fsbuu['Fsbuu10']}

def BGHWpsbuu (Fpsbuu):
    return {'BGHWsbuu3p' : -Fpsbuu['Fsbuu1p']/3 + (4*Fpsbuu['Fsbuu3p'])/3 - Fpsbuu['Fsbuu2p']/(3*Nc) + (4*Fpsbuu['Fsbuu4p'])/(3*Nc),
 'BGHWsbuu4p' : (-2*Fpsbuu['Fsbuu2p'])/3 + (8*Fpsbuu['Fsbuu4p'])/3,
 'BGHWsbuu5p' : Fpsbuu['Fsbuu1p']/12 - Fpsbuu['Fsbuu3p']/12 + Fpsbuu['Fsbuu2p']/(12*Nc) - Fpsbuu['Fsbuu4p']/(12*Nc),
 'BGHWsbuu6p' : Fpsbuu['Fsbuu2p']/6 - Fpsbuu['Fsbuu4p']/6,
 'BGHWsbuu15p' : Fpsbuu['Fsbuu5p'],
 'BGHWsbuu16p' : Fpsbuu['Fsbuu6p'],
 'BGHWsbuu17p' : Fpsbuu['Fsbuu7p'],
 'BGHWsbuu18p' : Fpsbuu['Fsbuu8p'],
 'BGHWsbuu19p' : Fpsbuu['Fsbuu9p'],
 'BGHWsbuu20p' : Fpsbuu['Fsbuu10p']}

 # dbuu

def BGHWdbuu (Fdbuu):
    return {'BGHWdbuu3' : -Fdbuu['Fdbuu1']/3 + (4*Fdbuu['Fdbuu3'])/3 - Fdbuu['Fdbuu2']/(3*Nc) + (4*Fdbuu['Fdbuu4'])/(3*Nc),
 'BGHWdbuu4' : (-2*Fdbuu['Fdbuu2'])/3 + (8*Fdbuu['Fdbuu4'])/3,
 'BGHWdbuu5' : Fdbuu['Fdbuu1']/12 - Fdbuu['Fdbuu3']/12 + Fdbuu['Fdbuu2']/(12*Nc) - Fdbuu['Fdbuu4']/(12*Nc),
 'BGHWdbuu6' : Fdbuu['Fdbuu2']/6 - Fdbuu['Fdbuu4']/6,
 'BGHWdbuu15' : Fdbuu['Fdbuu5'],
 'BGHWdbuu16' : Fdbuu['Fdbuu6'],
 'BGHWdbuu17' : Fdbuu['Fdbuu7'],
 'BGHWdbuu18' : Fdbuu['Fdbuu8'],
 'BGHWdbuu19' : Fdbuu['Fdbuu9'],
 'BGHWdbuu20' : Fdbuu['Fdbuu10']}


def BGHWpdbuu (Fpdbuu):
    return {'BGHWdbuu3p' : -Fpdbuu['Fdbuu1p']/3 + (4*Fpdbuu['Fdbuu3p'])/3 - Fpdbuu['Fdbuu2p']/(3*Nc) + (4*Fpdbuu['Fdbuu4p'])/(3*Nc),
 'BGHWdbuu4p' : (-2*Fpdbuu['Fdbuu2p'])/3 + (8*Fpdbuu['Fdbuu4p'])/3,
 'BGHWdbuu5p' : Fpdbuu['Fdbuu1p']/12 - Fpdbuu['Fdbuu3p']/12 + Fpdbuu['Fdbuu2p']/(12*Nc) - Fpdbuu['Fdbuu4p']/(12*Nc),
 'BGHWdbuu6p' : Fpdbuu['Fdbuu2p']/6 - Fpdbuu['Fdbuu4p']/6,
 'BGHWdbuu15p' : Fpdbuu['Fdbuu5p'],
 'BGHWdbuu16p' : Fpdbuu['Fdbuu6p'],
 'BGHWdbuu17p' : Fpdbuu['Fdbuu7p'],
 'BGHWdbuu18p' : Fpdbuu['Fdbuu8p'],
 'BGHWdbuu19p' : Fpdbuu['Fdbuu9p'],
 'BGHWdbuu20p' : Fpdbuu['Fdbuu10p']}


# sbcc

def BGHWsbcc12 (Fsbcc):
    return {'BGHWsbcc1' : Fsbcc['Fsbcc1']/2 - Fsbcc['Fsbcc2']/(2*Nc),
 'BGHWsbcc2' : Fsbcc['Fsbcc2']}

def BGHWsbcc (Fsbcc):
    return {'BGHWsbcc3' : -Fsbcc['Fsbcc1']/3 + (4*Fsbcc['Fsbcc3'])/3 - Fsbcc['Fsbcc2']/(3*Nc) + (4*Fsbcc['Fsbcc4'])/(3*Nc),
 'BGHWsbcc4' : (-2*Fsbcc['Fsbcc2'])/3 + (8*Fsbcc['Fsbcc4'])/3,
 'BGHWsbcc5' : Fsbcc['Fsbcc1']/12 - Fsbcc['Fsbcc3']/12 + Fsbcc['Fsbcc2']/(12*Nc) - Fsbcc['Fsbcc4']/(12*Nc),
 'BGHWsbcc6' : Fsbcc['Fsbcc2']/6 - Fsbcc['Fsbcc4']/6,
 'BGHWsbcc15' : Fsbcc['Fsbcc5'],
 'BGHWsbcc16' : Fsbcc['Fsbcc6'],
 'BGHWsbcc17' : Fsbcc['Fsbcc7'],
 'BGHWsbcc18' : Fsbcc['Fsbcc8'],
 'BGHWsbcc19' : Fsbcc['Fsbcc9'],
 'BGHWsbcc20' : Fsbcc['Fsbcc10']}

def BGHWpsbcc12 (Fsbcc):
    return {'BGHWsbcc1p' : Fpsbcc['Fsbcc1p']/2 - Fpsbcc['Fsbcc2p']/(2*Nc),
 'BGHWsbcc2p' : Fpsbcc['Fsbcc2p']}

def BGHWpsbcc (Fpsbcc):
    return {'BGHWsbcc3p' : -Fpsbcc['Fsbcc1p']/3 + (4*Fpsbcc['Fsbcc3p'])/3 - Fpsbcc['Fsbcc2p']/(3*Nc) + (4*Fpsbcc['Fsbcc4p'])/(3*Nc),
 'BGHWsbcc4p' : (-2*Fpsbcc['Fsbcc2p'])/3 + (8*Fpsbcc['Fsbcc4p'])/3,
 'BGHWsbcc5p' : Fpsbcc['Fsbcc1p']/12 - Fpsbcc['Fsbcc3p']/12 + Fpsbcc['Fsbcc2p']/(12*Nc) - Fpsbcc['Fsbcc4p']/(12*Nc),
 'BGHWsbcc6p' : Fpsbcc['Fsbcc2p']/6 - Fpsbcc['Fsbcc4p']/6,
 'BGHWsbcc15p' : Fpsbcc['Fsbcc5p'],
 'BGHWsbcc16p' : Fpsbcc['Fsbcc6p'],
 'BGHWsbcc17p' : Fpsbcc['Fsbcc7p'],
 'BGHWsbcc18p' : Fpsbcc['Fsbcc8p'],
 'BGHWsbcc19p' : Fpsbcc['Fsbcc9p'],
 'BGHWsbcc20p' : Fpsbcc['Fsbcc10p']}

# dbcc

def BGHWdbcc12 (Fdbcc):
    return {'BGHWdbcc1' : Fdbcc['Fdbcc1']/2 - Fdbcc['Fdbcc2']/(2*Nc),
 'BGHWdbcc2' : Fdbcc['Fdbcc2']}

def BGHWdbcc (Fdbcc):
    return {'BGHWdbcc3' : -Fdbcc['Fdbcc1']/3 + (4*Fdbcc['Fdbcc3'])/3 - Fdbcc['Fdbcc2']/(3*Nc) + (4*Fdbcc['Fdbcc4'])/(3*Nc),
 'BGHWdbcc4' : (-2*Fdbcc['Fdbcc2'])/3 + (8*Fdbcc['Fdbcc4'])/3,
 'BGHWdbcc5' : Fdbcc['Fdbcc1']/12 - Fdbcc['Fdbcc3']/12 + Fdbcc['Fdbcc2']/(12*Nc) - Fdbcc['Fdbcc4']/(12*Nc),
 'BGHWdbcc6' : Fdbcc['Fdbcc2']/6 - Fdbcc['Fdbcc4']/6,
 'BGHWdbcc15' : Fdbcc['Fdbcc5'],
 'BGHWdbcc16' : Fdbcc['Fdbcc6'],
 'BGHWdbcc17' : Fdbcc['Fdbcc7'],
 'BGHWdbcc18' : Fdbcc['Fdbcc8'],
 'BGHWdbcc19' : Fdbcc['Fdbcc9'],
 'BGHWdbcc20' : Fdbcc['Fdbcc10']}

def BGHWpdbcc12 (Fdbcc):
    return {'BGHWdbcc1p' : Fpdbcc['Fdbcc1p']/2 - Fpdbcc['Fdbcc2p']/(2*Nc),
 'BGHWdbcc2p' : Fpdbcc['Fdbcc2p']}


def BGHWpdbcc (Fpdbcc):
    return {'BGHWdbcc3p' : -Fpdbcc['Fdbcc1p']/3 + (4*Fpdbcc['Fdbcc3p'])/3 - Fpdbcc['Fdbcc2p']/(3*Nc) + (4*Fpdbcc['Fdbcc4p'])/(3*Nc),
 'BGHWdbcc4p' : (-2*Fpdbcc['Fdbcc2p'])/3 + (8*Fpdbcc['Fdbcc4p'])/3,
 'BGHWdbcc5p' : Fpdbcc['Fdbcc1p']/12 - Fpdbcc['Fdbcc3p']/12 + Fpdbcc['Fdbcc2p']/(12*Nc) - Fpdbcc['Fdbcc4p']/(12*Nc),
 'BGHWdbcc6p' : Fpdbcc['Fdbcc2p']/6 - Fpdbcc['Fdbcc4p']/6,
 'BGHWdbcc15p' : Fpdbcc['Fdbcc5p'],
 'BGHWdbcc16p' : Fpdbcc['Fdbcc6p'],
 'BGHWdbcc17p' : Fpdbcc['Fdbcc7p'],
 'BGHWdbcc18p' : Fpdbcc['Fdbcc8p'],
 'BGHWdbcc19p' : Fpdbcc['Fdbcc9p'],
 'BGHWdbcc20p' : Fpdbcc['Fdbcc10p']}

# sbdd

def BGHWsbdd (Fsbdd):
    return {'BGHWsbdd3' : -Fsbdd['Fsbdd1']/3 + (4*Fsbdd['Fsbdd3'])/3 - Fsbdd['Fsbdd2']/(3*Nc) + (4*Fsbdd['Fsbdd4'])/(3*Nc),
 'BGHWsbdd4' : (-2*Fsbdd['Fsbdd2'])/3 + (8*Fsbdd['Fsbdd4'])/3,
 'BGHWsbdd5' : Fsbdd['Fsbdd1']/12 - Fsbdd['Fsbdd3']/12 + Fsbdd['Fsbdd2']/(12*Nc) - Fsbdd['Fsbdd4']/(12*Nc),
 'BGHWsbdd6' : Fsbdd['Fsbdd2']/6 - Fsbdd['Fsbdd4']/6,
 'BGHWsbdd15' : Fsbdd['Fsbdd5'],
 'BGHWsbdd16' : Fsbdd['Fsbdd6'],
 'BGHWsbdd17' : Fsbdd['Fsbdd7'],
 'BGHWsbdd18' : Fsbdd['Fsbdd8'],
 'BGHWsbdd19' : Fsbdd['Fsbdd9'],
 'BGHWsbdd20' : Fsbdd['Fsbdd10']}

def BGHWpsbdd (Fpsbdd):
    return {'BGHWsbdd3p' : -Fpsbdd['Fsbdd1p']/3 + (4*Fpsbdd['Fsbdd3p'])/3 - Fpsbdd['Fsbdd2p']/(3*Nc) + (4*Fpsbdd['Fsbdd4p'])/(3*Nc),
 'BGHWsbdd4p' : (-2*Fpsbdd['Fsbdd2p'])/3 + (8*Fpsbdd['Fsbdd4p'])/3,
 'BGHWsbdd5p' : Fpsbdd['Fsbdd1p']/12 - Fpsbdd['Fsbdd3p']/12 + Fpsbdd['Fsbdd2p']/(12*Nc) - Fpsbdd['Fsbdd4p']/(12*Nc),
 'BGHWsbdd6p' : Fpsbdd['Fsbdd2p']/6 - Fpsbdd['Fsbdd4p']/6,
 'BGHWsbdd15p' : Fpsbdd['Fsbdd5p'],
 'BGHWsbdd16p' : Fpsbdd['Fsbdd6p'],
 'BGHWsbdd17p' : Fpsbdd['Fsbdd7p'],
 'BGHWsbdd18p' : Fpsbdd['Fsbdd8p'],
 'BGHWsbdd19p' : Fpsbdd['Fsbdd9p'],
 'BGHWsbdd20p' : Fpsbdd['Fsbdd10p']}

 # dbdd

def BGHWdbdd (Fdbdd):
    return {'BGHWdbdd3' : -Fdbdd['Fdbdd1']/3 + (4*Fdbdd['Fdbdd3'])/3 - Fdbdd['Fdbdd2']/(3*Nc) + (4*Fdbdd['Fdbdd4'])/(3*Nc),
 'BGHWdbdd4' : (-2*Fdbdd['Fdbdd2'])/3 + (8*Fdbdd['Fdbdd4'])/3,
 'BGHWdbdd5' : Fdbdd['Fdbdd1']/12 - Fdbdd['Fdbdd3']/12 + Fdbdd['Fdbdd2']/(12*Nc) - Fdbdd['Fdbdd4']/(12*Nc),
 'BGHWdbdd6' : Fdbdd['Fdbdd2']/6 - Fdbdd['Fdbdd4']/6,
 'BGHWdbdd15' : Fdbdd['Fdbdd5'],
 'BGHWdbdd16' : Fdbdd['Fdbdd6'],
 'BGHWdbdd17' : Fdbdd['Fdbdd7'],
 'BGHWdbdd18' : Fdbdd['Fdbdd8'],
 'BGHWdbdd19' : Fdbdd['Fdbdd9'],
 'BGHWdbdd20' : Fdbdd['Fdbdd10']}

def BGHWpdbdd (Fpdbdd):
    return {'BGHWdbdd3p' : -Fpdbdd['Fdbdd1p']/3 + (4*Fpdbdd['Fdbdd3p'])/3 - Fpdbdd['Fdbdd2p']/(3*Nc) + (4*Fpdbdd['Fdbdd4p'])/(3*Nc),
 'BGHWdbdd4p' : (-2*Fpdbdd['Fdbdd2p'])/3 + (8*Fpdbdd['Fdbdd4p'])/3,
 'BGHWdbdd5p' : Fpdbdd['Fdbdd1p']/12 - Fpdbdd['Fdbdd3p']/12 + Fpdbdd['Fdbdd2p']/(12*Nc) - Fpdbdd['Fdbdd4p']/(12*Nc),
 'BGHWdbdd6p' : Fpdbdd['Fdbdd2p']/6 - Fpdbdd['Fdbdd4p']/6,
 'BGHWdbdd15p' : Fpdbdd['Fdbdd5p'],
 'BGHWdbdd16p' : Fpdbdd['Fdbdd6p'],
 'BGHWdbdd17p' : Fpdbdd['Fdbdd7p'],
 'BGHWdbdd18p' : Fpdbdd['Fdbdd8p'],
 'BGHWdbdd19p' : Fpdbdd['Fdbdd9p'],
 'BGHWdbdd20p' : Fpdbdd['Fdbdd10p']}



# sbss

def BGHWsbss (Fsbss):
    return {'BGHWsbss3' : -Fsbss['Fsbss1']/3 + (4*Fsbss['Fsbss3'])/3 - Fsbss['Fsbss2']/(3*Nc) + (4*Fsbss['Fsbss4'])/(3*Nc),
 'BGHWsbss4' : (-2*Fsbss['Fsbss2'])/3 + (8*Fsbss['Fsbss4'])/3,
 'BGHWsbss5' : Fsbss['Fsbss1']/12 - Fsbss['Fsbss3']/12 + Fsbss['Fsbss2']/(12*Nc) - Fsbss['Fsbss4']/(12*Nc),
 'BGHWsbss6' : Fsbss['Fsbss2']/6 - Fsbss['Fsbss4']/6,
 'BGHWsbss15' : Fsbss['Fsbss5'],
 'BGHWsbss16' : Fsbss['Fsbss6'],
 'BGHWsbss17' : Fsbss['Fsbss7'],
 'BGHWsbss18' : Fsbss['Fsbss8'],
 'BGHWsbss19' : Fsbss['Fsbss9'],
 'BGHWsbss20' : Fsbss['Fsbss10']}


def BGHWpsbss (Fpsbss):
    return {'BGHWsbss3p' : -Fpsbss['Fsbss1p']/3 + (4*Fpsbss['Fsbss3p'])/3 - Fpsbss['Fsbss2p']/(3*Nc) + (4*Fpsbss['Fsbss4p'])/(3*Nc),
 'BGHWsbss4p' : (-2*Fpsbss['Fsbss2p'])/3 + (8*Fpsbss['Fsbss4p'])/3,
 'BGHWsbss5p' : Fpsbss['Fsbss1p']/12 - Fpsbss['Fsbss3p']/12 + Fpsbss['Fsbss2p']/(12*Nc) - Fpsbss['Fsbss4p']/(12*Nc),
 'BGHWsbss6p' : Fpsbss['Fsbss2p']/6 - Fpsbss['Fsbss4p']/6,
 'BGHWsbss15p' : Fpsbss['Fsbss5p'],
 'BGHWsbss16p' : Fpsbss['Fsbss6p'],
 'BGHWsbss17p' : Fpsbss['Fsbss7p'],
 'BGHWsbss18p' : Fpsbss['Fsbss8p'],
 'BGHWsbss19p' : Fpsbss['Fsbss9p'],
 'BGHWsbss20p' : Fpsbss['Fsbss10p']}


# dbss

def BGHWdbss (Fdbss):
    return {'BGHWdbss3' : -Fdbss['Fdbss1']/3 + (4*Fdbss['Fdbss3'])/3 - Fdbss['Fdbss2']/(3*Nc) + (4*Fdbss['Fdbss4'])/(3*Nc),
 'BGHWdbss4' : (-2*Fdbss['Fdbss2'])/3 + (8*Fdbss['Fdbss4'])/3,
 'BGHWdbss5' : Fdbss['Fdbss1']/12 - Fdbss['Fdbss3']/12 + Fdbss['Fdbss2']/(12*Nc) - Fdbss['Fdbss4']/(12*Nc),
 'BGHWdbss6' : Fdbss['Fdbss2']/6 - Fdbss['Fdbss4']/6,
 'BGHWdbss15' : Fdbss['Fdbss5'],
 'BGHWdbss16' : Fdbss['Fdbss6'],
 'BGHWdbss17' : Fdbss['Fdbss7'],
 'BGHWdbss18' : Fdbss['Fdbss8'],
 'BGHWdbss19' : Fdbss['Fdbss9'],
 'BGHWdbss20' : Fdbss['Fdbss10']}


def BGHWpdbss (Fpdbss):
    return {'BGHWdbss3p' : -Fpdbss['Fdbss1p']/3 + (4*Fpdbss['Fdbss3p'])/3 - Fpdbss['Fdbss2p']/(3*Nc) + (4*Fpdbss['Fdbss4p'])/(3*Nc),
 'BGHWdbss4p' : (-2*Fpdbss['Fdbss2p'])/3 + (8*Fpdbss['Fdbss4p'])/3,
 'BGHWdbss5p' : Fpdbss['Fdbss1p']/12 - Fpdbss['Fdbss3p']/12 + Fpdbss['Fdbss2p']/(12*Nc) - Fpdbss['Fdbss4p']/(12*Nc),
 'BGHWdbss6p' : Fpdbss['Fdbss2p']/6 - Fpdbss['Fdbss4p']/6,
 'BGHWdbss15p' : Fpdbss['Fdbss5p'],
 'BGHWdbss16p' : Fpdbss['Fdbss6p'],
 'BGHWdbss17p' : Fpdbss['Fdbss7p'],
 'BGHWdbss18p' : Fpdbss['Fdbss8p'],
 'BGHWdbss19p' : Fpdbss['Fdbss9p'],
 'BGHWdbss20p' : Fpdbss['Fdbss10p']}


# sbbb

def BGHWsbbb (Fsbbb):
    return {'BGHWsbbb3' : -Fsbbb['Fsbbb1']/3 + (4*Fsbbb['Fsbbb3'])/3 - Fsbbb['Fsbbb2']/(3*Nc) + (4*Fsbbb['Fsbbb4'])/(3*Nc),
 'BGHWsbbb4' : (-2*Fsbbb['Fsbbb2'])/3 + (8*Fsbbb['Fsbbb4'])/3,
 'BGHWsbbb5' : Fsbbb['Fsbbb1']/12 - Fsbbb['Fsbbb3']/12 + Fsbbb['Fsbbb2']/(12*Nc) - Fsbbb['Fsbbb4']/(12*Nc),
 'BGHWsbbb6' : Fsbbb['Fsbbb2']/6 - Fsbbb['Fsbbb4']/6,
 'BGHWsbbb15' : Fsbbb['Fsbbb5'],
 'BGHWsbbb16' : Fsbbb['Fsbbb6'],
 'BGHWsbbb17' : Fsbbb['Fsbbb7'],
 'BGHWsbbb18' : Fsbbb['Fsbbb8'],
 'BGHWsbbb19' : Fsbbb['Fsbbb9'],
 'BGHWsbbb20' : Fsbbb['Fsbbb10']}


def BGHWpsbbb (Fpsbbb):
    return {'BGHWsbbb3p' : -Fpsbbb['Fsbbb1p']/3 + (4*Fpsbbb['Fsbbb3p'])/3 - Fpsbbb['Fsbbb2p']/(3*Nc) + (4*Fpsbbb['Fsbbb4p'])/(3*Nc),
 'BGHWsbbb4p' : (-2*Fpsbbb['Fsbbb2p'])/3 + (8*Fpsbbb['Fsbbb4p'])/3,
 'BGHWsbbb5p' : Fpsbbb['Fsbbb1p']/12 - Fpsbbb['Fsbbb3p']/12 + Fpsbbb['Fsbbb2p']/(12*Nc) - Fpsbbb['Fsbbb4p']/(12*Nc),
 'BGHWsbbb6p' : Fpsbbb['Fsbbb2p']/6 - Fpsbbb['Fsbbb4p']/6,
 'BGHWsbbb15p' : Fpsbbb['Fsbbb5p'],
 'BGHWsbbb16p' : Fpsbbb['Fsbbb6p'],
 'BGHWsbbb17p' : Fpsbbb['Fsbbb7p'],
 'BGHWsbbb18p' : Fpsbbb['Fsbbb8p'],
 'BGHWsbbb19p' : Fpsbbb['Fsbbb9p'],
 'BGHWsbbb20p' : Fpsbbb['Fsbbb10p']}

# dbbb


def BGHWdbbb (Fdbbb):
    return {'BGHWdbbb3' : -Fdbbb['Fdbbb1']/3 + (4*Fdbbb['Fdbbb3'])/3 - Fdbbb['Fdbbb2']/(3*Nc) + (4*Fdbbb['Fdbbb4'])/(3*Nc),
 'BGHWdbbb4' : (-2*Fdbbb['Fdbbb2'])/3 + (8*Fdbbb['Fdbbb4'])/3,
 'BGHWdbbb5' : Fdbbb['Fdbbb1']/12 - Fdbbb['Fdbbb3']/12 + Fdbbb['Fdbbb2']/(12*Nc) - Fdbbb['Fdbbb4']/(12*Nc),
 'BGHWdbbb6' : Fdbbb['Fdbbb2']/6 - Fdbbb['Fdbbb4']/6,
 'BGHWdbbb15' : Fdbbb['Fdbbb5'],
 'BGHWdbbb16' : Fdbbb['Fdbbb6'],
 'BGHWdbbb17' : Fdbbb['Fdbbb7'],
 'BGHWdbbb18' : Fdbbb['Fdbbb8'],
 'BGHWdbbb19' : Fdbbb['Fdbbb9'],
 'BGHWdbbb20' : Fdbbb['Fdbbb10']}


def BGHWpdbbb (Fpdbbb):
    return {'BGHWdbbb3p' : -Fpdbbb['Fdbbb1p']/3 + (4*Fpdbbb['Fdbbb3p'])/3 - Fpdbbb['Fdbbb2p']/(3*Nc) + (4*Fpdbbb['Fdbbb4p'])/(3*Nc),
 'BGHWdbbb4p' : (-2*Fpdbbb['Fdbbb2p'])/3 + (8*Fpdbbb['Fdbbb4p'])/3,
 'BGHWdbbb5p' : Fpdbbb['Fdbbb1p']/12 - Fpdbbb['Fdbbb3p']/12 + Fpdbbb['Fdbbb2p']/(12*Nc) - Fpdbbb['Fdbbb4p']/(12*Nc),
 'BGHWdbbb6p' : Fpdbbb['Fdbbb2p']/6 - Fpdbbb['Fdbbb4p']/6,
 'BGHWdbbb15p' : Fpdbbb['Fdbbb5p'],
 'BGHWdbbb16p' : Fpdbbb['Fdbbb6p'],
 'BGHWdbbb17p' : Fpdbbb['Fdbbb7p'],
 'BGHWdbbb18p' : Fpdbbb['Fdbbb8p'],
 'BGHWdbbb19p' : Fpdbbb['Fdbbb9p'],
 'BGHWdbbb20p' : Fpdbbb['Fdbbb10p']}


# Delta F =1 basis

# sbqq

def DF1sbqq (Fsbuu,Fsbdd,Fsbcc,Fsbss,Fsbbb):
    return {'DF1sbqq1' : -Fsbcc['Fsbcc1']/4 + Fsbuu['Fsbuu1']/4,
 'DF1sbqq2' : -Fsbcc['Fsbcc2']/4 + Fsbuu['Fsbuu2']/4,
 'DF1sbqq3' : Fsbbb['Fsbbb1']/18 + Fsbcc['Fsbcc1']/12 + Fsbdd['Fsbdd1']/18 + Fsbss['Fsbss1']/18,
 'DF1sbqq4' : Fsbbb['Fsbbb2']/18 + Fsbcc['Fsbcc2']/12 + Fsbdd['Fsbdd2']/18 + Fsbss['Fsbss2']/18,
 'DF1sbqq5' : Fsbbb['Fsbbb3']/18 + Fsbcc['Fsbcc3']/24 + Fsbdd['Fsbdd3']/18 + Fsbss['Fsbss3']/18 + Fsbuu['Fsbuu3']/24,
 'DF1sbqq6' : Fsbbb['Fsbbb4']/18 + Fsbcc['Fsbcc4']/24 + Fsbdd['Fsbdd4']/18 + Fsbss['Fsbss4']/18 + Fsbuu['Fsbuu4']/24,
 'DF1sbqq7' : -Fsbbb['Fsbbb3']/18 + Fsbcc['Fsbcc3']/12 - Fsbdd['Fsbdd3']/18 - Fsbss['Fsbss3']/18 + Fsbuu['Fsbuu3']/12,
 'DF1sbqq8' : -Fsbbb['Fsbbb4']/18 + Fsbcc['Fsbcc4']/12 - Fsbdd['Fsbdd4']/18 - Fsbss['Fsbss4']/18 + Fsbuu['Fsbuu4']/12,
 'DF1sbqq9' : -Fsbbb['Fsbbb1']/18 + Fsbcc['Fsbcc1']/6 - Fsbdd['Fsbdd1']/18 - Fsbss['Fsbss1']/18,
 'DF1sbqq10' : -Fsbbb['Fsbbb2']/18 + Fsbcc['Fsbcc2']/6 - Fsbdd['Fsbdd2']/18 - Fsbss['Fsbss2']/18}

def DF1psbqq (Fpsbuu,Fpsbdd,Fpsbcc,Fpsbss,Fpsbbb):
    return {'DF1sbqq1p' : -Fpsbcc['Fsbcc1p']/4 + Fpsbuu['Fsbuu1p']/4,
 'DF1sbqq2p' : -Fpsbcc['Fsbcc2p']/4 + Fpsbuu['Fsbuu2p']/4,
 'DF1sbqq3p' : Fpsbbb['Fsbbb1p']/18 + Fpsbcc['Fsbcc1p']/12 + Fpsbdd['Fsbdd1p']/18 + Fpsbss['Fsbss1p']/18,
 'DF1sbqq4p' : Fpsbbb['Fsbbb2p']/18 + Fpsbcc['Fsbcc2p']/12 + Fpsbdd['Fsbdd2p']/18 + Fpsbss['Fsbss2p']/18,
 'DF1sbqq5p' : Fpsbbb['Fsbbb3p']/18 + Fpsbcc['Fsbcc3p']/24 + Fpsbdd['Fsbdd3p']/18 + Fpsbss['Fsbss3p']/18 + Fpsbuu['Fsbuu3p']/24,
 'DF1sbqq6p' : Fpsbbb['Fsbbb4p']/18 + Fpsbcc['Fsbcc4p']/24 + Fpsbdd['Fsbdd4p']/18 + Fpsbss['Fsbss4p']/18 + Fpsbuu['Fsbuu4p']/24,
 'DF1sbqq7p' : -Fpsbbb['Fsbbb3p']/18 + Fpsbcc['Fsbcc3p']/12 - Fpsbdd['Fsbdd3p']/18 - Fpsbss['Fsbss3p']/18 + Fpsbuu['Fsbuu3p']/12,
 'DF1sbqq8p' : -Fpsbbb['Fsbbb4p']/18 + Fpsbcc['Fsbcc4p']/12 - Fpsbdd['Fsbdd4p']/18 - Fpsbss['Fsbss4p']/18 + Fpsbuu['Fsbuu4p']/12,
 'DF1sbqq9p' : -Fpsbbb['Fsbbb1p']/18 + Fpsbcc['Fsbcc1p']/6 - Fpsbdd['Fsbdd1p']/18 - Fpsbss['Fsbss1p']/18,
 'DF1sbqq10p' : -Fpsbbb['Fsbbb2p']/18 + Fpsbcc['Fsbcc2p']/6 - Fpsbdd['Fsbdd2p']/18 - Fpsbss['Fsbss2p']/18}

# dbqq

def DF1dbqq (Fdbuu,Fdbdd,Fdbcc,Fdbss,Fdbbb):
    return {'DF1dbqq1' : -Fdbcc['Fdbcc1']/4 + Fdbuu['Fdbuu1']/4,
 'DF1dbqq2' : -Fdbcc['Fdbcc2']/4 + Fdbuu['Fdbuu2']/4,
 'DF1dbqq3' : Fdbbb['Fdbbb1']/18 + Fdbcc['Fdbcc1']/12 + Fdbdd['Fdbdd1']/18 + Fdbss['Fdbss1']/18,
 'DF1dbqq4' : Fdbbb['Fdbbb2']/18 + Fdbcc['Fdbcc2']/12 + Fdbdd['Fdbdd2']/18 + Fdbss['Fdbss2']/18,
 'DF1dbqq5' : Fdbbb['Fdbbb3']/18 + Fdbcc['Fdbcc3']/24 + Fdbdd['Fdbdd3']/18 + Fdbss['Fdbss3']/18 + Fdbuu['Fdbuu3']/24,
 'DF1dbqq6' : Fdbbb['Fdbbb4']/18 + Fdbcc['Fdbcc4']/24 + Fdbdd['Fdbdd4']/18 + Fdbss['Fdbss4']/18 + Fdbuu['Fdbuu4']/24,
 'DF1dbqq7' : -Fdbbb['Fdbbb3']/18 + Fdbcc['Fdbcc3']/12 - Fdbdd['Fdbdd3']/18 - Fdbss['Fdbss3']/18 + Fdbuu['Fdbuu3']/12,
 'DF1dbqq8' : -Fdbbb['Fdbbb4']/18 + Fdbcc['Fdbcc4']/12 - Fdbdd['Fdbdd4']/18 - Fdbss['Fdbss4']/18 + Fdbuu['Fdbuu4']/12,
 'DF1dbqq9' : -Fdbbb['Fdbbb1']/18 + Fdbcc['Fdbcc1']/6 - Fdbdd['Fdbdd1']/18 - Fdbss['Fdbss1']/18,
 'DF1dbqq10' : -Fdbbb['Fdbbb2']/18 + Fdbcc['Fdbcc2']/6 - Fdbdd['Fdbdd2']/18 - Fdbss['Fdbss2']/18}


def DF1pdbqq (Fpdbuu,Fpdbdd,Fpdbcc,Fpdbss,Fpdbbb):
    return {'DF1dbqq1p' : -Fpdbcc['Fdbcc1p']/4 + Fpdbuu['Fdbuu1p']/4,
 'DF1dbqq2p' : -Fpdbcc['Fdbcc2p']/4 + Fpdbuu['Fdbuu2p']/4,
 'DF1dbqq3p' : Fpdbbb['Fdbbb1p']/18 + Fpdbcc['Fdbcc1p']/12 + Fpdbdd['Fdbdd1p']/18 + Fpdbss['Fdbss1p']/18,
 'DF1dbqq4p' : Fpdbbb['Fdbbb2p']/18 + Fpdbcc['Fdbcc2p']/12 + Fpdbdd['Fdbdd2p']/18 + Fpdbss['Fdbss2p']/18,
 'DF1dbqq5p' : Fpdbbb['Fdbbb3p']/18 + Fpdbcc['Fdbcc3p']/24 + Fpdbdd['Fdbdd3p']/18 + Fpdbss['Fdbss3p']/18 + Fpdbuu['Fdbuu3p']/24,
 'DF1dbqq6p' : Fpdbbb['Fdbbb4p']/18 + Fpdbcc['Fdbcc4p']/24 + Fpdbdd['Fdbdd4p']/18 + Fpdbss['Fdbss4p']/18 + Fpdbuu['Fdbuu4p']/24,
 'DF1dbqq7p' : -Fpdbbb['Fdbbb3p']/18 + Fpdbcc['Fdbcc3p']/12 - Fpdbdd['Fdbdd3p']/18 - Fpdbss['Fdbss3p']/18 + Fpdbuu['Fdbuu3p']/12,
 'DF1dbqq8p' : -Fpdbbb['Fdbbb4p']/18 + Fpdbcc['Fdbcc4p']/12 - Fpdbdd['Fdbdd4p']/18 - Fpdbss['Fdbss4p']/18 + Fpdbuu['Fdbuu4p']/12,
 'DF1dbqq9p' : -Fpdbbb['Fdbbb1p']/18 + Fpdbcc['Fdbcc1p']/6 - Fpdbdd['Fdbdd1p']/18 - Fpdbss['Fdbss1p']/18,
 'DF1dbqq10p' : -Fpdbbb['Fdbbb2p']/18 + Fpdbcc['Fdbcc2p']/6 - Fpdbdd['Fdbdd2p']/18 - Fpdbss['Fdbss2p']/18}


# EOS basis not unique!
#EOS Basis
def EOSsbqq (Fsbuu,Fsbdd,Fsbcc,Fsbss,Fsbbb):
    return {'EOSsbqqu1' : -2*Fsbuu['Fsbuu1'] + 2*Fsbuu['Fsbuu1'],
 'EOSsbqqu2' : -Fsbcc['Fsbcc1']/3 - Fsbcc['Fsbcc2'] + Fsbuu['Fsbuu1']/3 + Fsbuu['Fsbuu2'],
'EOSsbqqc1' : -2*Fsbcc['Fsbcc1'] + 2*Fsbuu['Fsbuu1'],
 'EOSsbqqc2' : -Fsbcc['Fsbcc1']/3 - Fsbcc['Fsbcc2'] + Fsbuu['Fsbuu1']/3 + Fsbuu['Fsbuu2'],
 'EOSsbqq3' : (-2*Fsbbb['Fsbbb2'])/81 + (8*Fsbbb['Fsbbb3'])/27 + (8*Fsbbb['Fsbbb4'])/81 - Fsbcc['Fsbcc1']/9 - Fsbcc['Fsbcc2']/27 + (2*Fsbcc['Fsbcc3'])/9 + (2*Fsbcc['Fsbcc4'])/27 - Fsbdd['Fsbdd1']/9 - (2*Fsbdd['Fsbdd2'])/81 + (8*Fsbdd['Fsbdd3'])/27 + (8*Fsbdd['Fsbdd4'])/81 - Fsbss['Fsbss1']/9 - (2*Fsbss['Fsbss2'])/81 + (8*Fsbss['Fsbss3'])/27 + (8*Fsbss['Fsbss4'])/81 + (2*Fsbuu['Fsbuu3'])/9 + (2*Fsbuu['Fsbuu4'])/27,
 'EOSsbqq4' : (-4*Fsbbb['Fsbbb2'])/27 + (16*Fsbbb['Fsbbb4'])/27 - (2*Fsbcc['Fsbcc2'])/9 + (4*Fsbcc['Fsbcc4'])/9 - (4*Fsbdd['Fsbdd2'])/27 + (16*Fsbdd['Fsbdd4'])/27 - (4*Fsbss['Fsbss2'])/27 + (16*Fsbss['Fsbss4'])/27 + (4*Fsbuu['Fsbuu4'])/9,
 'EOSsbqq5' : Fsbbb['Fsbbb2']/162 - Fsbbb['Fsbbb3']/54 - Fsbbb['Fsbbb4']/162 + Fsbcc['Fsbcc1']/36 + Fsbcc['Fsbcc2']/108 - Fsbcc['Fsbcc3']/72 - Fsbcc['Fsbcc4']/216 + Fsbdd['Fsbdd1']/36 + Fsbdd['Fsbdd2']/162 - Fsbdd['Fsbdd3']/54 - Fsbdd['Fsbdd4']/162 + Fsbss['Fsbss1']/36 + Fsbss['Fsbss2']/162 - Fsbss['Fsbss3']/54 - Fsbss['Fsbss4']/162 - Fsbuu['Fsbuu3']/72 - Fsbuu['Fsbuu4']/216,
 'EOSsbqq6' : Fsbbb['Fsbbb2']/27 - Fsbbb['Fsbbb4']/27 + Fsbcc['Fsbcc2']/18 - Fsbcc['Fsbcc4']/36 + Fsbdd['Fsbdd2']/27 - Fsbdd['Fsbdd4']/27 + Fsbss['Fsbss2']/27 - Fsbss['Fsbss4']/27 - Fsbuu['Fsbuu4']/36,
 'EOSsbqq3Q' : Fsbbb['Fsbbb2']/27 - (4*Fsbbb['Fsbbb3'])/9 - (4*Fsbbb['Fsbbb4'])/27 - Fsbcc['Fsbcc1']/3 - Fsbcc['Fsbcc2']/9 + (2*Fsbcc['Fsbcc3'])/3 + (2*Fsbcc['Fsbcc4'])/9 + Fsbdd['Fsbdd1']/6 + Fsbdd['Fsbdd2']/27 - (4*Fsbdd['Fsbdd3'])/9 - (4*Fsbdd['Fsbdd4'])/27 + Fsbss['Fsbss1']/6 + Fsbss['Fsbss2']/27 - (4*Fsbss['Fsbss3'])/9 - (4*Fsbss['Fsbss4'])/27 + (2*Fsbuu['Fsbuu3'])/3 + (2*Fsbuu['Fsbuu4'])/9,
 'EOSsbqq4Q' : (2*Fsbbb['Fsbbb2'])/9 - (8*Fsbbb['Fsbbb4'])/9 - (2*Fsbcc['Fsbcc2'])/3 + (4*Fsbcc['Fsbcc4'])/3 + (2*Fsbdd['Fsbdd2'])/9 - (8*Fsbdd['Fsbdd4'])/9 + (2*Fsbss['Fsbss2'])/9 - (8*Fsbss['Fsbss4'])/9 + (4*Fsbuu['Fsbuu4'])/3,
 'EOSsbqq5Q' : -Fsbbb['Fsbbb2']/108 + Fsbbb['Fsbbb3']/36 + Fsbbb['Fsbbb4']/108 + Fsbcc['Fsbcc1']/12 + Fsbcc['Fsbcc2']/36 - Fsbcc['Fsbcc3']/24 - Fsbcc['Fsbcc4']/72 - Fsbdd['Fsbdd1']/24 - Fsbdd['Fsbdd2']/108 + Fsbdd['Fsbdd3']/36 + Fsbdd['Fsbdd4']/108 - Fsbss['Fsbss1']/24 - Fsbss['Fsbss2']/108 + Fsbss['Fsbss3']/36 + Fsbss['Fsbss4']/108 - Fsbuu['Fsbuu3']/24 - Fsbuu['Fsbuu4']/72,
 'EOSsbqq6Q' : -Fsbbb['Fsbbb2']/18 + Fsbbb['Fsbbb4']/18 + Fsbcc['Fsbcc2']/6 - Fsbcc['Fsbcc4']/12 - Fsbdd['Fsbdd2']/18 + Fsbdd['Fsbdd4']/18 - Fsbss['Fsbss2']/18 + Fsbss['Fsbss4']/18 - Fsbuu['Fsbuu4']/12,
 'EOSsbqqb' : Fsbbb['Fsbbb1'] - Fsbdd['Fsbdd1']/2 - Fsbss['Fsbss1']/2}


def EOSpsbqq (Fpsbuu,Fpsbdd,Fpsbcc,Fpsbss,Fpsbbb):
    return {'EOSsbqq1p' : -2*Fpsbcc['Fsbcc1p'] + 2*Fpsbuu['Fsbuu1p'],
 'EOSsbqq2p' : -Fpsbcc['Fsbcc1p']/3 - Fpsbcc['Fsbcc2p'] + Fpsbuu['Fsbuu1p']/3 + Fpsbuu['Fsbuu2p'],
 'EOSsbqq3p' : (-2*Fpsbbb['Fsbbb2p'])/81 + (8*Fpsbbb['Fsbbb3p'])/27 + (8*Fpsbbb['Fsbbb4p'])/81 - Fpsbcc['Fsbcc1p']/9 - Fpsbcc['Fsbcc2p']/27 + (2*Fpsbcc['Fsbcc3p'])/9 + (2*Fpsbcc['Fsbcc4p'])/27 - Fpsbdd['Fsbdd1p']/9 - (2*Fpsbdd['Fsbdd2p'])/81 + (8*Fpsbdd['Fsbdd3p'])/27 + (8*Fpsbdd['Fsbdd4p'])/81 - Fpsbss['Fsbss1p']/9 - (2*Fpsbss['Fsbss2p'])/81 + (8*Fpsbss['Fsbss3p'])/27 + (8*Fpsbss['Fsbss4p'])/81 + (2*Fpsbuu['Fsbuu3p'])/9 + (2*Fpsbuu['Fsbuu4p'])/27,
 'EOSsbqq4p' : (-4*Fpsbbb['Fsbbb2p'])/27 + (16*Fpsbbb['Fsbbb4p'])/27 - (2*Fpsbcc['Fsbcc2p'])/9 + (4*Fpsbcc['Fsbcc4p'])/9 - (4*Fpsbdd['Fsbdd2p'])/27 + (16*Fpsbdd['Fsbdd4p'])/27 - (4*Fpsbss['Fsbss2p'])/27 + (16*Fpsbss['Fsbss4p'])/27 + (4*Fpsbuu['Fsbuu4p'])/9,
 'EOSsbqq5p' : Fpsbbb['Fsbbb2p']/162 - Fpsbbb['Fsbbb3p']/54 - Fpsbbb['Fsbbb4p']/162 + Fpsbcc['Fsbcc1p']/36 + Fpsbcc['Fsbcc2p']/108 - Fpsbcc['Fsbcc3p']/72 - Fpsbcc['Fsbcc4p']/216 + Fpsbdd['Fsbdd1p']/36 + Fpsbdd['Fsbdd2p']/162 - Fpsbdd['Fsbdd3p']/54 - Fpsbdd['Fsbdd4p']/162 + Fpsbss['Fsbss1p']/36 + Fpsbss['Fsbss2p']/162 - Fpsbss['Fsbss3p']/54 - Fpsbss['Fsbss4p']/162 - Fpsbuu['Fsbuu3p']/72 - Fpsbuu['Fsbuu4p']/216,
 'EOSsbqq6p' : Fpsbbb['Fsbbb2p']/27 - Fpsbbb['Fsbbb4p']/27 + Fpsbcc['Fsbcc2p']/18 - Fpsbcc['Fsbcc4p']/36 + Fpsbdd['Fsbdd2p']/27 - Fpsbdd['Fsbdd4p']/27 + Fpsbss['Fsbss2p']/27 - Fpsbss['Fsbss4p']/27 - Fpsbuu['Fsbuu4p']/36,
 'EOSsbqq7p' : Fpsbbb['Fsbbb2p']/27 - (4*Fpsbbb['Fsbbb3p'])/9 - (4*Fpsbbb['Fsbbb4p'])/27 - Fpsbcc['Fsbcc1p']/3 - Fpsbcc['Fsbcc2p']/9 + (2*Fpsbcc['Fsbcc3p'])/3 + (2*Fpsbcc['Fsbcc4p'])/9 + Fpsbdd['Fsbdd1p']/6 + Fpsbdd['Fsbdd2p']/27 - (4*Fpsbdd['Fsbdd3p'])/9 - (4*Fpsbdd['Fsbdd4p'])/27 + Fpsbss['Fsbss1p']/6 + Fpsbss['Fsbss2p']/27 - (4*Fpsbss['Fsbss3p'])/9 - (4*Fpsbss['Fsbss4p'])/27 + (2*Fpsbuu['Fsbuu3p'])/3 + (2*Fpsbuu['Fsbuu4p'])/9,
 'EOSsbqq8p' : (2*Fpsbbb['Fsbbb2p'])/9 - (8*Fpsbbb['Fsbbb4p'])/9 - (2*Fpsbcc['Fsbcc2p'])/3 + (4*Fpsbcc['Fsbcc4p'])/3 + (2*Fpsbdd['Fsbdd2p'])/9 - (8*Fpsbdd['Fsbdd4p'])/9 + (2*Fpsbss['Fsbss2p'])/9 - (8*Fpsbss['Fsbss4p'])/9 + (4*Fpsbuu['Fsbuu4p'])/3,
 'EOSsbqq9p' : -Fpsbbb['Fsbbb2p']/108 + Fpsbbb['Fsbbb3p']/36 + Fpsbbb['Fsbbb4p']/108 + Fpsbcc['Fsbcc1p']/12 + Fpsbcc['Fsbcc2p']/36 - Fpsbcc['Fsbcc3p']/24 - Fpsbcc['Fsbcc4p']/72 - Fpsbdd['Fsbdd1p']/24 - Fpsbdd['Fsbdd2p']/108 + Fpsbdd['Fsbdd3p']/36 + Fpsbdd['Fsbdd4p']/108 - Fpsbss['Fsbss1p']/24 - Fpsbss['Fsbss2p']/108 + Fpsbss['Fsbss3p']/36 + Fpsbss['Fsbss4p']/108 - Fpsbuu['Fsbuu3p']/24 - Fpsbuu['Fsbuu4p']/72,
 'EOSsbqq10p' : -Fpsbbb['Fsbbb2p']/18 + Fpsbbb['Fsbbb4p']/18 + Fpsbcc['Fsbcc2p']/6 - Fpsbcc['Fsbcc4p']/12 - Fpsbdd['Fsbdd2p']/18 + Fpsbdd['Fsbdd4p']/18 - Fpsbss['Fsbss2p']/18 + Fpsbss['Fsbss4p']/18 - Fpsbuu['Fsbuu4p']/12,
 'EOSsbqq11p' : Fpsbbb['Fsbbb1p'] - Fpsbdd['Fsbdd1p']/2 - Fpsbss['Fsbss1p']/2}


# semileptonic operators sbllp

def Fsbllp (C):
    return {
"F9sbllp": C["VdeLR"][1,2,:,:]/2 + C["VedLL"][:,:,1,2]/2,
"F10sbllp": C["VdeLR"][1,2,:,:]/2 - C["VedLL"][:,:,1,2]/2,
"FSsbllp": np.swapaxes(C["SedRL"], 0, 1)[:,:,2,1].conjugate()/2 + C["SedRR"][:,:,1,2]/2,
"FPsbllp": -np.swapaxes(C["SedRL"], 0, 1)[:,:,2,1].conjugate()/2 + C["SedRR"][:,:,1,2]/2,
"FTsbllp": C["TedRR"][:,:,1,2]/2 + np.swapaxes(C["TedRR"], 0, 1)[:,:,2,1].conjugate()/2,
"FT5sbllp": C["TedRR"][:,:,1,2]/2 - np.swapaxes(C["TedRR"], 0, 1)[:,:,2,1].conjugate()/2,
"F9psbllp": C["VedLR"][:,:,1,2]/2 + C["VedRR"][:,:,1,2]/2,
"F10psbllp": -C["VedLR"][:,:,1,2]/2 + C["VedRR"][:,:,1,2]/2,
"FSpsbllp": C["SedRL"][:,:,1,2]/2 + np.swapaxes(C["SedRR"], 0, 1)[:,:,2,1].conjugate()/2,
"FPpsbllp": C["SedRL"][:,:,1,2]/2 - np.swapaxes(C["SedRR"], 0, 1)[:,:,2,1].conjugate()/2,
"Fnusbllp": C["VnudLL"][:,:,1,2],
"Fnupsbllp": C["VnudLR"][:,:,1,2]}


def Bernsbllp (Fsbllp):
    return {"1sbllp": (5*Fsbllp["F10sbllp"])/3 + Fsbllp["F9sbllp"],
"3sbllp": -Fsbllp["F10sbllp"]/6,
"5sbllp": (-5*Fsbllp["FPsbllp"])/3 + Fsbllp["FSsbllp"],
"7sbllp": (2*Fsbllp["FPsbllp"])/3 + Fsbllp["FT5sbllp"] + Fsbllp["FTsbllp"],
"9sbllp": Fsbllp["FPsbllp"]/24,
"1psbllp": (-5*Fsbllp["F10psbllp"])/3 + Fsbllp["F9psbllp"],
"3psbllp": Fsbllp["F10psbllp"]/6,
"5psbllp": (5*Fsbllp["FPpsbllp"])/3 + Fsbllp["FSpsbllp"],
"7psbllp": (-2*Fsbllp["FPpsbllp"])/3 - Fsbllp["FT5sbllp"] + Fsbllp["FTsbllp"],
"9psbllp": -Fsbllp["FPpsbllp"]/24,
"nu1sbllp": Fsbllp["Fnusbllp"],
"nu1psbllp": Fsbllp["Fnupsbllp"]}


def Flaviosbllp (Fsbllp):
    return {
"C9_bs": (16*pi**2)/e**2*Fsbllp["F9sbllp"],
"C9p_bs": (16*pi**2)/e**2*Fsbllp["F9psbllp"],
"C10_bs": (16*pi**2)/e**2*Fsbllp["F10sbllp"],
"C10p_bs": (16*pi**2)/e**2*Fsbllp["F10psbllp"],
"CS_bs": (16*pi**2)/e**2/mb*Fsbllp["FSsbllp"],
"CSp_bs": (16*pi**2)/e**2/mb*Fsbllp["FSpsbllp"],
"CP_bs": (16*pi**2)/e**2/mb*Fsbllp["FPsbllp"],
"CPp_bs": (16*pi**2)/e**2/mb*Fsbllp["FPpsbllp"],
"CL_bs": (8*pi**2)/e**2*Fsbllp["Fnusbllp"],
"CR_bs": (8*pi**2)/e**2*Fsbllp["Fnupsbllp"]
}

def EOSsbllp (Fsbllp):
    return {
"C9_bs": (16*pi**2)/e**2*Fsbllp["F9sbllp"],
"C9p_bs": (16*pi**2)/e**2*Fsbllp["F9psbllp"],
"C10_bs": (16*pi**2)/e**2*Fsbllp["F10sbllp"],
"C10p_bs": (16*pi**2)/e**2*Fsbllp["F10psbllp"],
"CS_bs": (16*pi**2)/e**2*Fsbllp["FSsbllp"],
"CSp_bs": (16*pi**2)/e**2*Fsbllp["FSpsbllp"],
"CP_bs": (16*pi**2)/e**2**Fsbllp["FPsbllp"],
"CPp_bs": (16*pi**2)/e**2*Fsbllp["FPpsbllp"],
"CT_bs": (16*pi**2)/e**2*Fsbllp["FTsbllp"],
"CT5_bs": (16*pi**2)/e**2*Fsbllp["FT5sbllp"]
}


# chromomagnetic operators sbF,
# sbG,

def Fchrombs (C):
    return {
"F7bsgamma": C['dgamma'][1,2],
"F8bsg": C['dG'][1,2],
"F7pbsgamma": C['dgamma'][2,1].conjugate(),
"F8pbsg": C['dG'][2,1].conjugate()
 }


def Bernchrombs (Fchrombs):
    return {
"7gammasb": (gs**2)/e/mb*Fchrombs['F7bsgamma'],
"8gsb": gs/mb*Fchrombs['F8bsg'],
"7pgammasb": (gs**2)/e/mb*Fchrombs['F7pbsgamma'],
"8pgsb": gs/mb*Fchrombs['F8pbsg']
}


def Flaviochrombs (Fchrombs):
    return {
"C7_bs": (16*pi**2)/e/mb*Fchrombs['F7bsgamma'],
"C8_bs": (16*pi**2)/gs/mb*Fchrombs['F8bsg'],
"C7p_bs": (16*pi**2)/e/mb*Fchrombs['F7pbsgamma'],
"C8p_bs": (16*pi**2)/gs/mb*Fchrombs['F8pbsg']
}

def EOSchrombs (Fchrombs):
    return {
"C7_bs": (gs**2/e)/(mb**2+ms**2)*(mb*Fchrombs['F7bsgamma']+ms*Fchrombs['F7pbsgamma']),
"C8_bs": gs/(mb**2+ms**2)*(mb*Fchrombs['F8bsg']+ms*Fchrombs['F8pbsg'])
}

# chromomagnetic operators dbF,
# dbG,

def Fchrombd (C):
    return {
"F7bdgamma": C['dgamma'][0,2],
"F8bdg": C['dG'][0,2],
"F7pbdgamma": C['dgamma'][2,0].conjugate(),
"F8pbdg": C['dG'][2,0].conjugate()
 }

def Bernchrombd (Fchrombd):
    return {
"7gammadb": (gs**2)/e/mb*Fchrombd['F7bdgamma'],
"8gdb": gs/mb*Fchrombd['F8bdg'],
"7pgammadb": (gs**2)/e/mb*Fchrombd['F7pbdgamma'],
"8pgdb": gs/mb*Fchrombd['F8pbdg']
}

def Flaviochrombd (Fchrombd):
    return {
"C7_bd": (16*pi**2)/e/mb*Fchrombd['F7bdgamma'],
"C8_bd": (16*pi**2)/gs/mb*Fchrombd['F8bdg'],
"C7p_bd": (16*pi**2)/e/mb*Fchrombd['F7pbdgamma'],
"C8p_bd": (16*pi**2)/gs/mb*Fchrombd['F8pbdg']
}

def EOSchrombd (Fchrombd):
    return {
"C7_bd": (gs**2/e)/(mb**2+ms**2)*(mb*Fchrombd['F7bdgamma']+ms*Fchrombd['F7pbdgamma']),
"C8_bd": gs/(mb**2+ms**2)*(mb*Fchrombd['F8bdg']+ms*Fchrombd['F8pbdg'])
}



# Class I

def _JMS_to_SUSY(C):
    d = {}
    d.update(SUSYsbsb(C))
    d.update(SUSYdbdb(C))
    return d

def _SUSY_to_Flavio(C):
    d = {}
    d.update(Flaviosbsb(C))
    d.update(Flaviodbdb(C))
    return d

# Class II

def _JMS_to_Bern(C):
    d = {}
    d.update(Bernublnu(C))
    d.update(Berncblnu(C))
    return d

def _Bern_to_ACFG(C):
    d = {}
    d.update(ACFGublnu(C))
    d.update(ACFGcblnu(C))
    return d

def _Bern_to_Flavio(C):
    d = {}
    d.update(Flavioublnu(C))
    d.update(Flaviocblnu(C))
    return d

def _Fierz_to_Bern(C):
    d = {}
    d.update(Bernsbuc(C))
    d.update(Bernpsbuc(C))
    d.update(Bernsbcu(C))
    d.update(Bernpsbcu(C))
    d.update(Berndbuc(C))
    d.update(Bernpdbuc(C))
    d.update(Berndbcu(C))
    d.update(Bernpdbcu(C))
    d.update(Bernsbsd(C))
    d.update(Bernpsbsd(C))
    # d.update(Berndbsd(C))
    # d.update(Bernpdbsd(C))
    # d.update(Berndbsb(C))
    # d.update(Bernpdbsb(C))
    d.update(Bernsbuu(C))
    d.update(Bernpsbuu(C))
    d.update(Bernsbcc(C))
    d.update(Bernpsbcc(C))
    d.update(Bernsbdd(C))
    d.update(Bernpsbdd(C))
    d.update(Bernsbss(C))
    d.update(Bernpsbss(C))
    d.update(Bernsbbb(C))
    d.update(Bernpsbbb(C))
    d.update(Bernchrombs(C))
    d.update(Berndbcc(C))
    d.update(Berndbdd(C))
    d.update(Berndbds(C))
    d.update(Berndbss(C))
    d.update(Berndbuu(C))
    d.update(Bernpdbcc(C))
    d.update(Bernpdbdd(C))
    d.update(Bernpdbds(C))
    d.update(Bernpdbss(C))
    d.update(Bernpdbuu(C))
    return d

def _Fierz_to_Bern_bsll(C):
    return Bernsbllp(C)

def _JMS_to_Fierz(C):
    d = {}
    d.update(Fsbuu(C))
    d.update(Fpsbuu(C))
    d.update(Fdbuu(C))
    d.update(Fpdbuu(C))
    d.update(Fsbdd(C))
    d.update(Fpsbdd(C))
    d.update(Fdbdd(C))
    d.update(Fpdbdd(C))
    d.update(Fsbllp(C))
    d.update(Fchrombs(C))
    d.update(Fdbbb(C))
    d.update(Fdbcc(C))
    d.update(Fdbcu(C))
    d.update(Fdbds(C))
    d.update(Fdbss(C))
    d.update(Fdbuc(C))
    d.update(Fpdbbb(C))
    d.update(Fpdbcc(C))
    d.update(Fpdbcu(C))
    d.update(Fpdbds(C))
    d.update(Fpdbss(C))
    d.update(Fpdbuc(C))
    d.update(Fpsbbb(C))
    d.update(Fpsbcc(C))
    d.update(Fpsbcu(C))
    d.update(Fpsbsd(C))
    d.update(Fpsbss(C))
    d.update(Fpsbuc(C))
    d.update(Fsbbb(C))
    d.update(Fsbcc(C))
    d.update(Fsbcu(C))
    d.update(Fsbsd(C))
    d.update(Fsbss(C))
    d.update(Fsbuc(C))
    return d

def _Fierz_to_Buras(C):
    d = {}
    d.update(Burassbqq(C))
    d.update(Buraspsbqq(C))
    d.update(Burasdbqq(C))
    d.update(Buraspdbqq(C))
    return d

def _Fierz_to_BGHW(C):
    d = {}
    d.update(BGHWsbuu(C))
    d.update(BGHWpsbuu(C))
    d.update(BGHWsbcc(C))
    d.update(BGHWpsbcc(C))
    d.update(BGHWsbdd(C))
    d.update(BGHWpsbdd(C))
    d.update(BGHWsbss(C))
    d.update(BGHWpsbss(C))
    d.update(BGHWsbbb(C))
    d.update(BGHWpsbbb(C))
    return d

def _Fierz_to_DeltaF1(C):
    d = {}
    d.update(DF1sbqq(C))
    d.update(DF1psbqq(C))
    d.update(DF1dbqq(C))
    d.update(DF1pdbqq(C))
    return d

def _Fierz_to_Flavio(C):
    d = {}
    d.update(Flaviosbllp(C))
    d.update(Flaviochrombs(C))
    return d

def _JMS_to_array(C):
    """For a dictionary with JMS Wilson coefficients, return an dictionary
    of arrays."""
    Ca = _scalar2array(C)
    for k in Ca:
        if k in ["VedLL", "VedLR", "VdeLR", "VedRR", "VnudLL", "VnudLR"]:
            Ca[k] = _symm_herm(Ca[k])
        if k in ["VddLL", "VddRR"]:
            Ca[k] = _symm_current(Ca[k])
    return Ca

# Combined translators

def JMS_to_flavio(C):
    Ca = _JMS_to_array(C)
    d = {}
    SUSY = _JMS_to_SUSY(Ca)
    d.update(_SUSY_to_Flavio(SUSY))
    Fierz = _JMS_to_Fierz(Ca)
    d.update(_Fierz_to_Flavio(Fierz))
    # transition from tensor-valued dictionary to flat dictionary
    l = ['e', 'mu', 'tau']
    nu = ['nue', 'numu', 'nutau']
    for op in ['C9', 'C10', 'CS', 'CP']:
        for prime in ['', 'p']:
            for bq in ['_bs']:
                label = op + prime + bq
                for i, l1 in enumerate(l):
                    for j, l2 in enumerate(l):
                        d[label + l1 + l2] = d[label][i, j]
                del d[label]
    for op in ['CL', 'CR']:
        for bq in ['_bs']:
            label = op + bq
            for i, l1 in enumerate(nu):
                for j, l2 in enumerate(nu):
                    d[label + l1 + l2] = d[label][i, j]
            del d[label]
    Bern_bqlnu = _JMS_to_Bern(Ca)
    d_bqlnu = _Bern_to_Flavio(Bern_bqlnu)
    for k, v in d_bqlnu.items():
        for i, l1 in enumerate(l):
            for j, l2 in enumerate(l):
                label = k.replace('lnu', l1 + 'nu' + l2)
                d[label] = v[i, j]
    return d

def JMS_to_Bern(C):
    Ca = _JMS_to_array(C)
    d = _JMS_to_SUSY(Ca)
    Fierz = _JMS_to_Fierz(Ca)
    d.update(_Fierz_to_Bern(Fierz))
    d_bsll = _Fierz_to_Bern_bsll(Fierz)
    l = ['e', 'mu', 'tau']
    for k, v in d_bsll.items():
        for i, l1 in enumerate(l):
            for j, l2 in enumerate(l):
                label = k.replace('llp', l1 + l2)
                d[label] = v[i, j]
    d_bqlnu = _JMS_to_Bern(Ca)
    for k, v in d_bqlnu.items():
        for i, l1 in enumerate(l):
            for j, l2 in enumerate(l):
                label = k.replace('llp', l1 + l2)
                d[label] = v[i, j]
    return d
