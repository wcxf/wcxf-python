from . import smeftmass
import wcxf

@wcxf.translator('SMEFT', 'Warsaw', 'Warsaw mass')
def warsaw_to_warsawmass(C):
    return smeftmass.warsaw_to_warsawmass(C)
