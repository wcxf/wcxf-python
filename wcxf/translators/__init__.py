from . import smeftmass
from . import wet
import wcxf


@wcxf.translator('SMEFT', 'Warsaw', 'Warsaw mass')
def warsaw_to_warsawmass(C, parameters):
    return smeftmass.warsaw_to_warsawmass(C)


@wcxf.translator('WET', 'JMS', 'flavio')
def JMS_to_flavio(C, parameters):
    return wet.JMS_to_flavio(C, parameters)


@wcxf.translator('WET', 'JMS', 'EOS')
def JMS_to_EOS(C, parameters):
    return wet.JMS_to_EOS(C, parameters)


@wcxf.translator('WET', 'JMS', 'AFGV')
def JMS_to_Bern(C, parameters):
    return wet.JMS_to_Bern(C, parameters)

@wcxf.translator('WET', 'JMS', 'formflavor')
def JMS_to_FormFlavor(C, parameters):
    return wet.JMS_to_FormFlavor(C, parameters)
