from . import smeft
from . import wet
import wcxf


@wcxf.translator('SMEFT', 'Warsaw', 'Warsaw mass')
def warsaw_to_warsawmass(C, parameters):
    return smeft.warsaw_to_warsawmass(C)


@wcxf.translator('SMEFT', 'Warsaw', 'Warsaw up')
def warsaw_to_warsaw_up(C, parameters):
    return smeft.warsaw_to_warsaw_up(C)


@wcxf.translator('SMEFT', 'Warsaw up', 'Warsaw')
def warsaw_up_to_warsaw(C, parameters):
    return smeft.warsaw_up_to_warsaw(C)


@wcxf.translator('WET', 'JMS', 'flavio')
def JMS_to_flavio(C, parameters):
    return wet.JMS_to_flavio(C, parameters)


@wcxf.translator('WET', 'Bern', 'flavio')
def Bern_to_flavio(C, parameters):
    return wet.Bern_to_flavio(C, parameters)


@wcxf.translator('WET', 'flavio', 'Bern')
def flavio_to_Bern(C, parameters):
    return wet.flavio_to_Bern(C, parameters)


@wcxf.translator('WET', 'JMS', 'EOS')
def JMS_to_EOS(C, parameters):
    return wet.JMS_to_EOS(C, parameters)


@wcxf.translator('WET', 'JMS', 'Bern')
def JMS_to_Bern(C, parameters):
    return wet.JMS_to_Bern(C, parameters)


@wcxf.translator('WET', 'JMS', 'formflavor')
def JMS_to_FormFlavor(C, parameters):
    return wet.JMS_to_FormFlavor(C, parameters)


@wcxf.translator('WET', 'FlavorKit', 'JMS')
def FlavorKit_to_JMS(C, parameters):
    return wet.FlavorKit_to_JMS(C, parameters)


@wcxf.translator('WET', 'JMS', 'FlavorKit')
def JMS_to_FlavorKit(C, parameters):
    return wet.JMS_to_FlavorKit(C, parameters)
