import wcxf
from . import smeft


@wcxf.matcher('SMEFT', 'Warsaw up', 'WET', 'JMS')
def warsaw_up_to_jms(C, parameters):
    return smeft.match_all(C, parameters)


@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'JMS')
def warsaw_to_jms(C, parameters):
    C_warsawup = wcxf.translators.smeft.warsaw_to_warsaw_up(C, parameters)
    return smeft.match_all(C_warsawup, parameters)


@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'flavio')
def warsaw_to_flavio(C, parameters):
    C_warsawup = wcxf.translators.smeft.warsaw_to_warsaw_up(C, parameters)
    C_JMS = smeft.match_all(C_warsawup, parameters)
    return wcxf.translators.JMS_to_flavio(C_JMS, parameters)


@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'EOS')
def warsaw_to_eos(C, parameters):
    C_warsawup = wcxf.translators.smeft.warsaw_to_warsaw_up(C, parameters)
    C_JMS = smeft.match_all(C_warsawup, parameters)
    return wcxf.translators.JMS_to_EOS(C_JMS, parameters)


@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'Bern')
def warsaw_to_bern(C, parameters):
    C_warsawup = wcxf.translators.smeft.warsaw_to_warsaw_up(C, parameters)
    C_JMS = smeft.match_all(C_warsawup, parameters)
    return wcxf.translators.JMS_to_Bern(C_JMS, parameters)
