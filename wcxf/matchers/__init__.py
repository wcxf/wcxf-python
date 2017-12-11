import wcxf
from . import smeft


@wcxf.matcher('SMEFT', 'Warsaw up', 'WET', 'JMS')
def warsaw_up_to_jms(C, parameters):
    return smeft.match_all(C, parameters)


@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'JMS')
def warsaw_to_jms(C, parameters):
    C_warsawup = wcxf.translators.smeft.warsaw_to_warsaw_up(C)
    return smeft.match_all(C_warsawup, parameters)
