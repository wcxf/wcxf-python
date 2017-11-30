import wcxf
from . import smeft

@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'JMS')
def warsaw_to_jms(C, parameters):
    return smeft.match_all(C, parameters)
