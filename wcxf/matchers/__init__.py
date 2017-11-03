import wcxf
from . import smeft

@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'JMS')
def warsaw_to_jms(C):
    return smeft.match_all(C)
