import pylha
import json
import pkgutil
from wilson.util import smeftutil


smeftfr_info = json.loads(
    pkgutil.get_data('wcxf', 'data/smeftfr_block_info.json').decode('utf-8')
)


def wcxf2smeftfr_dict(wc):
    """Take a WC instance and turn it into a dictionary suitable for producing
    and LHA file for SmeftFR using pylha."""
    # go from flat dict to dict of arrays, symmetrizing them
    C = smeftutil.wcxf2arrays_symmetrized(wc.dict)
    # go back to flat dict but keeping non-redundant WCs an symm. fac.s
    d = smeftutil.arrays2wcxf(C)
    # rearrange to dict in right format for SmeftFR
    card = {}
    for block_name, block in smeftfr_info.items():
        card[block_name] = {'values': []}
        for info in block:
            line = [*info['index'], d[info['coeff']].real, info['comment']]
            card[block_name]['values'].append(line)
    return card


def wcxf2smeftfr(wc, stream):
    """Take a WC instance and dump it into an LHA file (or return as string)
    in a format suitable for use as `param_card.dat` file for MadGraph
    with SmeftFR
    """
    if (wc.eft, wc.basis) == ('SMEFT', 'Warsaw mass'):
        wc_m = wc
    else:
        if wc.eft != 'SMEFT':
            raise ValueError("wcxf2smeftfr only support SMEFT Wilson coefficients")
        else:
            try:
                wc_m = wc.translate('Warsaw mass')
            except ValueError:
                raise ValueError("wcxf2smeftfr requires a basis that can be translated to the Warsaw mass basis")
    return pylha.dump({'BLOCK': wcxf2smeftfr_dict(wc_m)},
                      fmt='lha', stream=stream)
