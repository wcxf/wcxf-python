
# elements that are redundant and can thus be omitted in the input/output
redundant_elements = {'G': [],
 'Gd': [],
 'Ge': [],
 'Gtilde': [],
 'Gu': [],
 'Lambda': [],
 'Theta': [],
 'Thetap': [],
 'Thetas': [],
 'W': [],
 'Wtilde': [],
 'dB': [],
 'dG': [],
 'dW': [],
 'dd': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (0, 1, 0, 0),
  (0, 2, 0, 0),
  (0, 2, 0, 1),
  (0, 2, 1, 0),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 0, 1),
  (1, 1, 0, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (1, 2, 0, 0),
  (1, 2, 0, 1),
  (1, 2, 0, 2),
  (1, 2, 1, 0),
  (1, 2, 1, 1),
  (1, 2, 2, 0),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 0, 0),
  (2, 2, 0, 1),
  (2, 2, 0, 2),
  (2, 2, 1, 0),
  (2, 2, 1, 1),
  (2, 2, 1, 2),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'dphi': [],
 'duql': [],
 'duue': [],
 'eB': [],
 'eW': [],
 'ed': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'ee': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (0, 1, 0, 0),
  (0, 1, 1, 0),
  (0, 1, 2, 0),
  (0, 2, 0, 0),
  (0, 2, 0, 1),
  (0, 2, 1, 0),
  (0, 2, 1, 1),
  (0, 2, 2, 0),
  (0, 2, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 0, 1),
  (1, 1, 0, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (1, 2, 0, 0),
  (1, 2, 0, 1),
  (1, 2, 0, 2),
  (1, 2, 1, 0),
  (1, 2, 1, 1),
  (1, 2, 2, 0),
  (1, 2, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 0, 0),
  (2, 2, 0, 1),
  (2, 2, 0, 2),
  (2, 2, 1, 0),
  (2, 2, 1, 1),
  (2, 2, 1, 2),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'ephi': [],
 'eu': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'g': [],
 'gp': [],
 'gs': [],
 'ld': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'le': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'ledq': [],
 'lequ1': [],
 'lequ3': [],
 'll': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (0, 1, 0, 0),
  (0, 2, 0, 0),
  (0, 2, 0, 1),
  (0, 2, 1, 0),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 0, 1),
  (1, 1, 0, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (1, 2, 0, 0),
  (1, 2, 0, 1),
  (1, 2, 0, 2),
  (1, 2, 1, 0),
  (1, 2, 1, 1),
  (1, 2, 2, 0),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 0, 0),
  (2, 2, 0, 1),
  (2, 2, 0, 2),
  (2, 2, 1, 0),
  (2, 2, 1, 1),
  (2, 2, 1, 2),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'llphiphi': [(1, 0), (2, 0), (2, 1)],
 'lq1': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'lq3': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'lu': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'm2': [],
 'phi': [],
 'phiB': [],
 'phiBox': [],
 'phiBtilde': [],
 'phiD': [],
 'phiG': [],
 'phiGtilde': [],
 'phiW': [],
 'phiWB': [],
 'phiWtilde': [],
 'phiWtildeB': [],
 'phid': [(1, 0), (2, 0), (2, 1)],
 'phie': [(1, 0), (2, 0), (2, 1)],
 'phil1': [(1, 0), (2, 0), (2, 1)],
 'phil3': [(1, 0), (2, 0), (2, 1)],
 'phiq1': [(1, 0), (2, 0), (2, 1)],
 'phiq3': [(1, 0), (2, 0), (2, 1)],
 'phiu': [(1, 0), (2, 0), (2, 1)],
 'phiud': [],
 'qd1': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'qd8': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'qe': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'qq1': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (0, 1, 0, 0),
  (0, 2, 0, 0),
  (0, 2, 0, 1),
  (0, 2, 1, 0),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 0, 1),
  (1, 1, 0, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (1, 2, 0, 0),
  (1, 2, 0, 1),
  (1, 2, 0, 2),
  (1, 2, 1, 0),
  (1, 2, 1, 1),
  (1, 2, 2, 0),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 0, 0),
  (2, 2, 0, 1),
  (2, 2, 0, 2),
  (2, 2, 1, 0),
  (2, 2, 1, 1),
  (2, 2, 1, 2),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'qq3': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (0, 1, 0, 0),
  (0, 2, 0, 0),
  (0, 2, 0, 1),
  (0, 2, 1, 0),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 0, 1),
  (1, 1, 0, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (1, 2, 0, 0),
  (1, 2, 0, 1),
  (1, 2, 0, 2),
  (1, 2, 1, 0),
  (1, 2, 1, 1),
  (1, 2, 2, 0),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 0, 0),
  (2, 2, 0, 1),
  (2, 2, 0, 2),
  (2, 2, 1, 0),
  (2, 2, 1, 1),
  (2, 2, 1, 2),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'qqql': [(1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 1, 0, 0),
  (1, 1, 0, 1),
  (1, 1, 0, 2),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 2, 0, 0),
  (2, 2, 0, 1),
  (2, 2, 0, 2),
  (2, 2, 1, 0),
  (2, 2, 1, 1),
  (2, 2, 1, 2)],
 'qque': [(1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2)],
 'qu1': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'qu8': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'quqd1': [],
 'quqd8': [],
 'uB': [],
 'uG': [],
 'uW': [],
 'ud1': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'ud8': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 1, 0),
  (2, 2, 2, 0),
  (2, 2, 2, 1)],
 'uphi': [],
 'uu': [(0, 0, 1, 0),
  (0, 0, 2, 0),
  (0, 0, 2, 1),
  (0, 1, 0, 0),
  (0, 2, 0, 0),
  (0, 2, 0, 1),
  (0, 2, 1, 0),
  (1, 0, 0, 0),
  (1, 0, 0, 1),
  (1, 0, 0, 2),
  (1, 0, 1, 0),
  (1, 0, 1, 1),
  (1, 0, 1, 2),
  (1, 0, 2, 0),
  (1, 0, 2, 1),
  (1, 0, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 0, 1),
  (1, 1, 0, 2),
  (1, 1, 1, 0),
  (1, 1, 2, 0),
  (1, 1, 2, 1),
  (1, 2, 0, 0),
  (1, 2, 0, 1),
  (1, 2, 0, 2),
  (1, 2, 1, 0),
  (1, 2, 1, 1),
  (1, 2, 2, 0),
  (2, 0, 0, 0),
  (2, 0, 0, 1),
  (2, 0, 0, 2),
  (2, 0, 1, 0),
  (2, 0, 1, 1),
  (2, 0, 1, 2),
  (2, 0, 2, 0),
  (2, 0, 2, 1),
  (2, 0, 2, 2),
  (2, 1, 0, 0),
  (2, 1, 0, 1),
  (2, 1, 0, 2),
  (2, 1, 1, 0),
  (2, 1, 1, 1),
  (2, 1, 1, 2),
  (2, 1, 2, 0),
  (2, 1, 2, 1),
  (2, 1, 2, 2),
  (2, 2, 0, 0),
  (2, 2, 0, 1),
  (2, 2, 0, 2),
  (2, 2, 1, 0),
  (2, 2, 1, 1),
  (2, 2, 1, 2),
  (2, 2, 2, 0),
  (2, 2, 2, 1)]}

# elements where the imaginary part must be zero and which can thus
# be omitted in the input/output
vanishing_im_parts = {'G': [],
 'Gd': [],
 'Ge': [],
 'Gtilde': [],
 'Gu': [],
 'Lambda': [],
 'Theta': [],
 'Thetap': [],
 'Thetas': [],
 'W': [],
 'Wtilde': [],
 'dB': [],
 'dG': [],
 'dW': [],
 'dd': [(1, 1, 2, 2),
  (2, 2, 2, 2),
  (1, 2, 2, 1),
  (1, 1, 1, 1),
  (0, 2, 2, 0),
  (0, 1, 1, 0),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2)],
 'dphi': [],
 'duql': [],
 'duue': [],
 'eB': [],
 'eW': [],
 'ed': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'ee': [(1, 1, 2, 2),
  (2, 2, 2, 2),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2)],
 'ephi': [],
 'eu': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'g': [],
 'gp': [],
 'gs': [],
 'ld': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'le': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'ledq': [],
 'lequ1': [],
 'lequ3': [],
 'll': [(1, 1, 2, 2),
  (2, 2, 2, 2),
  (1, 2, 2, 1),
  (1, 1, 1, 1),
  (0, 2, 2, 0),
  (0, 1, 1, 0),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2)],
 'llphiphi': [],
 'lq1': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'lq3': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'lu': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'm2': [],
 'phi': [],
 'phiB': [],
 'phiBox': [],
 'phiBtilde': [],
 'phiD': [],
 'phiG': [],
 'phiGtilde': [],
 'phiW': [],
 'phiWB': [],
 'phiWtilde': [],
 'phiWtildeB': [],
 'phid': [(0, 0), (1, 1), (2, 2)],
 'phie': [(0, 0), (1, 1), (2, 2)],
 'phil1': [(0, 0), (1, 1), (2, 2)],
 'phil3': [(0, 0), (1, 1), (2, 2)],
 'phiq1': [(0, 0), (1, 1), (2, 2)],
 'phiq3': [(0, 0), (1, 1), (2, 2)],
 'phiu': [(0, 0), (1, 1), (2, 2)],
 'phiud': [],
 'qd1': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'qd8': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'qe': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'qq1': [(1, 1, 2, 2),
  (2, 2, 2, 2),
  (1, 2, 2, 1),
  (1, 1, 1, 1),
  (0, 2, 2, 0),
  (0, 1, 1, 0),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2)],
 'qq3': [(1, 1, 2, 2),
  (2, 2, 2, 2),
  (1, 2, 2, 1),
  (1, 1, 1, 1),
  (0, 2, 2, 0),
  (0, 1, 1, 0),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2)],
 'qqql': [],
 'qque': [],
 'qu1': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'qu8': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'quqd1': [],
 'quqd8': [],
 'uB': [],
 'uG': [],
 'uW': [],
 'ud1': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'ud8': [(1, 1, 2, 2),
  (2, 2, 1, 1),
  (2, 2, 2, 2),
  (1, 1, 0, 0),
  (1, 1, 1, 1),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2),
  (2, 2, 0, 0)],
 'uphi': [],
 'uu': [(1, 1, 2, 2),
  (2, 2, 2, 2),
  (1, 2, 2, 1),
  (1, 1, 1, 1),
  (0, 2, 2, 0),
  (0, 1, 1, 0),
  (0, 0, 0, 0),
  (0, 0, 1, 1),
  (0, 0, 2, 2)]}
