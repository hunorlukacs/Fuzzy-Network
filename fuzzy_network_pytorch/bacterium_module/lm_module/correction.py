

import copy
import numpy as np


def correction(pi, pj, Dpi, Dpj):
  '''
    If the pi < pj constraint is comporomised, this method can be applied.
    
    Parameters:
    --------
    pi: scalar
        The "pi" point before any modification applied.
    pj: scalar
        The "pj" point before any modification applied.
    Dpi: scalar
        The modification applied on the pi point. (It can be negative as well)
    Dpj: scalar
        The modification applied on the pj point. (It can be negative as well)

    Returns:
    -------
      (pi_new, pj_new): tuple(scalar, scalar)

    Example:
    --------
    For instance the "b" and "c" breakpoints in a FRB trapezoidal membership function.
     - The breakpoint "b" is pi and breakpoint "c" is pj. pi<pj ("b" < "c") constraint should hold.
     - but after the modification, this constraint would not hold:

    >>> correction(pi=2, pj=2.2, Dpi=.1, Dpj=-0.5)
    >>> # (2.0166666666666666, 2.1166666666666667)
  '''
  print('pi, pj, Dpi, Dpj: ', pi, pj, Dpi, Dpj)
  ### START CODE ###
  if Dpj == Dpi:
    # There was no modification.
    return pi, pj

  g = (pj - pi) / (2 * (Dpj - Dpi))
  print('g:', g)
  pj_next = pj - g * Dpj
  pi_next = pi - g * Dpi
  print('pi_next, pj_next: ', pi_next, pj_next)
  ### END CODE ###
  return pi_next, pj_next

def correction_simple(b, boundaries, PADDING_RATE):
  '''
    Example:
    -------
    >>> b = np.array([[[1.0,2,3,4], [1,3,4,5], [1,2,3,4]],
    >>>           [[1,6,5,4], [4,6,7,5], [6,5,4,3]]])
    >>> boundaries = np.array([[1,3], [1,8], [3,5]])
    >>> PADDING_RATE = 0.3
    >>> print(correction_simple(b=b, boundaries=boundaries, PADDING_RATE=PADDING_RATE))
    >>> # [[[1.  2.  3.  3.6]
    >>> #   [1.  3.  4.  5. ]
    >>> #   [2.4 2.4 3.  4. ]]
    >>> # [[1.  3.6 3.6 3.6]
    >>> #   [4.  5.  6.  7. ]
    >>> #   [3.  4.  5.  5.6]]]
  '''
  b_new = b.copy()
  b_new.sort(axis=2)
  len_bounds = np.array(list(map(lambda x: x[1]-x[0], boundaries)))
  pad = len_bounds * PADDING_RATE
  lower_boundaries = boundaries[:, 0] - pad 
  upper_boundaries = boundaries[:, 1] + pad
  for i in range(len(boundaries)):
    b_tmp = b_new[:, i]
    b_tmp[b_tmp < lower_boundaries[i]] = lower_boundaries[i]
    b_tmp[b_tmp > upper_boundaries[i]] = upper_boundaries[i]
    b_new[:, i] = b_tmp
  return b_new

def frbs_correction_sort(b, s):
  # TODO: if it goes out the boundaries.
  b_new = copy.deepcopy(b)
  b_new = b_new + s

  for i in range(0, len(b)-1, 4):
    b_new[i], b_new[i+1], b_new[i+2], b_new[i+3] = np.sort([b_new[i], b_new[i+1], b_new[i+2], b_new[i+3]])
  return b_new

def frbs_correction(b, s):
  '''
    If fuzzy breakpoint constraints are comporomised, this method can be applied.
  '''
  ### START CODE ###

  b_new = copy.deepcopy(b)

  for i in range(len(b)-1):
    if i!=0 and i % 4 == 3:
      continue
    b_new[i], b_new[i+1] = correction(b_new[i], b_new[i+1], s[i], s[i+1])
    print('b_new-should be updated: ', b_new[i], b_new[i+1])

  return b_new

  ### END CODE ###