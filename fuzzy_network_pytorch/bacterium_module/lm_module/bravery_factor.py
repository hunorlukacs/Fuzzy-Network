def bravery_factor(gamma, r):
  '''
    Parameters:
    -----------
    gamma: scalar
    r: scalar

    Returns:
    ---------
      gamma: scalar
            Bravery factor
  '''
  # bravery_factor
  # gamma
  if r < 0.25:
    return 4*gamma
  elif r > 0.75:
    return gamma/2
  else:
    return gamma