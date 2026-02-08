def evaluation(b_old, b_new, f_obj_old, f_obj_updated):
  '''
    Parameters:
    -----------

    f_obj_old, f_obj_updated: scalar

    Returns:
    ---------
    b_next:
        Updated (or original) bacterium
    isUpdated: bool
        If the bacterium is updated
  '''
  f_obj_new = f_obj_old
  isUpdated = False
  if f_obj_updated < f_obj_old:
    b_next = b_new
    isUpdated = True
    f_obj_new = f_obj_updated
  else:
    b_next = b_old
  return b_next, f_obj_new, isUpdated 