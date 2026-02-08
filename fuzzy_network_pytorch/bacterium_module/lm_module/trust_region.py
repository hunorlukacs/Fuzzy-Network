

def trust_region(grad_vec, f_obj_updated, f_obj_old, update_vec):
  denominator = grad_vec.T @ update_vec
  if denominator == 0:
    denominator = 10
  r = (f_obj_updated - f_obj_old) / denominator
  return r