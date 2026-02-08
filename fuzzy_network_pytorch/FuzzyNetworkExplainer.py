import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from scipy.interpolate import PchipInterpolator
from typing import Optional, Tuple, List, Any
import torch # Added torch import for tensor operations

# Assuming these are the PyTorch versions of your custom modules
# NOTE: The actual implementation of 'FuzzyNetwork', 'ModelWrapper', and 
# the internal 'trapmf' used by the model must be PyTorch-compatible.
# We explicitly use skfuzzy.trapmf for the analysis functions below.
try:
    from fuzzy_network_pytorch.FuzzyNetwork import FuzzyNetwork
    from fuzzy_network_pytorch.bma import ModelWrapper
    from fuzzy_network_pytorch.FuzzyLayer_Gauss import trapmf
    from fuzzy_network_pytorch import bma
    import fuzzy_network_pytorch.levenberg_marquardt_pytorch as tlm
except ImportError:
    print("Warning: Could not import custom fuzzy_network_pytorch modules.")
    # Define placeholder classes/modules if needed for the script to run standalone
    # For a full conversion, ensure your library imports are correct.
    pass


def calculate_area_below_cut(cut_value: float, breakpoints: np.ndarray) -> float:
    """
    Calculate the area below the cut value for a fuzzy set defined by its breakpoints 
    using the trapezoidal rule (np.trapz) on a densely sampled universe.
    
    This function relies on the `skfuzzy` library for trapezoidal membership functions.

    Args:
        cut_value (float): The cut value for the fuzzy set (w_min).
        breakpoints (np.ndarray): The breakpoints of the trapezoidal membership 
                                  function [a, b, c, d].

    Returns:
        float: The area below the cut.
    """
    # Define the universe of discourse based on the breakpoints range plus margins
    a, d = breakpoints[0], breakpoints[-1]
    
    # Define the universe of discourse with high resolution
    x = np.linspace(a - (d-a)*0.1, d + (d-a)*0.1, 1000)

    # Define the fuzzy set (trapezoidal membership function)
    fuzzy_set = fuzz.trapmf(x, breakpoints)

    # Calculate the area below the cut using numerical integration
    # We take the minimum of the fuzzy set values and the cut value, 
    # then integrate that shape.
    area_below_cut = np.trapz(np.minimum(fuzzy_set, cut_value), x)

    return area_below_cut


def compute_area_for_inputs(single_rule_model: 'bma.ModelWrapper', 
                            x_inputs: np.ndarray, 
                            cons_nr: int = 0) -> Tuple[List[float], List[float]]:
    """
    Compute the significance metric for given x inputs using the single rule model.
    
    - If cons_type == 'trap': Returns (Prediction, Area of truncated trapezoid)
    - If cons_type == 'gauss': Returns (Prediction, Firing strength w)
    """
    # 1. Access Layer Properties
    layer = single_rule_model.model.f_layers[0]
    device = layer.device
    
    ante_type = getattr(layer, 'ante_type', 'trap') # Default to trap if attr missing
    cons_type = getattr(layer, 'cons_type', 'trap')

    # 2. Get Antecedent Parameters (NumPy)
    antecedent_tensors = layer.Antes[0]
    antecedents_np = antecedent_tensors.detach().cpu().numpy()
    
    # Sort parameters only if Trapezoidal (Gaussian params are [mean, sigma], sorting breaks them)
    if ante_type == 'trap':
        antecedents_np = np.sort(antecedents_np, axis=1)

    # 3. Get Consequent Parameters (NumPy)
    consequent_np = layer.Cons[0][0].detach().cpu().numpy()
    if cons_type == 'trap':
        consequent_np = np.sort(consequent_np) # Sort required for skfuzzy.trapmf

    metric_values = []
    predictions = []
    
    # Iterate over each input row
    for x_row in x_inputs:
        # --- A. Model Prediction (PyTorch) ---
        x_tensor = torch.tensor(x_row, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            y_tensor = single_rule_model.model(x_tensor)
        
        predictions.append(y_tensor.squeeze().item()) 

        # --- B. Firing Strength (w_min) Calculation ---
        membership_values = []
        for i in range(x_row.shape[0]):
            params = antecedents_np[i]
            val = x_row[i]

            if ante_type == 'trap':
                # skfuzzy trapmf expects sorted [a,b,c,d]
                mu = fuzz.trapmf(np.array([val]), params)[0]
            elif ante_type == 'gauss':
                # params are [mean, sigma]
                mean, sigma = params[0], abs(params[1])
                # skfuzzy gaussmf: exp(-((x - mean)**2.) / (2 * sigma**2.))
                mu = fuzz.gaussmf(np.array([val]), mean, sigma)[0]
            
            membership_values.append(mu)
        
        # The firing strength (w_min) is the minimum of all membership values
        w_min = np.min(membership_values)
        
        # --- C. Significance Metric Calculation ---
        if cons_type == 'trap':
            # Geometry-based: Area of the shape cut at w_min
            metric = calculate_area_below_cut(w_min, consequent_np)
        else:
            # Gaussian / Height Defuzzification: Significance is simply the firing strength
            metric = w_min

        metric_values.append(metric)

    return predictions, metric_values

def create_single_rule_model(original_model: 'bma.ModelWrapper', 
                             layer_nr: int, 
                             rule_nr: int, 
                             cons_nr: int) -> 'bma.ModelWrapper':
    """
    Create a model consisting of only one rule from the original trained model.
    """
    original_layer = original_model.model.f_layers[layer_nr]
    in_dim = original_layer.in_dim
    
    # Check types
    ante_type = getattr(original_layer, 'ante_type', 'trap')
    cons_type = getattr(original_layer, 'cons_type', 'trap')

    # 1. Create a new FuzzyNetwork with a single rule
    # Pass the mf types to the constructor
    single_rule_f_network = FuzzyNetwork(f_layers=[[in_dim, 1, 1]], 
                                         ante_memb=ante_type,
                                         cons_memb=cons_type,
                                         device=original_layer.device).to(original_layer.device)

    # 2. Create a model wrapper
    single_rule_model_wrapper = bma.ModelWrapper(single_rule_f_network)
    single_rule_model_wrapper.compile()

    # 3. Transfer Parameters
    ante_params = original_layer.Antes[rule_nr]
    cons_params = original_layer.Cons[rule_nr, cons_nr].unsqueeze(0)

    params = torch.cat([ante_params.flatten(), cons_params.flatten()])
    single_rule_model_wrapper.model.set_trainable_params(params.detach().cpu().numpy())

    return single_rule_model_wrapper


def fuzzyNetwork_significance(model_wrapper: 'bma.ModelWrapper', 
                              x_inputs: np.ndarray, 
                              verbose: bool = False, 
                              output_dim_only: Optional[int] = None) -> np.ndarray:
    """
    Calculate the significance of each rule in a fuzzy network for given observations.

    Parameters:
    ----------
    model_wrapper : bma.ModelWrapper
        An object that wraps the fuzzy network model.
    x_inputs : numpy.ndarray
        A 2D array of shape (n_samples, n_features).
    verbose : bool, optional
        If True, prints detailed information. Default is False.
    output_dim_only : int, optional
        If set, limits the calculation to a specific output dimension.

    Returns:
    -------
    rule_significance : numpy.ndarray
        A 3D array of shape (n_samples, nr_consequent, nr_rules) containing the 
        significance (area ratio) of each rule.
    """
    
    # We assume a single fuzzy layer for simplicity based on the implementation structure
    layer_nr = 0 
    
    # Get the number of rules and output dimensions from the model
    current_layer = model_wrapper.model.f_layers[layer_nr]
    nr_consequent = current_layer.out_dim
    nr_rules = current_layer.nr_rules

    out_dim_range = range(nr_consequent)
    if output_dim_only is not None:
        out_dim_range = [output_dim_only]
        nr_consequent = 1 # Adjust size if only one dim is requested

    # Initialize a 3D array to store rule significance
    # Shape: (n_samples, nr_consequent, nr_rules)
    rule_significance = np.zeros((len(x_inputs), len(out_dim_range), nr_rules))

    # Initialize a total areas array: (nr_consequent, n_samples)
    total_areas = np.zeros((len(out_dim_range), len(x_inputs))) 

    # --- Phase 1: Calculate and Accumulate Total Areas ---
    
    for cons_idx, cons_nr in enumerate(out_dim_range):
        current_rule_areas = []
        for nr_rule in range(nr_rules):
            single_rule_model = create_single_rule_model(model_wrapper, layer_nr=layer_nr, rule_nr=nr_rule, cons_nr=cons_nr)
            
            # Get predictions and areas for the current rule
            _, areas = compute_area_for_inputs(single_rule_model, x_inputs, cons_nr=cons_nr)
            current_rule_areas.append(areas)
            
            # Accumulate the total areas for this consequent
            total_areas[cons_idx] += areas
            
            if verbose:
                 for idx, (x, area) in enumerate(zip(x_inputs, areas)):
                     print(f'Input x = {x_inputs[idx]}: Rule {nr_rule}, Area below the cut = {area}')

        # Print the total areas for each input
        if verbose:
            for idx, x in enumerate(x_inputs):
                print(f'Total area below the cut for input x = {x}: {total_areas[cons_idx][idx]}')

    # --- Phase 2: Calculate Significance Ratios ---
    
    for cons_idx, cons_nr in enumerate(out_dim_range):
        for rule_number in range(nr_rules):
            single_rule_model = create_single_rule_model(model_wrapper, layer_nr=layer_nr, rule_nr=rule_number, cons_nr=cons_nr)
            _, areas = compute_area_for_inputs(single_rule_model, x_inputs, cons_nr=cons_nr)

            # Calculate the ratios for significance using NumPy for vectorization
            # Ratios = Rule Area / Total Area
            
            # Use np.divide and np.where for robust, vectorized division
            total_area_vector = total_areas[cons_idx]
            ratios = np.divide(areas, total_area_vector, 
                               out=np.zeros_like(areas), 
                               where=total_area_vector > 1e-10) # Avoid division by zero/near-zero

            # Store the significance ratios
            rule_significance[:, cons_idx, rule_number] = ratios

            if verbose:
                print(f'Rule {rule_number}, Consequent {cons_nr} ratios: {ratios}')

    # Returns shape (n_samples, nr_consequent_actual, nr_rules)
    return rule_significance

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
import skfuzzy as fuzz

# Assuming calculate_area_below_cut and compute_area_for_inputs are defined as before

def analyze_fuzzy_rules(model_wrapper: 'bma.ModelWrapper', 
                        x_inputs: np.ndarray, 
                        plot: bool = True, 
                        verbose: bool = True):
    """
    Analyze fuzzy rules with Dynamic Y-Axis scaling.
    Supports both Gaussian and Trapezoidal architectures.
    """
    
    layer_nr = 0
    current_layer = model_wrapper.model.f_layers[layer_nr]
    nr_rules = current_layer.nr_rules
    out_dim = current_layer.out_dim

    # --- Phase 1: Pre-Calculation & Range Finding ---
    
    rule_results = {} 
    total_metrics = np.zeros((out_dim, len(x_inputs))) 
    
    global_y_min = float('inf')
    global_y_max = float('-inf')

    if verbose:
        print(f"Scanning output range for {nr_rules} rules...")

    for cons_nr in range(out_dim):
        for nr_rule in range(nr_rules):
            # 1. Create Model
            single_rule_model = create_single_rule_model(model_wrapper, layer_nr=layer_nr, rule_nr=nr_rule, cons_nr=cons_nr)
            
            # 2. Compute Predictions and Significance Metrics (Area or w)
            predictions, metrics = compute_area_for_inputs(single_rule_model, x_inputs, cons_nr=cons_nr)
            predictions = np.array(predictions)
            metrics = np.array(metrics)

            # 3. Store
            rule_results[(nr_rule, cons_nr)] = (predictions, metrics)
            total_metrics[cons_nr] += metrics
            
            # 4. Update Global Min/Max
            current_min = np.min(predictions)
            current_max = np.max(predictions)
            if current_min < global_y_min: global_y_min = current_min
            if current_max > global_y_max: global_y_max = current_max

    # --- Calculate Final Y-Limits with Margin ---
    y_range = global_y_max - global_y_min
    if y_range == 0: 
        margin = 0.5 
    else:
        margin = y_range * 0.1 

    final_ylim = (global_y_min - margin, global_y_max + margin)
    
    if verbose:
        print(f"Global Y-Axis Range: {final_ylim}")

    # --- Phase 2: Plotting ---
    if plot:
        fig, axs = plt.subplots(nr_rules, out_dim, figsize=(10, 4 * nr_rules), 
                                sharex=True, sharey=True)

        if nr_rules == 1 and out_dim == 1: axs = np.array([[axs]])
        elif nr_rules == 1: axs = np.expand_dims(axs, axis=0)
        elif out_dim == 1: axs = np.expand_dims(axs, axis=1)

        x_interp_inputs = x_inputs.flatten() 
        sort_indices = np.argsort(x_interp_inputs)
        x_sorted = x_interp_inputs[sort_indices]

        # Interpolation grid
        if len(x_sorted) > 1 and x_sorted[-1] > x_sorted[0]:
            x_interp = np.linspace(np.min(x_sorted), np.max(x_sorted), 500)
        else:
            x_interp = np.linspace(x_sorted[0]-0.1, x_sorted[0]+0.1, 500) if len(x_sorted)>0 else np.array([0,1])

        for cons_nr in range(out_dim):
            for rule_number in range(nr_rules):
                ax = axs[rule_number, cons_nr]
                
                # Retrieve data
                predictions, metrics = rule_results[(rule_number, cons_nr)]
                
                predictions_sorted = predictions[sort_indices]
                metrics_sorted = metrics[sort_indices]
                total_metrics_sorted = total_metrics[cons_nr][sort_indices]

                # Ratios for Coloring (Metric / Total Metric)
                # Trap: Area / Total Area
                # Gauss: w / Sum(w)
                ratios = np.divide(metrics_sorted, total_metrics_sorted, 
                                   out=np.zeros_like(metrics_sorted), 
                                   where=total_metrics_sorted > 1e-10)

                # Interpolation
                x_unique, unique_indices = np.unique(x_sorted, return_index=True)
                y_unique = predictions_sorted[unique_indices]

                if len(x_unique) > 1:
                    pchip = PchipInterpolator(x_unique, y_unique)
                    y_interp = pchip(x_interp)
                else:
                    val = y_unique[0] if len(y_unique)>0 else 0
                    y_interp = np.full_like(x_interp, val)

                # Coloring
                min_ratio, max_ratio = np.min(ratios), np.max(ratios)
                normalized_ratios = (ratios - min_ratio) / (max_ratio - min_ratio) if max_ratio > min_ratio else np.zeros_like(ratios)
                cmap = plt.cm.RdYlGn
                
                for i in range(len(x_interp) - 1):
                    nearest_idx = np.searchsorted(x_sorted, x_interp[i], side='right') - 1
                    nearest_idx = max(0, min(len(x_sorted) - 1, nearest_idx))
                    color_val = normalized_ratios[nearest_idx]
                    ax.plot(x_interp[i:i+2], y_interp[i:i+2], color=cmap(color_val), lw=3)

                ax.set_title(f'Rule {rule_number} Output (Dim {cons_nr})')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_ylim(final_ylim)
                ax.tick_params(labelbottom=True) 

        plt.tight_layout() 
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def analyze_specific_fuzzy_rules(model_wrapper, x_inputs, target_rules=None, verbose=True):
    """
    Modified version of analyze_fuzzy_rules to plot only specific rule numbers.
    target_rules: List of rule numbers (e.g., [5, 10, 15])
    """
    layer_nr = 0
    current_layer = model_wrapper.model.f_layers[layer_nr]
    nr_rules = current_layer.nr_rules
    out_dim = current_layer.out_dim

    # --- Phase 1: Pre-Calculation (Same as original) ---
    rule_results = {} 
    total_metrics = np.zeros((out_dim, len(x_inputs))) 
    global_y_min, global_y_max = float('inf'), float('-inf')

    for cons_nr in range(out_dim):
        for nr_rule in range(nr_rules):
            # These helper functions (create_single_rule_model, etc.) must be in your namespace
            single_rule_model = create_single_rule_model(model_wrapper, layer_nr=layer_nr, rule_nr=nr_rule, cons_nr=cons_nr)
            predictions, metrics = compute_area_for_inputs(single_rule_model, x_inputs, cons_nr=cons_nr)
            
            predictions = np.array(predictions)
            metrics = np.array(metrics)
            rule_results[(nr_rule, cons_nr)] = (predictions, metrics)
            total_metrics[cons_nr] += metrics
            
            global_y_min = min(global_y_min, np.min(predictions))
            global_y_max = max(global_y_max, np.max(predictions))

    y_range = global_y_max - global_y_min
    margin = y_range * 0.1 if y_range != 0 else 0.5
    final_ylim = (global_y_min - margin, global_y_max + margin)

    # --- Phase 2: Targeted Plotting ---
    # If no target_rules provided, plot everything
    if target_rules is None:
        plot_indices = list(range(nr_rules))
    else:
        # Convert 1-based rule numbers to 0-based indices if necessary
        plot_indices = [r for r in target_rules if 0 <= r < nr_rules]

    num_to_plot = len(plot_indices)
    if num_to_plot == 0:
        print("No valid rules selected for plotting.")
        return

    fig, axs = plt.subplots(num_to_plot, out_dim, figsize=(10, 4 * num_to_plot), squeeze=False)

    x_sorted_indices = np.argsort(x_inputs.flatten())
    x_sorted = x_inputs.flatten()[x_sorted_indices]
    x_interp = np.linspace(np.min(x_sorted), np.max(x_sorted), 500)

    for cons_nr in range(out_dim):
        for plot_idx, rule_number in enumerate(plot_indices):
            ax = axs[plot_idx, cons_nr]
            predictions, metrics = rule_results[(rule_number, cons_nr)]
            
            # Extract and sort data
            pred_s = predictions[x_sorted_indices]
            met_s = metrics[x_sorted_indices]
            tot_s = total_metrics[cons_nr][x_sorted_indices]

            # Firing Ratio for coloring
            ratios = np.divide(met_s, tot_s, out=np.zeros_like(met_s), where=tot_s > 1e-10)

            # Interpolation
            x_u, u_idx = np.unique(x_sorted, return_index=True)
            y_u = pred_s[u_idx]
            pchip = PchipInterpolator(x_u, y_u)
            y_interp = pchip(x_interp)

            # Color mapping
            norm = plt.Normalize(np.min(ratios), np.max(ratios))
            cmap = plt.cm.RdYlGn

            for i in range(len(x_interp) - 1):
                # Simple color attribution based on nearest sorted x
                c_idx = np.searchsorted(x_sorted, x_interp[i], side='right') - 1
                c_idx = max(0, min(len(x_sorted) - 1, c_idx))
                ax.plot(x_interp[i:i+2], y_interp[i:i+2], color=cmap(norm(ratios[c_idx])), lw=3)

            ax.set_title(f'Rule {rule_number} Output (Localized Significance)')
            ax.set_ylim(final_ylim)
            ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

# Example: Only show rules 5, 10, and 15
# analyze_specific_fuzzy_rules(model_wrapper, obs, target_rules=[5, 10, 15])

def get_sorted_rule_significance(model_wrapper, x_input, cons_idx=0):
    """
    Calculates and sorts the significance of each rule for a specific input observation.
    
    Args:
        model_wrapper: The trained ModelWrapper object.
        x_input (list or np.ndarray): A single input observation (e.g., [0.5, 0.2, 0.8]).
        cons_idx (int): The output dimension index to analyze (default: 0).
        
    Returns:
        list of tuples: [(rule_index, significance_score), ...] sorted descending.
    """
    # 1. Setup & Device Handling
    layer = model_wrapper.model.f_layers[0]
    device = layer.device
    
    # Ensure input is a tensor on the correct device
    if not torch.is_tensor(x_input):
        x_input = torch.tensor(x_input, dtype=torch.float32, device=device)
    
    if x_input.dim() == 1:
        x_input = x_input.unsqueeze(0) # Shape: (1, in_dim)

    # 2. Get Architecture Types
    ante_type = getattr(layer, 'ante_type', 'trap')
    cons_type = getattr(layer, 'cons_type', 'trap')

    # 3. Calculate Firing Strength (w) for all rules
    # This logic mimics the forward pass of the FuzzyLayer
    with torch.no_grad():
        x_exp = x_input.unsqueeze(0).unsqueeze(-1) # Shape for broadcasting
        
        # Calculate Membership
        if ante_type == 'trap':
            # Use the model's internal trapmf or re-implement simple version
            # (Assuming internal params are used directly)
            from fuzzy_network_pytorch.FuzzyLayer_Gauss import trapmf
            memb = trapmf(x_exp, layer.Antes.unsqueeze(1)).squeeze(-1)
        else: # gauss
            from fuzzy_network_pytorch.FuzzyLayer_Gauss import gaussmf
            memb = gaussmf(x_exp, layer.Antes.unsqueeze(1)).squeeze(-1)

        # Firing Strength: Min across input features
        # w_vals shape: (nr_rules, 1)
        w_vals = torch.min(memb, dim=-1).values.flatten()

    # 4. Calculate Significance Metrics
    metrics = []
    
    # Move to CPU for analysis
    w_vals_np = w_vals.cpu().numpy()
    
    if cons_type == 'gauss':
        # --- Gaussian Significance ---
        # Significance is strictly the normalized firing strength
        metrics = w_vals_np
        
    elif cons_type == 'trap':
        # --- Trapezoidal Significance ---
        # Significance is the Geometric Area of the truncated consequent
        consequent_params = layer.Cons[:, cons_idx].detach().cpu().numpy()
        
        for i, w in enumerate(w_vals_np):
            if w < 1e-8:
                metrics.append(0.0)
                continue
            
            # Sort params for skfuzzy
            params = np.sort(consequent_params[i])
            
            # Calculate Area using numerical integration
            # Define universe slightly wider than the specific trapezoid
            uni_min, uni_max = params[0], params[3]
            margin = (uni_max - uni_min) * 0.1
            x_uni = np.linspace(uni_min - margin, uni_max + margin, 200)
            
            # Create Fuzzy Set & Cut
            mf = fuzz.trapmf(x_uni, params)
            cut_mf = np.minimum(mf, w)
            
            # Integrate
            area = np.trapz(cut_mf, x_uni)
            metrics.append(area)
            
        metrics = np.array(metrics)

    # 5. Normalize (Calculate Ratios)
    total_metric = np.sum(metrics)
    
    if total_metric > 1e-9:
        significance_scores = metrics / total_metric
    else:
        # If no rule fires (total = 0), all significance is 0
        significance_scores = np.zeros_like(metrics)

    # 6. Sort and Format Output
    # Create list of (index, score)
    indexed_scores = list(enumerate(significance_scores))
    
    # Sort descending by score
    sorted_rules = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    return sorted_rules

def get_top_n_rule_model(original_model_wrapper, x_input, n_rules=1):
    """
    Creates a new, smaller Fuzzy Network containing only the top 'n' most significant 
    rules for a specific input observation.

    Args:
        original_model_wrapper (ModelWrapper): The original trained model.
        x_input (list or np.ndarray): The input observation (e.g., [0.5, 0.2, 0.8]).
        n_rules (int): Number of top rules to keep.

    Returns:
        ModelWrapper: A new, executable model containing only the top n rules.
    """
    # 1. Get Sorted Significance (Reuse previous logic)
    # Returns list of tuples: [(rule_idx, score), ...]
    sorted_rules = get_sorted_rule_significance(original_model_wrapper, x_input)
    
    # 2. Select Top N Indices
    # Cap n_rules to the total number of rules available
    n_rules = min(n_rules, len(sorted_rules))
    
    # Extract just the indices (the first element of the tuple)
    top_indices = [idx for idx, score in sorted_rules[:n_rules]]
    top_indices_tensor = torch.tensor(top_indices, dtype=torch.long, device=original_model_wrapper.model.f_layers[0].device)
    
    print(f"Creating sub-model with rules: {top_indices} (Total Significance: {sum(s for i,s in sorted_rules[:n_rules]):.2%})")

    # 3. Access Original Layer Properties
    # We assume a single-layer network for this operation
    orig_layer = original_model_wrapper.model.f_layers[0]
    device = orig_layer.device
    in_dim = orig_layer.in_dim
    out_dim = orig_layer.out_dim
    
    # Get MF types (default to 'trap' if attribute missing)
    ante_type = getattr(orig_layer, 'ante_type', 'trap')
    cons_type = getattr(orig_layer, 'cons_type', 'trap')

    # 4. Create New Network Structure
    # Define layer structure: [Input Dim, Output Dim, NEW Rule Count]
    new_layer_def = [[in_dim, out_dim, n_rules]]
    
    new_network = FuzzyNetwork(f_layers=new_layer_def,
                               ante_memb=ante_type,
                               cons_memb=cons_type,
                               device=device).to(device)
    
    new_wrapper = ModelWrapper(new_network)
    new_wrapper.compile() # Initialize optimizer/loss (standard practice, though not needed for inference)

    # 5. Transplant Parameters
    # We copy the specific weights from the old model to the new model
    new_layer = new_network.f_layers[0]
    
    with torch.no_grad():
        # Copy Antecedents: Select specific rules using the indices
        # orig_layer.Antes shape: (Total_Rules, In_Dim, Params)
        # new_layer.Antes shape:  (N_Rules, In_Dim, Params)
        new_layer.Antes.data = orig_layer.Antes.data[top_indices_tensor].clone()
        
        # Copy Consequents
        # orig_layer.Cons shape: (Total_Rules, Out_Dim, Params)
        new_layer.Cons.data = orig_layer.Cons.data[top_indices_tensor].clone()

    return new_wrapper

import torch
import numpy as np
import skfuzzy as fuzz
from fuzzy_network_pytorch.FuzzyNetwork import FuzzyNetwork
from fuzzy_network_pytorch.bma import ModelWrapper

def get_averaged_rule_significance(model_wrapper, x_inputs, cons_idx=0):
    """
    Calculates the significance of each rule averaged over a batch of inputs.
    
    Args:
        model_wrapper: The trained ModelWrapper object.
        x_inputs (np.ndarray or list): Batch of inputs, shape (N_samples, N_features).
        cons_idx (int): The output dimension index to analyze.
        
    Returns:
        list of tuples: [(rule_index, average_score), ...] sorted descending.
    """
    # 1. Setup & Device Handling
    layer = model_wrapper.model.f_layers[0]
    device = layer.device
    
    # Ensure input is a tensor on the correct device
    if not torch.is_tensor(x_inputs):
        x_inputs = torch.tensor(x_inputs, dtype=torch.float32, device=device)
    
    # Handle single input vs batch
    if x_inputs.dim() == 1:
        x_inputs = x_inputs.unsqueeze(0) # Shape: (1, in_dim)
        
    n_samples = x_inputs.shape[0]

    # 2. Get Architecture Types
    ante_type = getattr(layer, 'ante_type', 'trap')
    cons_type = getattr(layer, 'cons_type', 'trap')

    # 3. Calculate Firing Strength (w) for ALL samples at once (Vectorized)
    with torch.no_grad():
        # x_exp shape: (batch, 1, in_dim)
        x_exp = x_inputs.unsqueeze(1).unsqueeze(-1) 
        
        # Calculate Membership
        if ante_type == 'trap':
            from fuzzy_network_pytorch.FuzzyLayer_Gauss import trapmf
            # layer.Antes shape: (nr_rules, in_dim, 4) -> (1, nr_rules, in_dim, 4)
            memb = trapmf(x_exp, layer.Antes.unsqueeze(0)).squeeze(-1)
        else: # gauss
            from fuzzy_network_pytorch.FuzzyLayer_Gauss import gaussmf
            memb = gaussmf(x_exp, layer.Antes.unsqueeze(0)).squeeze(-1)

        # Firing Strength: Min across input features
        # w_vals shape: (batch, nr_rules)
        w_vals = torch.min(memb, dim=-1).values

    # 4. Calculate Metrics (Raw Significance)
    w_vals_np = w_vals.cpu().numpy() # (N, Rules)
    metrics_matrix = np.zeros_like(w_vals_np)
    
    if cons_type == 'gauss':
        # --- Gaussian Significance ---
        # Significance is strictly the firing strength w
        metrics_matrix = w_vals_np
        
    elif cons_type == 'trap':
        # --- Trapezoidal Significance ---
        # We must calculate area for each rule for each sample.
        # This can be slow for large N, but we stick to the numerical integration consistency.
        consequent_params = layer.Cons[:, cons_idx].detach().cpu().numpy()
        nr_rules = w_vals_np.shape[1]
        
        # Pre-sort parameters for skfuzzy
        sorted_params = [np.sort(p) for p in consequent_params]
        
        for r in range(nr_rules):
            params = sorted_params[r]
            
            # Create a universe for this specific rule to integrate over
            uni_min, uni_max = params[0], params[3]
            margin = (uni_max - uni_min) * 0.1
            x_uni = np.linspace(uni_min - margin, uni_max + margin, 200)
            base_mf = fuzz.trapmf(x_uni, params)
            
            # Calculate area for every sample's w for this rule
            # Optimization: If w is 0, area is 0.
            for i in range(n_samples):
                w = w_vals_np[i, r]
                if w < 1e-8:
                    metrics_matrix[i, r] = 0.0
                else:
                    cut_mf = np.minimum(base_mf, w)
                    metrics_matrix[i, r] = np.trapz(cut_mf, x_uni)

    # 5. Normalize Per Sample (Get Relative Significance)
    # Sum across rules (axis 1)
    row_sums = metrics_matrix.sum(axis=1, keepdims=True)
    
    # Avoid division by zero
    row_sums[row_sums < 1e-9] = 1.0 
    
    normalized_significance = metrics_matrix / row_sums

    # 6. Average Across Batch (Get Global Significance)
    avg_significance = normalized_significance.mean(axis=0) # Shape: (nr_rules,)

    # 7. Sort
    indexed_scores = list(enumerate(avg_significance))
    sorted_rules = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    return sorted_rules

def get_top_n_rule_model_from_batch(original_model_wrapper, x_inputs, n_rules=1):
    """
    Creates a new, smaller Fuzzy Network containing only the top 'n' most significant 
    rules based on the AVERAGE significance across a batch of inputs.

    Args:
        original_model_wrapper (ModelWrapper): The original trained model.
        x_inputs (np.ndarray): Batch of inputs (N, D).
        n_rules (int): Number of top rules to keep.

    Returns:
        ModelWrapper: A new, executable model containing only the top n rules.
    """
    # 1. Get Sorted Significance (Averaged over batch)
    sorted_rules = get_averaged_rule_significance(original_model_wrapper, x_inputs)
    
    # 2. Select Top N Indices
    n_rules = min(n_rules, len(sorted_rules))
    
    # Extract indices
    top_indices = [idx for idx, score in sorted_rules[:n_rules]]
    
    # Calculate covered significance percentage
    total_avg_sig = sum(score for idx, score in sorted_rules[:n_rules])
    
    print(f"Selecting Top {n_rules} Rules based on batch average.")
    print(f"Selected Rules: {top_indices}")
    print(f"This sub-model explains approx {total_avg_sig:.2%} of the total rule influence.")

    # 3. Create New Model (Standard cloning procedure)
    orig_layer = original_model_wrapper.model.f_layers[0]
    device = orig_layer.device
    in_dim = orig_layer.in_dim
    out_dim = orig_layer.out_dim
    
    ante_type = getattr(orig_layer, 'ante_type', 'trap')
    cons_type = getattr(orig_layer, 'cons_type', 'trap')

    # Define new structure
    new_layer_def = [[in_dim, out_dim, n_rules]]
    new_network = FuzzyNetwork(f_layers=new_layer_def,
                               ante_memb=ante_type,
                               cons_memb=cons_type,
                               device=device).to(device)
    
    new_wrapper = ModelWrapper(new_network)
    new_wrapper.compile() 

    # 4. Transplant Weights
    top_indices_tensor = torch.tensor(top_indices, dtype=torch.long, device=device)
    new_layer = new_network.f_layers[0]
    
    with torch.no_grad():
        new_layer.Antes.data = orig_layer.Antes.data[top_indices_tensor].clone()
        new_layer.Cons.data = orig_layer.Cons.data[top_indices_tensor].clone()

    return new_wrapper



def run_elbow_analysis(model_wrapper, x_batch, loss_fn, y_batch=None, plot=True):
    """
    Performs Elbow Method analysis to find the optimal number of rules on an arbitrary task.
    
    Args:
        model_wrapper: The full trained model (Big Model).
        x_batch (np.ndarray): Input batch (N, D).
        loss_fn: Loss function (e.g., tlm.MSELoss() or nn.MSELoss()).
        y_batch (np.ndarray, optional): Ground truth targets (N, 1). 
                                        If None, calculates Fidelity (Loss vs Big Model).
        plot (bool): Whether to generate the elbow plot.
        
    Returns:
        dict: containing 'n_rules', 'losses'
    """
    # 1. Setup
    layer = model_wrapper.model.f_layers[0]
    total_rules = layer.nr_rules
    device = layer.device
    
    # Prepare Inputs
    x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=device)
    
    # 2. Determine the "Reference" (Ground Truth)
    model_wrapper.model.eval()
    with torch.no_grad():
        y_big = model_wrapper(x_tensor)
        
    if y_batch is not None:
        # Mode A: Accuracy (Small vs Real Truth)
        print("Elbow Mode: Accuracy (Comparing against provided targets)")
        y_ref = torch.tensor(y_batch, dtype=torch.float32, device=device)
        if y_ref.dim() == 1: y_ref = y_ref.unsqueeze(1)
        
        # Calculate Baseline (Big Model Performance)
        baseline_loss = loss_fn(y_big, y_ref).item()
        ylabel_text = "Loss (vs Ground Truth)"
    else:
        # Mode B: Fidelity (Small vs Big Model)
        print("Elbow Mode: Fidelity (Comparing against Full Model outputs)")
        y_ref = y_big
        baseline_loss = 0.0 # By definition, Big Model vs Big Model is 0
        ylabel_text = "Fidelity Loss (vs Full Model)"

    # 3. Iteration Loop
    results = {
        'n_rules': [],
        'losses': []
    }
    
    print(f"Running analysis on {total_rules} rules...")
    
    # Iterate from 1 rule up to Total Rules
    for k in range(1, total_rules + 1):
        # A. Create Sub-Model with top k rules
        # Uses the previously defined batch-significance logic
        sub_model = get_top_n_rule_model_from_batch(model_wrapper, x_batch, n_rules=k)
        
        # B. Evaluate
        sub_model.model.eval()
        with torch.no_grad():
            y_small = sub_model(x_tensor)
            loss = loss_fn(y_small, y_ref).item()
            
        # C. Store
        results['n_rules'].append(k)
        results['losses'].append(loss)

    # 4. Plotting
    if plot:
        n_rules = results['n_rules']
        losses = results['losses']
        
        plt.figure(figsize=(10, 6))
        
        # Plot Curve
        plt.plot(n_rules, losses, 'bo-', linewidth=2, markersize=6, label='Sub-Model Error')
        
        # Plot Baseline (only relevant if comparing vs Ground Truth)
        if y_batch is not None:
            plt.axhline(y=baseline_loss, color='g', linestyle='--', 
                        label=f'Full Model Baseline ({baseline_loss:.4f})')
        
        plt.title(f'Rule Complexity vs {ylabel_text}')
        plt.xlabel('Number of Rules (Complexity)')
        plt.ylabel(ylabel_text)
        plt.xticks(n_rules) 
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Highlight "Elbow" (Simple heuristic: largest drop)
        if len(losses) > 2:
            drops = np.abs(np.diff(losses))
            elbow_idx = np.argmax(drops) + 1 # +1 because diff is shorter
        plt.show()
        
    return results

import torch
import numpy as np
import matplotlib.pyplot as plt
def visualize_rule_complexities(full_model, x_batch, rule_numbers, loss_fn, y_batch=None):
    """
    Visualizes and compares the output of sub-models with different numbers of rules.
    
    Args:
        full_model: The original full model wrapper.
        x_batch (np.ndarray): The input batch (N, D).
        rule_numbers (list of int): List of rule counts to evaluate (e.g., [1, 3, 5]).
        loss_fn: Loss function.
        y_batch (np.ndarray, optional): Ground Truth targets.
    """
    device = full_model.model.f_layers[0].device
    x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=device)
    
    # 1. Get Full Model Predictions
    full_model.model.eval()
    with torch.no_grad():
        y_full = full_model(x_tensor)
        y_full_np = y_full.cpu().numpy().flatten()
        
    full_loss = 0.0
    if y_batch is not None:
        y_true = torch.tensor(y_batch, dtype=torch.float32, device=device)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        full_loss = loss_fn(y_full, y_true).item()

    # 2. Iterate through requested rule counts
    sub_model_results = []
    
    print(f"\n--- Comparing Rule Complexities: {rule_numbers} ---")
    
    for k in rule_numbers:
        # Create sub-model from Full Model
        sub_model = get_top_n_rule_model_from_batch(full_model, x_batch, n_rules=k)
        sub_model.model.eval()
        
        with torch.no_grad():
            y_sub = sub_model(x_tensor)
            y_sub_np = y_sub.cpu().numpy().flatten()
            
            if y_batch is not None:
                loss = loss_fn(y_sub, y_true).item()
                loss_str = f"{loss:.5f}"
            else:
                loss = loss_fn(y_sub, y_full).item()
                loss_str = f"{loss:.5f} (fid)"
                
        sub_model_results.append({
            'k': k,
            'preds': y_sub_np,
            'loss': loss,
            'loss_str': loss_str
        })
        print(f"k={k} Rules | Loss: {loss_str}")

    # 3. Plotting
    plt.figure(figsize=(12, 7))
    
    if x_batch.shape[1] == 1:
        x_axis = x_batch[:, 0]
        xlabel = "Input Feature"
    else:
        target_feat = np.argmax(np.var(x_batch, axis=0))
        x_axis = x_batch[:, target_feat]
        xlabel = f"Input Feature {target_feat}"

    if y_batch is not None:
        plt.plot(x_axis, y_batch.flatten(), 'g-', alpha=0.2, linewidth=6, label='Ground Truth')
    
    plt.plot(x_axis, y_full_np, 'k-', linewidth=2.5, alpha=0.8, label=f'Full Model (Loss: {full_loss:.4f})')
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(rule_numbers)))
    
    for res, color in zip(sub_model_results, colors):
        k = res['k']
        plt.plot(x_axis, res['preds'], linestyle='--', linewidth=2, color=color, 
                 label=f'{k} Rules (Loss: {res["loss_str"]})')

    plt.title(f"Approximation Quality by Rule Count")
    plt.xlabel(xlabel)
    plt.ylabel("Model Output")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()


    
def analyze_complexity_tradeoff(full_model, x_batch, loss_fn, y_batch=None, plot=True):
    """
    Analyzes how model error changes as rule complexity increases.
    Replaces the 'Elbow Method' to account for cases where error might increase.
    
    Args:
        full_model: The full trained model wrapper.
        x_batch (np.ndarray): Input batch (N, D).
        loss_fn: Loss function.
        y_batch (np.ndarray, optional): Ground truth targets.
        plot (bool): Whether to generate the trade-off plot.
    """
    # 1. Setup
    layer = full_model.model.f_layers[0]
    total_rules = layer.nr_rules
    device = layer.device
    
    x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=device)
    
    # 2. Determine Reference
    full_model.model.eval()
    with torch.no_grad():
        y_full = full_model(x_tensor)
        
    if y_batch is not None:
        print("Analysis Mode: Accuracy (Error vs Ground Truth)")
        y_ref = torch.tensor(y_batch, dtype=torch.float32, device=device)
        if y_ref.dim() == 1: y_ref = y_ref.unsqueeze(1)
        
        baseline_loss = loss_fn(y_full, y_ref).item()
        ylabel_text = "Loss (vs Ground Truth)"
        title_text = "Accuracy vs. Complexity"
    else:
        print("Analysis Mode: Fidelity (Difference from Full Model)")
        y_ref = y_full
        baseline_loss = 0.0 
        ylabel_text = "Fidelity Loss"
        title_text = "Fidelity vs. Complexity"

    # 3. Iteration Loop
    results = {
        'n_rules': [],
        'losses': []
    }
    
    print(f"Evaluating subsets from 1 to {total_rules} rules...")
    
    for k in range(1, total_rules + 1):
        # Create Sub-Model
        sub_model = get_top_n_rule_model_from_batch(full_model, x_batch, n_rules=k)
        
        sub_model.model.eval()
        with torch.no_grad():
            y_small = sub_model(x_tensor)
            loss = loss_fn(y_small, y_ref).item()
            
        results['n_rules'].append(k)
        results['losses'].append(loss)

    # 4. Plotting
    if plot:
        n_rules = results['n_rules']
        losses = results['losses']
        
        # Find Best K (Minimum Loss)
        min_loss_idx = np.argmin(losses)
        best_k = n_rules[min_loss_idx]
        min_loss = losses[min_loss_idx]

        plt.figure(figsize=(10, 6))
        
        # Plot the Trade-off Curve
        plt.plot(n_rules, losses, 'bo-', linewidth=2, markersize=6, label='Sub-Model Error')
        
        # Plot Full Model Baseline
        if y_batch is not None:
            plt.axhline(y=baseline_loss, color='g', linestyle='--', 
                        label=f'Full Model Baseline ({baseline_loss:.4f})')
        
        # Highlight the "Optimal" point (Minimum Error)
        plt.scatter([best_k], [min_loss], color='red', s=100, zorder=5, label=f'Best Performance (k={best_k})')
        
        plt.title(f'{title_text}')
        plt.xlabel('Model Order (Number of Rules)')
        plt.ylabel(ylabel_text)
        plt.xticks(n_rules) 
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.show()
        
    return results


def create_linspace_dataset(start_values, end_values, steps=50):
    """
    Creates a batch of observations linearly spaced between start and end values.
    Handles both scalar (1D) and vector (ND) inputs robustly.
    
    Args:
        start_values: Start point(s). Can be a single float or a list [x, y, z].
        end_values: End point(s). Can be a single float or a list [x, y, z].
        steps: Number of samples to generate.
    """
    # Ensure inputs are treated as arrays (handles scalar vs list automatically)
    start_values = np.atleast_1d(np.array(start_values, dtype=float))
    end_values = np.atleast_1d(np.array(end_values, dtype=float))
    
    n_features = len(start_values)
    
    # Initialize output array (Steps, Features)
    batch_data = np.zeros((steps, n_features))
    
    # Generate linspace for each feature column
    for i in range(n_features):
        batch_data[:, i] = np.linspace(start_values[i], end_values[i], steps)
        
    return batch_data