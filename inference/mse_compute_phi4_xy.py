
import pickle
import numpy as np
from sklearn.metrics import r2_score
import sympy as sp
from joblib import Parallel, delayed
# from sympy import subs
import torch
import json
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim as optim 


class CustomDataset:
    def __init__(self, file_path, sample_size, fraction=1.0):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        x_data, y_data = data[0], data[1]
        if isinstance(y_data, list):
            y_data = np.array(y_data, dtype=np.float32)

        print(f"Type of x_data: {type(x_data)}")
        print(f"Type of y_data: {type(y_data)}")

        print("*****", x_data.shape, y_data.shape)
        total_points = x_data.shape[0]
        selected_indices = np.random.permutation(total_points)[:int(total_points * fraction)]
        selected_indices = selected_indices[:sample_size]

        self.x_data = x_data[selected_indices]
        self.y_data = y_data[selected_indices]

    def get_data(self):
        return self.x_data, self.y_data
    


def evaluate_lattice_chunk(lattice_chunk, H_expr, symbols):
    """
    Evaluate the symbolic expression H_expr for a single lattice chunk.
    """
    # print("^^^^", lattice_chunk)
    subs_dict = {symbols[k]: lattice_chunk[k] for k in range(len(lattice_chunk))}
    return float(H_expr.subs(subs_dict).evalf())



def compute_symbolic_H_lattice_in_batches(
    lattice_data, H_expr, symbols, chunk_size, stride, batch_size, n_jobs=8
):
    """
    Evaluate the symbolic expression H_expr for lattice data in smaller batches to improve efficiency.
    Returns:
        np.ndarray: Aggregated H(x_i) values as an array of shape [num_samples, 1].
    """
    num_samples = lattice_data.shape[0]
    H_values_all_pred = []
    # H_values_all_true = []

    # Process the dataset in smaller batches
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = lattice_data[start_idx:end_idx]
        # batch_y = lattice_target[start_idx:end_idx]

        print(f"Processing batch {start_idx} to {end_idx - 1}...")  # Debug info

        # Process the current batch
        H_hat_values_batch = compute_symbolic_H_lattice(
            batch_data, H_expr, symbols, chunk_size, stride, n_jobs
        )

        # Collect results
        H_values_all_pred.append(H_hat_values_batch)
        # H_values_all_true.append(H_values_batch)

    # Concatenate all batch results
    return np.vstack(H_values_all_pred)


def compute_symbolic_H_lattice(
    lattice_data, H_expr, symbols, chunk_size, stride, n_jobs=8
):
    """
    Evaluate the symbolic expression H_expr for all lattice chunks in a large dataset
    and aggregate the results for each lattice.
    Args:
        lattice_data (torch.Tensor): Lattice data of shape [num_samples, channels, height, width].
    Returns:
        np.ndarray: Aggregated H(x_i) values as an array of shape [num_samples, 1].
    """
    num_samples, channels, lattice_height, lattice_width = lattice_data.shape

    def process_single_lattice(sample_idx):
        """
        Process a single lattice (sample_idx) to compute H(x_i).
        Args:
            sample_idx (int): Index of the sample to process.
        Returns:
            float: Aggregated H(x_i) value for the lattice.
        """
        chunks = []
        aggregated_targets = []
        for i in range(0, lattice_height - chunk_size + 1, stride):
            for j in range(0, lattice_width - chunk_size + 1, stride):
                data_chunk = lattice_data[sample_idx, :, i:i + chunk_size, j:j + chunk_size]
                # print("&&&&&",data_chunk.shape)
                center_cross_chunk = data_chunk[:, [0, 1, 1, 1, 2], [1, 0, 1, 2, 1]]  # (batch_size, 1, 5)
                # print(")))))))))))))))", center_cross_chunk.shape)
                # flattened_chunks = center_cross_chunk.reshape(center_cross_chunk.size(0), -1)  # (batch_size, 5)
                # data_chunk_pro = center_cross_chunk.contiguous()  # Ensure memory layout is contiguous
                flattened_chunks = center_cross_chunk.squeeze(0)
                # print(")))))))(((((((((())))))))))))))))))", flattened_chunks.shape)
                # flattened_chunks = center_cross_chunk
                # target_chunk = lattice_target[sample_idx, :, i:i + chunk_size, j:j + chunk_size]  # (256, 1, 3, 3)
                # center_chunk = target_chunk[:, :, 1, 1]  # (256, 1, 1)
                # target_center_chunk = center_chunk.reshape(center_chunk.size(0), -1)  (256, 1)
                # data_chunk = np.ascontiguousarray(data_chunk)  # Ensure memory layout is contiguous
                if isinstance(flattened_chunks, np.ndarray):
                    flattened_chunks = torch.tensor(flattened_chunks)
                # Flatten the data_chunk and convert it to NumPy
                chunks.append(flattened_chunks)
                

        
        # Stack the list of flattened chunks into numpy arrays
        # flattened_chunks_data_array = np.vstack(chunks)  # Shape: (num_chunks, 5)
        # flattened_chunks_target_array = np.vstack(aggregated_outputs)  # Shape: (num_chunks, 1)
        # Parallel evaluation of chunks
        chunk_H_values = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_lattice_chunk)(chunk, H_expr, symbols) for chunk in chunks
        )
        H_hat_value = sum(chunk_H_values)
        # H_value = sum(aggregated_targets)

        # Aggregate results (e.g., sum all chunk results)
        return H_hat_value

    # Process all samples in parallel
    H_hat_all_values = Parallel(n_jobs=n_jobs)(
        delayed(process_single_lattice)(sample_idx) for sample_idx in range(num_samples)
    )

    pred = np.array(H_hat_all_values, dtype=np.float32).reshape(-1, 1)
    # true = np.array(H_all_values, dtype=np.float32).reshape(-1, 1)

    return pred





def compute_mean_normalized_mse(H, H_hat):
    """
    Compute the Mean Normalized MSE using NumPy.

    Args:
        H (np.ndarray): Ground truth values of shape [num_samples, 1].
        H_hat (np.ndarray): Predicted values of shape [num_samples, 1].

    Returns:
        float: Mean Normalized MSE value.
    """
    H_mean = np.mean(H)
    H_tilde = H - H_mean
    r2_com = np.mean((H_tilde) ** 2)
    print("**********", r2_com)
    H_hat_mean = np.mean(H_hat)
    H_hat_tilde = H_hat - H_hat_mean
    mse = np.mean((H_tilde - H_hat_tilde) ** 2)
    r2_test = r2_score(H_tilde, H_hat_tilde)
    return mse, r2_test


if __name__ == "__main__":
    # Define symbolic function
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Use cuda:", use_cuda, "Device:", device)
    x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3')
    # mass 1 score_NF H_expr = [0.338174]	R²: 0.977304	-3.23752*(-0.463594*x1 - 0.477209*x2 + 2.50547*x3 - 0.482296*x4 - 0.462577*x5)*(0.00349983*x1 + 0.00360262*x2 - 0.0189147*x3 + 0.00364102*x4 + 0.00349216*x5 - 1.90118*(-0.0866708*x1 - 0.0863942*x2 - x3 - 0.0876338*x4 - 0.100068*x5)**2 - 1.28859)
    # mass 1 score_logp  H_expr = [0.004670]	R²: 0.999660 -2.99117*(1.40062*(0.09704*x1 + 0.0972523*x2 + x3 + 0.094971*x4 + 0.0972112*x5)**2 + 0.899848)*(0.710809*x1 + 0.709707*x2 - 3.78166*x3 + 0.708839*x4 + 0.709186*x5)
    # mass -4 score_logp [0.447358]	R²: 0.973430	-0.13427*x1 - 0.126116*x2 - 0.238444*x3 - 0.139028*x4 - 0.140056*x5 + 2.14724*(-0.973077*(-0.283159*x1 - 0.258663*x2 + 0.867824*x3 - 0.289764*x4 - 0.302204*x5)*(-0.249425*x1 - 0.280963*x2 - 0.232333*x3 - 0.247842*x4 - 0.213926*x5) + 1.06613*(0.0270655*x1 + 0.0114352*x2 + x3 + 0.0264943*x4 + 0.0352934*x5)**4 + 1.52591*(0.20409*x1 + 0.156221*x2 + 1.25564*x3 + 0.230162*x4 + 0.266041*x5)*(0.309034*x1 + 0.313569*x2 - 0.897367*x3 + 0.317815*x4 + 0.304779*x5) - 1.22189)*(1.25476*x1 + 1.17857*x2 + 2.22828*x3 + 1.29923*x4 + 1.30884*x5 + 0.0144665) - 0.997365*(-0.268443*x1 - 0.252143*x2 - 0.476718*x3 - 0.277957*x4 - 0.280012*x5 - 0.416403*(-0.283159*x1 - 0.258663*x2 + 0.867824*x3 - 0.289764*x4 - 0.302204*x5)*(-0.249425*x1 - 0.280963*x2 - 0.232333*x3 - 0.247842*x4 - 0.213926*x5) - 0.429504*(0.0270655*x1 + 0.0114352*x2 + x3 + 0.0264943*x4 + 0.0352934*x5)**4 - 0.941951*(0.20409*x1 + 0.156221*x2 + 1.25564*x3 + 0.230162*x4 + 0.266041*x5)*(0.309034*x1 + 0.313569*x2 - 0.897367*x3 + 0.317815*x4 + 0.304779*x5) + 1)**4 + 1.01258*(-0.267367*x1 - 0.251132*x2 - 0.474807*x3 - 0.276843*x4 - 0.27889*x5 + 0.33862*(-0.283159*x1 - 0.258663*x2 + 0.867824*x3 - 0.289764*x4 - 0.302204*x5)*(-0.249425*x1 - 0.280963*x2 - 0.232333*x3 - 0.247842*x4 - 0.213926*x5) + 0.425865*(0.0270655*x1 + 0.0114352*x2 + x3 + 0.0264943*x4 + 0.0352934*x5)**4 + 0.965798*(0.20409*x1 + 0.156221*x2 + 1.25564*x3 + 0.230162*x4 + 0.266041*x5)*(0.309034*x1 + 0.313569*x2 - 0.897367*x3 + 0.317815*x4 + 0.304779*x5) - 1)**4
    # mass 1 score_scoremodel H_expr = [7.814778]	R²: -0.004578    -0.0342949
    # mass -4 score_smodel H_expr = [8.815805]	R²: -0.005284	0.0885664 - 2.24536*(x1*(-0.143642*x1 - 0.0118538*x2 - 0.51176*x4 - 0.0105186*x5) + 0.485506)**4
    #mass -4 score_nf H_expr = [2.344906]	R²: 0.866432	2.10109*(-0.556243*x1 - 0.560688*x2 - 1.45725*x3 - 0.571894*x4 - 0.578165*x5 - 0.273806*(0.0990568*x1 - x3 + 0.161769*x4 + 0.0865577*x5)**2 - 0.444119*(-0.019686*x1 - 0.606196*x2 + x3 - 0.0369553*x4 - 0.0146554*x5)**2 + 0.149391)*(-0.122109*x1 - 0.123085*x2 - 0.319902*x3 - 0.125545*x4 - 0.126922*x5 - 1.16719*(0.0167139*x1 + 1.02022*x3)*(0.0163244*x1 + 1.10708*x3 - 0.0137669*x4 - 0.118467*x5) + 1.51777*(-0.280203*x2 - 0.165862*x4 - 0.348948*x5)*(-0.485353*x1 - 0.310687*x2 + 0.509654*x3 - 0.234995*x4 - 0.0228974*x5) - 1.20639*(0.0990568*x1 - x3 + 0.161769*x4 + 0.0865577*x5)**2 + 0.594402) - 1.31873*(-0.326168*x1 - 0.328775*x2 - 0.854496*x3 - 0.335346*x4 - 0.339023*x5 + (0.0167139*x1 + 1.02022*x3)*(0.0163244*x1 + 1.10708*x3 - 0.0137669*x4 - 0.118467*x5) + 0.36577*(-0.019686*x1 - 0.606196*x2 + x3 - 0.0369553*x4 - 0.0146554*x5)**2 - 0.450433)**2 - 0.0369898

    H_expr = 0.194766*x0 - 0.0249554*x1 + 0.182251*x2 + 0.0435027*x3 - 0.0166694*(-0.802307*x0 + 0.137967*x1 + x2 - 0.339533*x3)**2 + 0.432351*(-0.547271*x0 + 0.253185*x1 - 0.0939368*x2 + 0.336714*x3)*(-0.260465*x0 + 0.0637665*x1 - 0.121797*x2 + 0.291907*x3) + 0.178332*(-0.0104435*x0 - 0.153878*x1 + 0.238359*x2 + 0.137921*x3)*(0.053326*x0 + 0.195785*x1 - 0.0941533*x2 + 0.0245926*x3) + 1.72474*(-0.0385779*x0 + 0.0336398*x1 + 0.00961086*x2 + 0.0853737*x3 + 0.0199962*(-0.802307*x0 + 0.137967*x1 + x2 - 0.339533*x3)**2 + 0.203083*(-0.547271*x0 + 0.253185*x1 - 0.0939368*x2 + 0.336714*x3)*(-0.260465*x0 + 0.0637665*x1 - 0.121797*x2 + 0.291907*x3) - 0.264419*(-0.0104435*x0 - 0.153878*x1 + 0.238359*x2 + 0.137921*x3)*(0.053326*x0 + 0.195785*x1 - 0.0941533*x2 + 0.0245926*x3) - 0.90492)*(0.0407389*x0 + 0.0081696*x1 + 0.0594488*x2 + 0.0529539*x3 - 0.111936*(-0.802307*x0 + 0.137967*x1 + x2 - 0.339533*x3)**2 - 0.236686*(-0.547271*x0 + 0.253185*x1 - 0.0939368*x2 + 0.336714*x3)*(-0.260465*x0 + 0.0637665*x1 - 0.121797*x2 + 0.291907*x3) + 1.03644*(-0.0104435*x0 - 0.153878*x1 + 0.238359*x2 + 0.137921*x3)*(0.053326*x0 + 0.195785*x1 - 0.0941533*x2 + 0.0245926*x3) - 0.402119) + 0.00468058*(-0.00229879*x0 + 0.00479666*x1 + 0.00502014*x2 + 0.0142323*x3 + 0.0873483*(-0.802307*x0 + 0.137967*x1 + x2 - 0.339533*x3)**2 + 0.682107*(-0.547271*x0 + 0.253185*x1 - 0.0939368*x2 + 0.336714*x3)*(-0.260465*x0 + 0.0637665*x1 - 0.121797*x2 + 0.291907*x3) - (-0.0104435*x0 - 0.153878*x1 + 0.238359*x2 + 0.137921*x3)*(0.053326*x0 + 0.195785*x1 - 0.0941533*x2 + 0.0245926*x3) - 0.658426)**2 + 0.210187*(0.0537816*x0 + 0.000814788*x1 + 0.0626001*x2 + 0.0372514*x3 + 0.0546041*(-0.802307*x0 + 0.137967*x1 + x2 - 0.339533*x3)**2 - 1.30656*(-0.547271*x0 + 0.253185*x1 - 0.0939368*x2 + 0.336714*x3)*(-0.260465*x0 + 0.0637665*x1 - 0.121797*x2 + 0.291907*x3) - 0.64657*(-0.0104435*x0 - 0.153878*x1 + 0.238359*x2 + 0.137921*x3)*(0.053326*x0 + 0.195785*x1 - 0.0941533*x2 + 0.0245926*x3) + 0.346489)*(0.113584*x0 - 0.0006788*x1 + 0.128386*x2 + 0.0708137*x3 - 0.00906477*(-0.802307*x0 + 0.137967*x1 + x2 - 0.339533*x3)**2 - 0.084189*(-0.547271*x0 + 0.253185*x1 - 0.0939368*x2 + 0.336714*x3)*(-0.260465*x0 + 0.0637665*x1 - 0.121797*x2 + 0.291907*x3) + 0.415077*(-0.0104435*x0 - 0.153878*x1 + 0.238359*x2 + 0.137921*x3)*(0.053326*x0 + 0.195785*x1 - 0.0941533*x2 + 0.0245926*x3) + 0.304616) + 0.253318

     #xy_score_logp [0.007924]	R²: 0.993625 -0.894256*sin(1.15268*sin(0.0122412*x1 + 0.0137464*x2 + 0.672999*x3 + 0.0128407*x4 - 0.551492*x5) + 2.62083*cos(0.189406*x3 - 0.281506*x5)) - 1.43187659828257*sin(0.50131*x1 - 0.998443*x3 + 0.499994*x4)*cos(0.499515*x1 - 0.498913*x4) - 1.07219*sin(-1.31197*sin(0.507398*x2 - 0.48543*x3) + 1.49665*cos(0.497933*x2 - 0.488952*x3) + 1.53009) + 0.434187
    # mass1 NF PHI4 H_expr = -0.490643*(0.0102861 - 2.57234*(-x1 + 0.0662378*x2 - 0.0506655*x3)**4)*(-1.87674*(-x1 + 0.0662378*x2 - 0.0506655*x3)**4 + 0.0102857*(-0.138247*x1 - 0.38224*x2 + 0.398069*x3 - 0.43261*x4)*(0.126087*x1 + 0.347212*x2 - 0.393579*x3 + 0.447666*x4) + 0.0184283) + 0.137539*(-x1 + 0.0662378*x2 - 0.0506655*x3)**4 - 0.354234*(-0.138247*x1 - 0.38224*x2 + 0.398069*x3 - 0.43261*x4)*(0.126087*x1 + 0.347212*x2 - 0.393579*x3 + 0.447666*x4) + 2.4474*(0.000225438*x1 + 0.0115723*x2 + 0.00494627*x3 - 0.0197314*x4 + (-x1 + 0.0662378*x2 - 0.0506655*x3)**4 - 0.497158*(-0.138247*x1 - 0.38224*x2 + 0.398069*x3 - 0.43261*x4)*(0.126087*x1 + 0.347212*x2 - 0.393579*x3 + 0.447666*x4) + 0.616092)**2 + 1.70107*(0.0114253*x1 + 0.58649*x2 + 0.25068*x3 - x4 + 0.0298697*(-x1 + 0.0662378*x2 - 0.0506655*x3)**4 + 0.134297*(-0.138247*x1 - 0.38224*x2 + 0.398069*x3 - 0.43261*x4)*(0.126087*x1 + 0.347212*x2 - 0.393579*x3 + 0.447666*x4) + 0.0094826)**2 - 0.888827
    
    eq = sp.expand(H_expr)
    eq_int = sp.integrate(eq, x3)
    new_terms = []
    for term in eq_int.as_ordered_terms():
        variables = term.free_symbols  # Get the variables in the term
        if len(variables) == 2:  # Check if the term is a product of two variables
            coef, rest = term.as_coeff_Mul()  # Separate coefficient and variables
            if len(rest.free_symbols) == 2:  # Confirm it's a product of two variables
        # Halve the coefficien
                term = coef / 2 * rest
        new_terms.append(term)
    new_expr = sum(new_terms)

    file_path = "/hdd_storage/data/riyansha/DeepSymRegTorch/dataset/eql_dataset/xy_data/xy_dataset_with_neglogp.pkl"
    # file_path = "/hdd_storage/data/riyansha/DeepSymRegTorch/dataset/eql_dataset/phi4_data/phi4_8x8_data_for_mass_1_alongwith_neglogp.pkl"
    # file_path = "/hdd_storage/data/riyansha/DeepSymRegTorch/dataset/eql_dataset/phi4_data/phi4_8x8_data_with_exact_neglogp.pkl"
    sample_size = 20000
    chunk_size = 2
    stride = 1
    batch_size = 512
    # Load dataset
    dataset = CustomDataset(file_path, sample_size)
    x_data, y_data = dataset.get_data()
    print(f"Data shapes: x_data {x_data.shape}, y_data {y_data.shape}")
    symbols = [x1, x2, x3, x4, x5]
    if torch.is_tensor(x_data):
        x_sample = x_data.clone().detach().float()
        y_sample = y_data.clone().detach().float()
    else:
        x_sample = torch.tensor(x_data, dtype=torch.float32)
        y_sample = torch.tensor(y_data, dtype=torch.float32)
    lattice_data = torch.nn.functional.pad(x_sample, (1, 1, 1, 1), mode="circular")
    # lattice_target_torch = torch.nn.functional.pad(y_data_torch, (1, 1, 1, 1), mode="circular")
    # lattice_data = lattice_data_torch.numpy()
    y_data_new = y_sample.numpy()
    
    # lattice_target = torch.nn.functional.pad(target, (1, 1, 1, 1), mode="circular")
    # Compute H_hat using the symbolic function

      # Process 1000 samples at a time


# Compute symbolic H values in batches
    H_hat_values = compute_symbolic_H_lattice_in_batches(
        lattice_data, new_expr, symbols, chunk_size, stride, batch_size, n_jobs=8
    )
    # H_values = y_values
    # print("H_values shape:", H_values.shape)  # Should output: (10000, 1)

   
    H_hat_values_squeeze = np.squeeze(H_hat_values)
    
    print("***", y_data_new.shape, H_hat_values_squeeze.shape)
    mse, r2_test = compute_mean_normalized_mse(y_data_new, H_hat_values_squeeze)

    # Print results
    print(f"Overall Mean Normalized MSE: {mse}")
    print(f"Overall R2 score: {r2_test}")

    # Save results
    output_file = "mse_results/phi4_score/results_score_NF_mass-4.txt"
    with open(output_file, "a") as f:
        f.write("====================================\n")
        f.write(f"Dimension computed: mass-4_NF(with_score)\n")
        f.write(f"Overall Mean Normalized MSE: {mse}\n")
        f.write(f"Overall R2 score: {r2_test}\n")
        f.write(f"Equation used for H(x_i): {str(H_expr)}\n")
        f.write("====================================\n")


# import pickle
# import numpy as np
# import random
# from sklearn.metrics import r2_score
# import sympy as sp
# from joblib import Parallel, delayed

# class CustomDataset:
#     def __init__(self, file_path, sample_size, fraction=1.0):
#         with open(file_path, "rb") as file:
#             data = pickle.load(file)
#         x_data, y_data = data[0], data[1]
#         if isinstance(y_data, list):
#             y_data = np.array(y_data, dtype=np.float32)

#         print(f"Type of x_data: {type(x_data)}")
#         print(f"Type of y_data: {type(y_data)}")
#         print("*****", x_data.shape, y_data.shape)

#         total_points = x_data.shape[0]
#         selected_indices = np.random.permutation(total_points)[:int(total_points * fraction)]
#         selected_indices = selected_indices[:sample_size]

#         self.x_data = x_data[selected_indices]
#         self.y_data = y_data[selected_indices]

#     def get_data(self):
#         return self.x_data, self.y_data
    
    
# # def compute_symbolic_H(all_data, H_expr, symbols):
    
# #     H_values = []
# #     chunk_size = len(symbols)

# #     for row in all_data:
# #         row_sum = 0
# #         for i in range(0, len(row), chunk_size):
# #             data_chunk = row[i:i + chunk_size]
# #             subs_dict = {symbols[j - i]: data_chunk[j - i] for j in range(i, i + chunk_size)}
# #             row_sum += float(H_expr.subs(subs_dict).evalf())
# #         H_values.append(row_sum)

# #     return np.array(H_values, dtype=np.float32).reshape(-1, 1)




# def evaluate_row(row, H_expr, symbols, chunk_size):
#     row_sum = 0
#     for i in range(0, len(row), chunk_size):
#         data_chunk = row[i:i + chunk_size]
#         subs_dict = {symbols[j - i]: data_chunk[j - i] for j in range(i, i + chunk_size)}
#         row_sum += float(H_expr.subs(subs_dict).evalf())
#     return row_sum


# def compute_symbolic_H_parallel(all_data, H_expr, symbols, n_jobs=8):
#     chunk_size = len(symbols)
#     H_values = Parallel(n_jobs=n_jobs)(
#         delayed(evaluate_row)(row, H_expr, symbols, chunk_size) for row in all_data
#     )
#     return np.array(H_values, dtype=np.float32).reshape(-1, 1)


# def compute_mean_normalized_mse(H, H_hat):
#     H_mean = np.mean(H)
#     H_tilde = H - H_mean
#     r2_com = np.mean((H_tilde) ** 2)
#     print("**********", r2_com)
#     H_hat_mean = np.mean(H_hat)
#     H_hat_tilde = H_hat - H_hat_mean
#     mse = np.mean((H_tilde - H_hat_tilde) ** 2)
#     r2_test = r2_score(H_tilde, H_hat_tilde)
#     return mse, r2_test


# if __name__ == "__main__":
#     # Define symbolic variables
#     x1, x2 = sp.symbols('x1 x2')
# #     [0.003422]	R²: 0.997702
# # Expr 1: 1.32964*x1 - 0.018834*x2**2 - 0.601287*x2 + 2.9527
# # Expr 2: -0.675399*x1 + 0.174632*x2**2 + 0.305427*x2 - 1.4326*(0.0116624*x1 + 0.351131*x2 - 1)**2 - 1.49984
# #    
#     expr1 = 1.3283*x1 - 0.674837*x2 + 2.97747
#     expr2 = -0.671931*x1 + 1.32828*x2 - 2.9816
     
#     # Integrate the expressions
#     integral_expr1 = sp.integrate(expr1, x1)
#     integral_expr2 = sp.integrate(expr2, x2)

#     # Combine the integrated expressions (e.g., summation for simplicity)
#     H_expr = integral_expr1 + integral_expr2
#     # C = random.uniform(4, 5)  # Random constant between -10 and 10
#     # result_with_constant = H_expr + C
#     print(f"Combined expression after integration: {H_expr}")

#     # file_path = "/home/riyanshas23/DeepSymRegTorch/dataset/eql_dataset/multivariate/multivariate_data_for_one_gaussian_with_neg_logp.pkl"
#     file_path = "/home/riyanshas23/DeepSymRegTorch/dataset/eql_dataset/multivariate/multivariate_data_for_one_gaussian_with_diagonal_cov_with_neg_logp.pkl"
#     sample_size = 20000
   

#     # Load dataset
#     dataset = CustomDataset(file_path, sample_size)
#     x_data, y_data = dataset.get_data()
#     print(f"Data shapes: x_data {x_data.shape}, y_data {y_data.shape}")

#     # Compute H_hat using the combined symbolic function
#     symbols = [x1, x2]
#     H_hat_values = compute_symbolic_H_parallel(x_data, H_expr, symbols, n_jobs=8)

#     # Compute Mean Normalized MSE and R2 score
#     H_hat_values_squeeze = np.squeeze(H_hat_values)
#     print("***", y_data.shape, H_hat_values_squeeze.shape)
#     mse, r2_test = compute_mean_normalized_mse(y_data, H_hat_values_squeeze)

#     # Print results
#     print(f"Overall Mean Normalized MSE: {mse}")
#     print(f"Overall R2 score: {r2_test}")

#     # Save results
#     output_file = "mse_results/multivariate/results_score_smodel.txt"
#     with open(output_file, "a") as f:
#         f.write("====================================\n")
#         f.write(f"Dimension computed: 2D_diagonal\n")
#         f.write(f"Overall Mean Normalized MSE: {mse}\n")
#         f.write(f"Overall R2 score: {r2_test}\n")
#         f.write(f"Equation used for H(x_i): {str(H_expr)}\n")
#         f.write("====================================\n")

