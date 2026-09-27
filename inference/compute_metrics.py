import pickle
import numpy as np
from sklearn.metrics import r2_score
import sympy as sp
from joblib import Parallel, delayed
import torch
import torch.nn.functional as F
from sympy import sin, cos
from sympy.utilities.lambdify import lambdify


class CustomDataset:
    def __init__(self, file_path, sample_size, fraction=1.0):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        x_data, y_data = data[0], data[1]
        if isinstance(y_data, list):
            y_data = np.array(y_data, dtype=np.float32)

        print(f"Type of x_data: {type(x_data)}, shape: {x_data.shape}")
        print(f"Type of y_data: {type(y_data)}, shape: {y_data.shape}")

        total_points = x_data.shape[0]
        selected_indices = np.random.permutation(total_points)[:int(total_points * fraction)]
        selected_indices = selected_indices[:sample_size]

        self.x_data = x_data[selected_indices]
        self.y_data = y_data[selected_indices]

    def get_data(self):
        return self.x_data, self.y_data


# ---------------------------------------------------------------------------
# OPTIMIZATION 1: Lambdify the SymPy expression ONCE into a fast NumPy func.
# This replaces per-chunk .subs().evalf() which is ~1000x slower.
# ---------------------------------------------------------------------------
def build_vectorized_H(H_expr, symbols):
    """
    Convert a SymPy expression into a vectorized NumPy function.
    The returned function accepts arrays of shape [N, num_symbols] and
    returns an array of shape [N].
    """
    H_func = lambdify(symbols, H_expr, modules="numpy")

    def vectorized_H(chunks_array):
        # chunks_array: [N, num_symbols]
        args = [chunks_array[:, k] for k in range(chunks_array.shape[1])]
        return H_func(*args)

    return vectorized_H


# ---------------------------------------------------------------------------
# OPTIMIZATION 2: Extract ALL chunks for ALL samples at once using
# vectorized NumPy sliding-window strides — no Python loops over samples.
# ---------------------------------------------------------------------------
def extract_all_chunks(lattice_data_np, chunk_size, stride):
    """
    Extract sliding-window chunks from all lattice samples simultaneously.

    Args:
        lattice_data_np: np.ndarray of shape [N, C, H, W]
        chunk_size: int
        stride: int

    Returns:
        chunks: np.ndarray of shape [N, num_chunks, C * chunk_size * chunk_size]
        num_chunks_h, num_chunks_w: grid dimensions
    """
    N, C, H, W = lattice_data_np.shape
    num_chunks_h = (H - chunk_size) // stride + 1
    num_chunks_w = (W - chunk_size) // stride + 1

    # Use numpy stride tricks to avoid copying data
    s_n, s_c, s_h, s_w = lattice_data_np.strides
    shape = (N, C, num_chunks_h, num_chunks_w, chunk_size, chunk_size)
    strides = (s_n, s_c, stride * s_h, stride * s_w, s_h, s_w)
    windows = np.lib.stride_tricks.as_strided(
        lattice_data_np, shape=shape, strides=strides
    )
    # windows: [N, C, nh, nw, chunk_h, chunk_w]
    # Reshape to [N, num_chunks, C*chunk_size*chunk_size]
    num_chunks = num_chunks_h * num_chunks_w
    flat_dim = C * chunk_size * chunk_size
    chunks = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N, num_chunks, flat_dim)
    return chunks.astype(np.float64)


# ---------------------------------------------------------------------------
# OPTIMIZATION 3: Evaluate H for all chunks in a single vectorized call,
# then sum over chunks per sample. No Python loops, no parallelism overhead.
# ---------------------------------------------------------------------------
def compute_H_lattice_vectorized(lattice_data, H_expr, symbols, chunk_size, stride):
    """
    Fully vectorized computation of H values for all samples and chunks.

    Args:
        lattice_data: torch.Tensor of shape [N, C, H, W]
        H_expr: SymPy expression
        symbols: list of SymPy symbols
        chunk_size: int
        stride: int

    Returns:
        np.ndarray of shape [N, 1]
    """
    # Convert to numpy once
    if isinstance(lattice_data, torch.Tensor):
        lattice_np = lattice_data.cpu().numpy()
    else:
        lattice_np = lattice_data

    print(f"Extracting chunks for {lattice_np.shape[0]} samples...")
    # [N, num_chunks, flat_dim]
    chunks = extract_all_chunks(lattice_np, chunk_size, stride)
    N, num_chunks, flat_dim = chunks.shape
    print(f"  → {num_chunks} chunks per sample, flat_dim={flat_dim}")

    # Build vectorized H function (compiled once)
    H_func = build_vectorized_H(H_expr, symbols)

    # Reshape to [N * num_chunks, flat_dim] for a single batched eval
    all_chunks_flat = chunks.reshape(-1, flat_dim)  # [N*num_chunks, flat_dim]

    print(f"  → Evaluating H on {all_chunks_flat.shape[0]} chunks (vectorized)...")
    H_vals = H_func(all_chunks_flat)  # [N*num_chunks]

    # Sum chunks per sample
    H_vals = H_vals.reshape(N, num_chunks)  # [N, num_chunks]
    H_summed = H_vals.sum(axis=1, keepdims=True)  # [N, 1]

    return H_summed.astype(np.float32)


# ---------------------------------------------------------------------------
# OPTIMIZATION 4: Batch processing still available for very large datasets
# (reduces peak memory), but now each batch is fully vectorized.
# ---------------------------------------------------------------------------
def compute_H_lattice_in_batches(
    lattice_data, H_expr, symbols, chunk_size, stride, batch_size
):
    """
    Process lattice data in batches to cap peak memory usage.
    Each batch is evaluated fully vectorized (no per-chunk Python loops).
    """
    if isinstance(lattice_data, torch.Tensor):
        N = lattice_data.shape[0]
    else:
        N = lattice_data.shape[0]

    results = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        print(f"Batch {start}–{end-1} / {N}")
        batch = lattice_data[start:end]
        H_batch = compute_H_lattice_vectorized(batch, H_expr, symbols, chunk_size, stride)
        results.append(H_batch)

    return np.vstack(results)


def compute_mean_normalized_mse(H, H_hat):
    H_mean = np.mean(H)
    H_tilde = H - H_mean
    H_hat_mean = np.mean(H_hat)
    H_hat_tilde = H_hat - H_hat_mean
    mse = np.mean((H_tilde - H_hat_tilde) ** 2)
    r2_test = r2_score(H_tilde, H_hat_tilde)
    return mse, r2_test

def log_prob_l(x, m2=1.0, lamda=4.0):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device)
    x.requires_grad_(True)
    potential = m2*x**2 + lamda*x**4
    Nd = len(x.shape) - 1
    dims = range(2, Nd + 1)
    for mu in dims:
        potential += 2*x**2
        potential -= x*torch.roll(x, -2, mu)
        potential -= x*torch.roll(x, 2, mu)
    return -torch.sum(potential, dim=[1, 2, 3]).detach().cpu().numpy()

def log_prob(x,m2 = 1.0 ,lamda = 4.0):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device)
    x.requires_grad_(True)
    potential = m2*x**2 + lamda*x**4
    Nd = len(x.shape)-1
    dims = range(2,Nd+1)
    for mu in dims:
        potential += 2*x**2
        potential -= x*torch.roll(x,-1,mu)
        potential -= x*torch.roll(x,1,mu)
        potential -= x*torch.roll(x,-2,mu)
        potential -= x*torch.roll(x,2,mu)
    return -torch.sum(potential , dim = [1,2,3]).to(device = x.device)


def simplify_and_prune(expr, threshold=1e-1, decimals=3):
    expr = sp.expand(expr)

    terms = []

    for term in expr.as_ordered_terms():
        coeff, rest = term.as_coeff_Mul()

        coeff = float(coeff)

        if abs(coeff) >= threshold:
            coeff = round(coeff, decimals)
            terms.append(coeff * rest)

    return sp.Add(*terms)

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Use cuda:", use_cuda, "Device:", device)

    x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3')

    H_expr = (
	-0.0465106*x0 + 0.00864121*x1 + 0.127782*x2 - 0.00527374*x3 + 0.472415*(0.335798*x0 + 0.064634*x1 - 0.241851*x2)*(0.293194*x0 - 0.0266168*x1 - 0.332716*x2 - 0.0306044*x3) + 1.17799*(0.256283*x0 - 0.378636*x1 + 0.123812*x2 - 0.0225913*x3)*(0.270455*x0 - 0.409539*x1 + 0.144079*x2 - 0.0242029*x3) + 0.422478*(-0.484548*x0 + 0.0471522*x1 + 0.420407*x2 + 0.0302831*x3 + 0.888753*(0.335798*x0 + 0.064634*x1 - 0.241851*x2)*(0.293194*x0 - 0.0266168*x1 - 0.332716*x2 - 0.0306044*x3) + 1.25436*(0.256283*x0 - 0.378636*x1 + 0.123812*x2 - 0.0225913*x3)*(0.270455*x0 - 0.409539*x1 + 0.144079*x2 - 0.0242029*x3) + 0.0966219)*(-0.228077*x0 + 0.041653*x1 + 0.174507*x2 + 0.0168307*x3 - 0.795417*(0.335798*x0 + 0.064634*x1 - 0.241851*x2)*(0.293194*x0 - 0.0266168*x1 - 0.332716*x2 - 0.0306044*x3) - 0.369597*(0.256283*x0 - 0.378636*x1 + 0.123812*x2 - 0.0225913*x3)*(0.270455*x0 - 0.409539*x1 + 0.144079*x2 - 0.0242029*x3) - 0.365899) - 0.640342*(-0.427662*x0 + 0.0772762*x1 + 0.124054*x2 + 0.0507336*x3 + 0.50813*(0.335798*x0 + 0.064634*x1 - 0.241851*x2)*(0.293194*x0 - 0.0266168*x1 - 0.332716*x2 - 0.0306044*x3) - 0.425031*(0.256283*x0 - 0.378636*x1 + 0.123812*x2 - 0.0225913*x3)*(0.270455*x0 - 0.409539*x1 + 0.144079*x2 - 0.0242029*x3) + 0.231324)*(0.223754*x0 - 0.0358602*x1 - 0.247933*x2 - 0.009169*x3 + 0.421088*(0.335798*x0 + 0.064634*x1 - 0.241851*x2)*(0.293194*x0 - 0.0266168*x1 - 0.332716*x2 - 0.0306044*x3) - 0.0440955*(0.256283*x0 - 0.378636*x1 + 0.123812*x2 - 0.0225913*x3)*(0.270455*x0 - 0.409539*x1 + 0.144079*x2 - 0.0242029*x3) + 0.0820819) + 0.947216
    )
    
    file_path = "/hdd_storage/data/riyansha/DeepSymRegTorch/dataset/eql_dataset/xy_data/xy_8x8_exact_value.pkl"

    # file_path = "/hdd_storage/data/riyansha/DeepSymRegTorch/dataset/eql_dataset/phi4_data/phi4_8x8_data_with_exact_neglogp.pkl" 
    sample_size = 200000
    chunk_size = 2
    stride = 1
    batch_size = 512

    # dataset = CustomDataset(file_path, sample_size)
    # x_data, y_data = dataset.get_data()
    dataset = CustomDataset(file_path, sample_size=10**9)  # or total size
    x_data, y_data = dataset.get_data()

    # Keep only last 20k samples
    x_data = x_data[-20000:]
    y_data = y_data[-20000:]
    print(f"Data shapes: x_data {x_data.shape}, y_data {y_data.shape}")

    symbols = [x0, x1, x2, x3]
    # target = y_data.squeeze(-1)
    target = y_data  # already 1D, no squeeze needed
    
    # target = log_prob_l(x_data)  # ground truth log probabilities
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    lattice_data = F.pad(x_tensor, (0, 1, 0, 1), mode="circular")

    y_values = np.array(target)
    # y_values = target.detach().cpu().numpy()

    # Vectorized batch computation — much faster than symbolic .subs() loops
    H_hat_values = compute_H_lattice_in_batches(
        lattice_data, H_expr, symbols, chunk_size, stride, batch_size
    )

    H_hat_values_squeeze = np.squeeze(H_hat_values)
    mse, r2_test = compute_mean_normalized_mse(y_values, H_hat_values_squeeze)

    print(f"Overall Mean Normalized MSE: {mse}")
    print(f"Overall R2 score: {r2_test}")
    
    print("Expanding and pruning small terms...")
    H_expr_pruned = simplify_and_prune(H_expr, threshold=1e-1, decimals=3)

    # sp.simplify(H_expr)


    output_file = "xy_without_cos_sin.txt"
    with open(output_file, "a") as f:
        f.write("====================================\n")
        f.write(f"Dimension computed: xy NF \n")
        f.write(f"Overall Mean Normalized MSE: {mse}\n")
        f.write(f"Overall R2 score: {r2_test}\n")
        f.write(f"Equation used for H(x_i): {str(H_expr)}\n")
        f.write(f"Simplified H(x_i): {str(H_expr_pruned)}\n")
        f.write("====================================\n")