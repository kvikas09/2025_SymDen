"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset. Uses L_1/2 regularization for the EQL network."""

import pickle
import numpy as np
import os
import torch
import json
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim as optim 
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split


from utils import pretty_print, functions
# from utils.symbolic_network import SymbolicNet
from utils.sym_net import SymbolicNet
from utils.regularization import L12Smooth
from inspect import signature
from sklearn.metrics import r2_score
import time
import argparse
import pickle


np.random.seed(5)
torch.manual_seed(5)


N_TRAIN = 256       # Size of training dataset
N_VAL = 100         # Size of validation dataset
DOMAIN = (-1, 1)    # Domain of dataset - range from which we sample x
# DOMAIN = np.array([[0, -1, -1], [1, 1, 1]])   # Use this format if each input variable has a different domain
N_TEST = 100        # Size of test dataset
DOMAIN_TEST = (-2, 2)   # Domain of test dataset - should be larger than training domain to test extrapolation
NOISE_SD = 0        # Standard deviation of noise for training dataset
var_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]

# Standard deviation of random distribution for weight initializations.
init_sd_first = 0.1
init_sd_last = 1.0
init_sd_middle = 0.5


class CustomDataset(Dataset):
    def __init__(self, file_path, sample_size, fraction=1.0):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        x_data, y_data = data[0], data[1]
        if isinstance(y_data, list):
            y_data = torch.tensor(y_data, dtype=torch.float32)
        print(f"Type of x_data: {type(x_data)}")
        print(f"Type of y_data: {type(y_data)}")

        print("*****", x_data.shape, y_data.shape)

        # Reduce data size based on the fraction
        total_points = x_data.shape[0]
        selected_indices = np.random.permutation(total_points)[:int(total_points * fraction)]

        # Further limit to sample_size
        # selected_indices = torch.tensor(selected_indices, dtype=torch.long)
        selected_indices = selected_indices[:sample_size]

        self.x_data = x_data[selected_indices]
        print(f"Type of selected_indices: {type(selected_indices)}")
        print(f"Selected indices: {selected_indices}")
        self.y_data = y_data[selected_indices]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
       
        if torch.is_tensor(self.x_data[idx]):
            x_sample = self.x_data[idx].clone().detach().float()
            y_sample = self.y_data[idx].clone().detach().float()
        else:
            x_sample = torch.tensor(self.x_data[idx]).clone().detach()
            y_sample = torch.tensor(self.y_data[idx]).clone().detach()
        return x_sample, y_sample



def get_data_loader(split_type, sample_size, fraction=1.0, batch_size=32, shuffle=False):

    if split_type == 'train':
        file_path = "./Training_data_xy_8x8_with_score_obtained_through_NF.pkl"
    elif split_type == 'test':
        file_path = "./xy_8x8_data_with_score_obtained_through_NF.pkl"
    else:
        raise ValueError("split_type must be either 'train' or 'test'.")

    dataset = CustomDataset(file_path=file_path, sample_size=sample_size, fraction=fraction)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)




class Benchmark:
    """Benchmark object just holds the results directory (results_dir) to save to and the hyper-parameters. So it is
    assumed all the results in results_dir share the same hyper-parameters. This is useful for benchmarking multiple
    functions with the same hyper-parameters."""
    def __init__(self, results_dir, n_layers=5, reg_weight=5e-3, learning_rate=1e-2,
                 n_epochs1=10001, n_epochs2=10001):
        """Set hyper-parameters"""
        self.activation_funcs = [
            *[functions.Constant()] * 1,
            *[functions.Identity()] * 1, #4
            *[functions.Square()] * 1,
            *[functions.Cos()] * 2,
            *[functions.Sin()] * 2,
            # *[functions.Sigmoid()] * 2,
            *[functions.Product()] * 1,
            #  *[functions.Pow()] * 2        
            # *[functions.Pow(power=4)] * 2,
            # *[functions.Product(1)] * 1
        ]

        self.n_layers = n_layers                # Number of hidden layers
        self.reg_weight = reg_weight            # Regularization weight
        self.learning_rate = learning_rate
        self.summary_step = 1000                # Number of iterations at which to print to screen
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir

        # Save hyperparameters to file
        result = {
            "learning_rate": self.learning_rate,
            "summary_step": self.summary_step,
            "n_epochs1": self.n_epochs1,
            "n_epochs2": self.n_epochs2,
            "activation_funcs_name": [func.name for func in self.activation_funcs],
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
        }
        with open(os.path.join(self.results_dir, 'params.pickle'), "wb+") as f:
            pickle.dump(result, f)

    def benchmark(self, func, func_name, trials):
        """Benchmark the EQL network on data generated by the given function. Print the results ordered by test error.

        Arguments:
            func: lambda function to generate dataset
            func_name: string that describes the function - this will be the directory name
            trials: number of trials to train from scratch. Will save the results for each trial.
        """

        print("Starting benchmark for function:\t%s" % func_name)
        print("==============================================")

        # Create a new sub-directory just for the specific function
        func_dir = os.path.join(self.results_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        
        expr_list, error_test_list, r2_test_list = self.train(func, func_name, trials, func_dir)

        error_expr_sorted = sorted(zip(error_test_list, r2_test_list, expr_list), key=lambda x: (x[0], x[1]))

        # Separate the sorted results
        error_test_sorted = [x[0] for x in error_expr_sorted]  # Sorted errors
        r2_test_sorted = [x[1] for x in error_expr_sorted]     # Sorted R² values
        expr_list_sorted = [x[2] for x in error_expr_sorted]   # Corresponding expressions



        # Write the sorted results to the file
        results_path = os.path.join(self.results_dir, 'score_xy_ring/score_xy_ring_NF.txt')
        with open(results_path, 'a') as fi:
            fi.write("\n{}\n".format(func_name))
            for i in range(trials):
                fi.write("[%f]\tR²: %f\t%s\n" % (error_test_sorted[i], r2_test_sorted[i], str(expr_list_sorted[i])))
            fi.close()

    def train(self, func, func_name='', trials=1, func_dir='results/test'):
        """Train the network to find a given function"""

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        print("Use cuda:", use_cuda, "Device:", device)
        # file_path = "/home/riyanshas23/DeepSymRegTorch/dataset/eql_dataset/phi4_data/Training_data_phi4_8x8_with_exact_neglogp.pkl"
        # sample_size = 100000 
        fraction = 1.0       
        train_ratio = 1    
        batch_size = 256     
        # train_loader, test_loader = get_data_loaders(file_path, sample_size, fraction, train_ratio, batch_size)
        train_loader = get_data_loader(split_type='train', sample_size=100000, fraction=1.0, batch_size=256)
        test_loader = get_data_loader(split_type='test', sample_size=20000, fraction=1.0, batch_size=256)

        # train_loader = get_data_loader(split_type='train', sample_size=120000, fraction=1.0, batch_size=128)
        # Iterate over the data in batches
        for x_batch, y_batch in train_loader:
            y_batch = y_batch.squeeze(-1)
            data, target = x_batch.to(device), y_batch.to(device) 
            print("**************", data.shape, target.shape)  #shape [256, 1, 8, 8]) torch.Size([256])
            break

        for x_batch_test, y_batch_test in test_loader:
            y_batch_test = y_batch_test.squeeze(-1)
            test_data, test_target = x_batch_test.to(device), y_batch_test.to(device)  
            print("}}}}}}}}}}}}}}", test_data.shape, test_target.shape) #shape [256, 1, 8, 8]) torch.Size([256])
            break

        # train_loader = get_data_loader(split_type='train', sample_size=120000, fraction=1.0, batch_size=128)
        # Iterate over the data in batches
       
        
        lattice_train = torch.nn.functional.pad(data, (1, 1, 1, 1), mode="circular")
        lattice_train_target = torch.nn.functional.pad(target, (1, 1, 1, 1), mode="circular")
        # print(f"After padding: {lattice_train.shape}") # torch.Size([256, 1, 10, 10])
        # lattice_test = np.pad(x_batch_test, pad_width=1, mode='wrap')
        lattice_test = torch.nn.functional.pad(test_data, (1, 1, 1, 1), mode="circular")
        lattice_test_target = torch.nn.functional.pad(test_target, (1, 1, 1, 1), mode="circular")
        x_dim = 9
        width = len(self.activation_funcs)
        n_double = functions.count_double(self.activation_funcs)
      
        # Arrays to keep track of various quantities as a function of epoch
        loss_list = []          # Total loss (MSE + regularization)
        error_list = []         # MSE
        reg_list = []           # Regularization
        error_test_list = []    # Test error

        error_test_final = []
        metric_r2_list = []
        metric_r2_final = []
        eq_list = []

        for trial in range(trials):
            print("Training on function " + func_name + " Trial " + str(trial+1) + " out of " + str(trials))

            # reinitialize for each trial
            net = SymbolicNet(self.n_layers,
                            
                              funcs=self.activation_funcs,
                              initial_weights=[
                                  # kind of a hack for truncated normal
                                  torch.fmod(torch.normal(0, init_sd_first, size=(x_dim, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                #   torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                #   torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_last, size=(width, 1)), 2)
                              ]).to(device)

            loss_val = np.nan
            while np.isnan(loss_val):
                # training restarts if gradients blow up
                criterion = nn.MSELoss()
                optimizer = optim.RMSprop(net.parameters(),
                                          lr=self.learning_rate * 10,
                                          alpha=0.9,  # smoothing constant
                                          eps=1e-10,
                                          momentum=0.0,
                                          centered=False)

                # adaptive learning rate
                lmbda = lambda epoch: 0.1
                scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
                # for param_group in optimizer.param_groups:
                #     print("Learning rate: %f" % param_group['lr'])

                t0 = time.time()

                # First stage of training, preceded by 0th warmup stage

                for epoch in range(self.n_epochs1 + 2000):
                    optimizer.zero_grad()  # zero the parameter gradients
                    # Initialize containers to aggregate results over all chunks
                    aggregated_outputs = []
                    aggregated_targets = []
                    chunks_n = []
                    chunk_size = 3
                    stride=1
                    lattice_height, lattice_width = lattice_train.shape[2:4]
                    num_chunks_per_lattice = (
                        (lattice_height - chunk_size) // stride + 1
                    ) * ((lattice_width - chunk_size) // stride + 1)


                    for i in range(0, lattice_train.shape[2] - chunk_size + 1, stride):
                        for j in range(0, lattice_train.shape[3]- chunk_size + 1, stride):
                            data_chunk = lattice_train[:, :, i:i + chunk_size, j:j + chunk_size]  # (256, 1, 3, 3)
                            # center_chunk = data_chunk[:, :, [0, 1, 1, 1, 2], [1, 0, 1, 2, 1]]  # (256, 1, 1)
                            flattened_chunks_train = data_chunk.reshape(data_chunk.size(0), -1)  # (256, 1)
                            # print(f"Flattened shape: {flattened_chunks_train.shape}")  # Shape should be (256, 5)
                            target_chunk = lattice_train_target[:, :, i:i + chunk_size, j:j + chunk_size]  # (256, 1, 3, 3)

                            target_center_chunk = target_chunk[:, :, 1, 1]  # (256, 1, 1)
                            train_target_chunk = target_center_chunk.reshape(target_center_chunk.size(0), -1)  # (256, 1)
                            train_outputs_chunk = net(flattened_chunks_train)
                            # print("shape_test", train_target_chunk.shape, train_outputs_chunk.shape)  #shape_test torch.Size([256]) torch.Size([256, 1]) 81
                            aggregated_outputs.append(train_outputs_chunk)
                            aggregated_targets.append(train_target_chunk)
                        # input_chunk = torch.tensor(chunk.flatten(), dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)
                        # flattened_chunks = chunks.reshape(chunks.shape[0], -1)  #shape=(81,4)
                    aggregated_train_outputs = torch.cat(aggregated_outputs, dim=1)
                    aggregated_train_targets = torch.cat(aggregated_targets, dim=1)
                    # print("aggregated_train_targets*****", aggregated_train_targets.size()) # torch.Size([256])
                    # print("aggregated_train_outputs*****", aggregated_train_outputs.size()) # torch.Size([256, 1])
                    mse_loss = criterion(aggregated_train_outputs.squeeze(-1), aggregated_train_targets)
                    # print("chunked target", aggregated_targets_sum[:20])
                    # print("main target", target[:20])
                    regularization = L12Smooth()
                    reg_loss = regularization(net.get_weights_tensor())
                    loss = mse_loss + self.reg_weight * reg_loss
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Logging metrics at specified intervals
                    if epoch % self.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = loss.item()
                        error_list.append(error_val)
                        reg_list.append(reg_val)
                        loss_list.append(loss_val)

                       
                        with torch.no_grad():  # test error
                            aggregated_outputs_sec = []
                            aggregated_targets_sec = []
                            chunks_n = []                              
                            # Assuming `lattice_test` is already padded with mode="circular"
                            chunk_size = 3
                            stride = 1
                            lattice_height_test, lattice_width_test = lattice_test.shape[2:4]

                            num_chunks_per_lattice_test = (
                                (lattice_height_test - chunk_size) // stride + 1
                            ) * ((lattice_width_test - chunk_size) // stride + 1)

                            for i in range(0, lattice_test.shape[2] - chunk_size + 1, stride):
                                for j in range(0, lattice_test.shape[3] - chunk_size + 1, stride):
                                    # Extract a 3x3 lattice
                                    test_data_chunk = lattice_test[:, :, i:i + chunk_size, j:j + chunk_size]  # (256, 1, 3, 3)
                                    # center_cross_chunk = test_data_chunk[:, :, [0, 1, 1, 1, 2], [1, 0, 1, 2, 1]]  # (256, 1, 5)
                                    flattened_chunks_test = test_data_chunk.reshape(test_data_chunk.size(0), -1)  # (256, 5)
                                    # print(f"Flattened test shape: {flattened_chunks_test.shape}")  # Shape should be (256, 5)
                                    test_target_chunk = lattice_test_target[:, :, i:i + chunk_size, j:j + chunk_size]  # (256, 1, 3, 3)

                                    target_center_cross_chunk = test_target_chunk[:, :, 1, 1]  # (256, 1, 1)
                                    flattened_chunks_target_test = target_center_cross_chunk.reshape(target_center_cross_chunk.size(0), -1)  # (256, 1)
                                    test_outputs_chunk = net(flattened_chunks_test)
                                    # print("shape_test", flattened_chunks_target_test.shape, test_outputs_chunk.shape)  #shape_test torch.Size([256]) torch.Size([256, 1]) 81
                                    aggregated_outputs_sec.append(test_outputs_chunk)
                                    aggregated_targets_sec.append(flattened_chunks_target_test)
                        
                            aggregated_test_outputs = torch.cat(aggregated_outputs_sec, dim=1)
                            aggregated_test_targets = torch.cat(aggregated_targets_sec, dim=1)
                            test_loss = F.mse_loss(aggregated_test_outputs.squeeze(-1), aggregated_test_targets)
                            test_target_np = aggregated_test_targets.cpu().numpy()
                            test_outputs_np = aggregated_test_outputs.squeeze(-1).cpu().numpy()
                            r2_test = r2_score(test_target_np, test_outputs_np)
                            error_test_val = test_loss.item()
                            error_test_list.append(error_test_val)
                            metric_r2_list.append(r2_test)

                           
                        print(f"Epoch: {epoch}\tTotal training loss: {loss_val:.4f}\t"
                            f"Test error: {error_test_val:.4f}\tR²: {r2_test:.4f}")

                        if np.isnan(loss_val) or loss_val > 1000:  # If loss goes to NaN, restart training
                            break

                    if epoch == 2000:
                        scheduler.step()  # lr /= 10

                scheduler.step()  # lr /= 10 again

                for epoch in range(self.n_epochs2):
                    optimizer.zero_grad()  # zero the parameter gradients
                    aggregated_outputs = []
                    aggregated_targets = []
                    chunks_n = []
                    chunk_size = 3
                    stride=1


                    for i in range(0, lattice_train.shape[2] - chunk_size + 1, stride):
                        for j in range(0, lattice_train.shape[3]- chunk_size + 1, stride):
                            data_chunk = lattice_train[:, :, i:i + chunk_size, j:j + chunk_size]  # (256, 1, 3, 3)
                            flattened_chunks_train = data_chunk.reshape(data_chunk.size(0), -1)  # (256, 5)
                            target_chunk = lattice_train_target[:, :, i:i + chunk_size, j:j + chunk_size]  # (256, 1, 3, 3)
                            target_center_chunk = target_chunk[:, :, 1, 1]  # (256, 1, 1)
                            train_target_chunk = target_center_chunk.reshape(target_center_chunk.size(0), -1)  # (256, 1)
                            train_outputs_chunk = net(flattened_chunks_train)
                            aggregated_outputs.append(train_outputs_chunk)
                            aggregated_targets.append(train_target_chunk)
                    aggregated_train_outputs = torch.cat(aggregated_outputs, dim=1)
                    aggregated_train_targets = torch.cat(aggregated_targets, dim=1)
                    mse_loss = criterion(aggregated_train_outputs.squeeze(-1), aggregated_train_targets)
                    regularization = L12Smooth()
                    reg_loss = regularization(net.get_weights_tensor())
                    loss = mse_loss + self.reg_weight * reg_loss
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Logging metrics at specified intervals
                    if epoch % self.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = loss.item()
                        error_list.append(error_val)
                        reg_list.append(reg_val)
                        loss_list.append(loss_val)


                        with torch.no_grad():  # test error
                            aggregated_outputs_sec = []
                            aggregated_targets_sec = []
                            chunks_n = []
                            chunk_size = 3
                            stride=1

                            for i in range(0, lattice_test.shape[2] - chunk_size + 1, stride):
                                for j in range(0, lattice_test.shape[3] - chunk_size + 1, stride):
                                    # Extract a 3x3 lattice
                                    test_data_chunk = lattice_test[:, :, i:i + chunk_size, j:j + chunk_size]  # (256, 1, 3, 3)
                                    flattened_chunks_test = test_data_chunk.reshape(test_data_chunk.size(0), -1)  # (256, 5)
                                    test_target_chunk = lattice_test_target[:, :, i:i + chunk_size, j:j + chunk_size]  # (256, 1, 3, 3)
                                    target_center_cross_chunk = test_target_chunk[:, :, 1, 1]  # (256, 1, 1)
                                    flattened_chunks_target_test = target_center_cross_chunk.reshape(target_center_cross_chunk.size(0), -1)  # (256, 1)
                                    test_outputs_chunk = net(flattened_chunks_test)
                                    aggregated_outputs_sec.append(test_outputs_chunk)
                                    aggregated_targets_sec.append(flattened_chunks_target_test)
                        
                            aggregated_test_outputs = torch.cat(aggregated_outputs_sec, dim=1)
                            aggregated_test_targets = torch.cat(aggregated_targets_sec, dim=1)

                            test_loss = F.mse_loss(aggregated_test_outputs.squeeze(-1), aggregated_test_targets)
                            test_target_np = aggregated_test_targets.cpu().numpy()
                            test_outputs_np = aggregated_test_outputs.squeeze(-1).cpu().numpy()
                            r2_test = r2_score(test_target_np, test_outputs_np)
                            error_test_val = test_loss.item()
                            error_test_list.append(error_test_val)
                            metric_r2_list.append(r2_test)

                            # Store metrics
                            error_test_val = test_loss.item()
                            error_test_list.append(error_test_val)
                            metric_r2_list.append(r2_test)

                        print(f"Epoch: {epoch}\tTotal training loss: {loss_val:.4f}\t"
                            f"Test error: {error_test_val:.4f}\tR²: {r2_test:.4f}")

                        if np.isnan(loss_val) or loss_val > 1000:  # If loss goes to NaN, restart training
                            break

                t1 = time.time()

            tot_time = t1-t0
            print(tot_time)

           
            with torch.no_grad():
                weights = net.get_weights()
                for i, weight_matrix in enumerate(weights):
                    print(f"Shape of weight matrix {i + 1}: {weight_matrix.shape}")
                expr = pretty_print.network(weights, self.activation_funcs, var_names[:x_dim])
                print(expr)

            # Save results
            trial_file = os.path.join(func_dir, 'trial%d.pickle' % trial)
            results = {
                "weights": weights,
                "loss_list": loss_list,
                "error_list": error_list,
                "reg_list": reg_list,
                "error_test": error_test_list,
                "expr": expr,
                "runtime": tot_time
            }
            with open(trial_file, "wb+") as f:
                pickle.dump(results, f)

            error_test_final.append(error_test_list[-1])
            metric_r2_final.append(metric_r2_list[-1])
            eq_list.append(expr)
           


        return eq_list, error_test_final, metric_r2_final


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str, default='results/benchmark/test')
    parser.add_argument("--n-layers", type=int, default=2, help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float, default=5e-3, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Base learning rate for training')
    parser.add_argument("--n-epochs1", type=int, default=10001, help="Number of epochs to train the first stage")
    parser.add_argument("--n-epochs2", type=int, default=10001,
                        help="Number of epochs to train the second stage, after freezing weights.")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')
    meta.write(json.dumps(kwargs))
    meta.close()

    bench = Benchmark(**kwargs)

    bench.benchmark(lambda x: x, func_name="x", trials=20)
    