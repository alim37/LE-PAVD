from models import string_to_dataset, string_to_model
from models_lePAVD import string_to_lePAVD, string_to_LEdataset
import torch
import yaml
import os
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt  
import pandas as pd
from torch.profiler import profile, ProfilerActivity
from torchinfo import summary 
import argparse, argcomplete

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def compute_percent_error(predicted, target):
    percent_errors = dict()
    for key in predicted.keys():
        if target.get(key):
            percent_errors[key] = np.abs(predicted[key] - target[key]) / target[key] * 100
    return percent_errors

        


def evaluate_predictions(model, test_data_loader, eval_coeffs, name, all_eval = False):
    
    test_losses = []
    predictions = []
    ground_truth = []
    inference_times = []
    max_errors = [0.0, 0.0, 0.0]
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if eval_coeffs:
        sys_params = []
    
    
    flops_counted = True # Only profile once
    total_flops = 0
    
    for inputs, labels, norm_inputs in test_data_loader:
        if model.is_rnn:
            h = model.init_hidden(inputs.shape[0])
            h = h.data

        inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
        
        # Profile only the first batch for FLOPs
        if flops_counted:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                if model.is_rnn:
                    start = time.time()
                    output, h, sysid = model(inputs, norm_inputs, h)
                    end = time.time()
                else:
                    start = time.time()
                    output, _, sysid = model(inputs, norm_inputs)
                    end = time.time()

            total_flops = sum([e.flops for e in prof.key_averages() if e.flops is not None])
            flops_counted = True
        
        else: 
            if model.is_rnn:
                start = time.time()
                output, h, sysid = model(inputs, norm_inputs, h)
                end = time.time()
            else:
                start = time.time()
                output, _, sysid = model(inputs, norm_inputs)
                end = time.time()
        
        inference_times.append(end-start)
        test_loss = model.loss_function(output.squeeze(), labels.squeeze().float())
        
        error = output.squeeze() - labels.squeeze().float()
        error = np.abs(error.cpu().detach().numpy())
        
        for i in range(3):
            if error[i] > max_errors[i]:
                max_errors[i] = error[i]
        
        test_losses.append(test_loss.cpu().detach().numpy())
        predictions.append(output.squeeze().cpu().detach().numpy())
        ground_truth.append(labels.cpu().detach().numpy())
        
        if eval_coeffs:
            sys_params.append(sysid.cpu().detach().numpy())

    # Compute RMSE
    
    rmse = np.sqrt(np.mean(test_losses, axis=0))
    avg_inference_time = np.mean(inference_times)
    final_loss = np.mean(test_losses, axis=0)

    print("RMSE:", rmse)
    print("Maximum Error:", max_errors)
    print("Final Losses: ", final_loss)
    print("Average Inference Time:", avg_inference_time)
    print(f"Estimated Total FLOPs (1 batch): {total_flops:,}")

    if eval_coeffs:
        means, _ = model.unpack_sys_params(np.mean(sys_params, axis=0))
        std_dev, _ = model.unpack_sys_params(np.std(sys_params, axis=0))
        percent_errors = compute_percent_error(*model.unpack_sys_params(np.mean(sys_params, axis=0)))
        
        print("Mean Coefficient Values")
        print("------------------------------------")
        pretty(means)
        print("Std Dev Coefficient Values")
        print("------------------------------------")
        pretty(std_dev)
        print("Percent Error")
        print("------------------------------------")
        pretty(percent_errors)
        print("------------------------------------")

    return final_loss

  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label point clouds with bounding boxes.")
    parser.add_argument("model_cfg", type=str, help="Config file for model")
    parser.add_argument("dataset_file", type=str, help="Dataset file")
    parser.add_argument("model_state_dict", type=str, help="Model weights file")
    parser.add_argument("--eval_coeffs", action="store_true", default=True, help="Print learned coefficients of model")
    parser.add_argument("--log_df", action='store_true', default=True, help="Log experiment in a pandas-df")
    # parser.add_argument("--data.set_split", type=float, default=1.0, help="Train/Val split ratio")
    eval = True
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    with open(argdict["model_cfg"], 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.path.dirname(argdict["model_state_dict"]), "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    data_npy = np.load(argdict["dataset_file"])
    
    if args.log_df:
        df = pd.DataFrame(columns=['Vx', 'Vy', 'yaw_rate', 'Frx', 'Fry', 'Ffy'])
        csv_path = f'{param_dict["MODEL"]["NAME"]}.csv'
        # Delete existing CSV file if it exists
        if os.path.exists(csv_path):
            os.remove(csv_path)  # ✅ Correct method for files
            print(f"Deleted existing file: {csv_path}")
            time.sleep(4)  # Wait for the file to be deleted
            
        # Create a fresh empty CSV
        df.to_csv(csv_path, index=False)
        if param_dict["MODEL"]["NAME"] not in string_to_model:
            model = string_to_lePAVD[param_dict["MODEL"]["NAME"]](param_dict, eval=eval, csv_path=csv_path)
            test_dataset = string_to_LEdataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"], scaler)
        else:
            model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=eval, csv_path=csv_path)
            test_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"], scaler)
            
    else:
        if param_dict["MODEL"]["NAME"] not in string_to_model:
            model = string_to_lePAVD[param_dict["MODEL"]["NAME"]](param_dict, eval=eval, csv_path=None)
            test_dataset = string_to_LEdataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"], scaler)
        else:
            model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=eval, csv_path=None)
            test_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"], scaler)

    # print(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set batch size and input dimensions
    batch_size = 1
    seq_len = param_dict["MODEL"]["HORIZON"]  # time steps / horizon
    input_dim = 7 # total state + action dims

    # Create dummy inputs
    dummy_x = torch.randn(batch_size, seq_len, input_dim).to(device)
    dummy_x_norm = torch.randn(batch_size, seq_len, input_dim).to(device)
    # print(f"Dummy input shape: {dummy_x.shape}, Dummy normalized input shape: {dummy_x_norm.shape}")
    # Use summary with input_data
    summary(model, input_data=(dummy_x, dummy_x_norm))
    
    model.load_state_dict(torch.load(argdict["model_state_dict"]))
    
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # print(f"Test dataset size: {len(test_dataset)}")
    losses = evaluate_predictions(model, test_data_loader, 
                                  argdict["eval_coeffs"], 
                                  param_dict["MODEL"]["NAME"],
                                  all_eval=False)
