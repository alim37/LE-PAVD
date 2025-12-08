from models import string_to_model, string_to_dataset
from models_lePAVD import string_to_lePAVD, string_to_LEdataset
import torch
import numpy as np
import os
import yaml
import pickle
import shutil

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train(model, train_data_loader, val_data_loader, experiment_name, output_dir, log_wandb=False, project_name=None, use_ray_tune=False):
    print("Starting experiment: {}".format(experiment_name))
    valid_loss_min = torch.inf
    model.train()
    model.cuda()

    # Load output weights
    if param_dict['MODEL']['NAME'] == "lePAVD_iac":
        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
        optimizer = model.optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=param_dict["MODEL"]["OPTIMIZATION"]["DECAY_STEPS"])
        # scheduler = CosineAnnealingWarmRestarts(
        #         optimizer,
        #         T_0=50,       # restart every 10 epochs initially
        #         T_mult=2,     # then 20, 40, ...
        #         eta_min=1e-8  # floor LR
        #     )
        weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
    else:
        optimizer = model.optimizer
        weights = torch.tensor([1.0, 1.0, 1.0]).to(device)

    # Set loss function
    loss_fn_name = param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]

    best_epoch = 0
    for epoch in range(model.epochs):
        
        train_steps = 0
        train_loss_accum = 0.0

        if model.is_rnn:
            h = model.init_hidden(model.batch_size)

        for inputs, labels, norm_inputs in train_data_loader:
            inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)

            if model.is_rnn:
                h = h.data

            model.zero_grad()

            if model.is_rnn:

                output, h, _ = model(inputs, norm_inputs, h)
            else:
                output, _, _ = model(inputs, norm_inputs)

            if loss_fn_name == "Huber":
                loss = model.weighted_huber_loss(output, labels, weights)
            elif loss_fn_name == "MSE":
                loss = model.weighted_mse_loss(output, labels, weights)

            train_loss_accum += loss.item()
            train_steps += 1

            loss.backward()

            if param_dict['MODEL']['NAME'] == "lePAVD_iac":
                # 🔧 Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), param_dict["MODEL"]["OPTIMIZATION"]["GRAD_CLIP"])

            # ✅ Weight update and scheduler
            optimizer.step()

            if param_dict['MODEL']['NAME'] == "lePAVD_iac":
                scheduler.step()

        # Validation loop
        model.eval()
        val_steps = 0
        val_loss_accum = 0.0

        for inp, lab, norm_inp in val_data_loader:
            if model.is_rnn:
                val_h = model.init_hidden(inp.shape[0])
            inp, lab, norm_inp = inp.to(device), lab.to(device), norm_inp.to(device)

            if model.is_rnn:
                val_h = val_h.data
                out, val_h, _ = model(inp, norm_inp, val_h)
            else:
                out, _, _ = model(inp, norm_inp)

            if loss_fn_name == "Huber":
                val_loss = model.weighted_huber_loss(out, lab, weights)
            elif loss_fn_name == "MSE":
                val_loss = model.weighted_mse_loss(out, lab, weights)

            val_loss_accum += val_loss.item()
            val_steps += 1

        mean_train_loss = train_loss_accum / train_steps
        mean_val_loss = val_loss_accum / val_steps

        print(f"Epoch: {epoch+1}/{model.epochs}... Loss: {mean_train_loss:.8f}... Val Loss: {mean_val_loss:.8f}")

        if mean_val_loss < valid_loss_min:
            best_epoch = epoch
            torch.save(model.state_dict(), f"{output_dir}/epoch_{epoch+1}.pth")
            print(f"Validation loss decreased at epoch {epoch} >>> {epoch+1} ({valid_loss_min:.8f} --> {mean_val_loss:.8f}).  Saving model ...")
            valid_loss_min = mean_val_loss

        if np.isnan(mean_val_loss):
            break
        
        model.train()  # Switch back to training mode for next epoch

    print("Training complete. Best model saved to with minimum validation loss.: {} at epoch {}".format(valid_loss_min, best_epoch + 1))
    return best_epoch + 1
    
if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Train a deep dynamics model.")
    parser.add_argument("model_cfg", type=str, help="Config file for model")
    parser.add_argument("dataset", type=str, help="Dataset file")
    parser.add_argument("experiment_name", type=str, help="Name for experiment")
    # parser.add_argument("--log_wandb", action='store_true', default=False, help="Log experiment in wandb")
    parser.add_argument("dataset_split", type=float, default=0.9, help="Train/Val split ratio")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    with open(argdict["model_cfg"], 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
    # print(param_dict)

    if param_dict["MODEL"]["NAME"] not in string_to_model:
        model = string_to_lePAVD[param_dict["MODEL"]["NAME"]](param_dict)
        data_npy = np.load(argdict["dataset"])
        dataset = string_to_LEdataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"])
    else:
        model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)    
        data_npy = np.load(argdict["dataset"])
        dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"])

    if not os.path.exists("../output"):
        os.mkdir("../output")
    if not os.path.exists("../output/%s" % (os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0])):
        os.mkdir("../output/%s" % (os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0]))
    output_dir = "../output/%s/%s" % (os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0], argdict["experiment_name"])

    if os.path.exists(output_dir):
        print("Experiment already exists. Deleting old experiment")
        shutil.rmtree(output_dir)
        print("Creating experiment directory: ", output_dir)
        os.makedirs(output_dir)
    else:
        print("Creating experiment directory")
        os.makedirs(output_dir)

    train_dataset, val_dataset = dataset.split(float(argdict["dataset_split"]))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model.batch_size, shuffle=False)
    
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(dataset.scaler, f)
    
        
    train(model, train_data_loader, val_data_loader, argdict["experiment_name"], output_dir)
        

    
