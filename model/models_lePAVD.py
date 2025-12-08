from torch import nn
from sklearn.preprocessing import StandardScaler
import torch
from build_network import build_network, string_to_torch, create_module
import yaml
import pickle
import numpy as np
from abc import abstractmethod
import pandas as pd
import math
import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def kmph_to_mps(kph):
    # Works for tensors
    return kph * (1000.0 / 3600.0)

def get_field(d, *names):
    # Returns first present key (tensor) or None
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


class DatasetBase(torch.utils.data.Dataset):
    def __init__(self, features, labels, scaler=None):
        self.X_data = torch.from_numpy(features).float().to(device)
        self.y_data = torch.from_numpy(labels).float().to(device)
        self.X_norm = torch.zeros(features.shape)
        num_instances, num_time_steps, num_features = features.shape
        train_data = features.reshape((-1, num_features))
        if scaler is None:
            self.scaler = StandardScaler()
            norm_train_data = self.scaler.fit_transform(train_data)
            self.X_norm = torch.from_numpy(norm_train_data.reshape((num_instances, num_time_steps, num_features))).float().to(device)
        else:
            self.scaler = scaler
            norm_train_data = self.scaler.transform(train_data)
            self.X_norm = torch.from_numpy(norm_train_data.reshape((num_instances, num_time_steps, num_features))).float().to(device)
    def __len__(self):
        return(self.X_data.shape[0])
    def __getitem__(self, idx):
        x = self.X_data[idx]
        y = self.y_data[idx]
        x_norm = self.X_norm[idx]
        return x, y, x_norm
    def split(self, percent):
        split_id = int(len(self)* percent)
        torch.manual_seed(47)
        return torch.utils.data.random_split(self, [split_id, (len(self) - split_id)])

class DeepDynamicsDataset(DatasetBase):
    def __init__(self, features, labels, scalers=None):
        super().__init__(features[:,:,:7], labels, scalers)
    

class ModelBase(nn.Module):
    def __init__(self, param_dict, output_module, eval=False):
        super().__init__()
        self.param_dict = param_dict
        layers = build_network(self.param_dict)
        self.batch_size = self.param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"]
        if self.param_dict["MODEL"]["LAYERS"][0].get("LAYERS"):
            self.is_rnn = True
            self.rnn_n_layers = self.param_dict["MODEL"]["LAYERS"][0].get("LAYERS")
            self.rnn_hiden_dim = self.param_dict["MODEL"]["HORIZON"]
            layers.insert(1, nn.Flatten())
        else:
            self.is_rnn = False
        self.horizon = self.param_dict["MODEL"]["HORIZON"]
        layers.extend(output_module)
        self.feed_forward = nn.ModuleList(layers)
        if eval:
            self.loss_function = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]](reduction='none')
        else:
            self.loss_function = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]]()
        self.optimizer = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["OPTIMIZER"]](self.parameters(), lr=self.param_dict["MODEL"]["OPTIMIZATION"]["LR"])
        self.epochs = self.param_dict["MODEL"]["OPTIMIZATION"]["NUM_EPOCHS"]
        self.state = list(self.param_dict["STATE"])
        self.actions = list(self.param_dict["ACTIONS"])
        self.sys_params = list([*(list(p.keys())[0] for p in self.param_dict["PARAMETERS"])])
        self.vehicle_specs = self.param_dict["VEHICLE_SPECS"]

    @abstractmethod
    def differential_equation(self, x, output):
        pass

    def forward(self, x, x_norm, h0=None):
        for i in range(len(self.feed_forward)):
            if i == 0:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0 = self.feed_forward[0](x_norm, h0)
                else:
                    ff = self.feed_forward[i](torch.reshape(x_norm, (len(x), -1)))
            else:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0 = self.feed_forward[0](ff, h0)
                else:
                    ff = self.feed_forward[i](ff)
        o = self.differential_equation(x, ff)
        return o, h0, ff


    def unpack_sys_params(self, o):
        sys_params_dict = dict()
        for i in range(len(self.sys_params)):
            sys_params_dict[self.sys_params[i]] = o[:,i]
        ground_truth_dict =  dict()
        for p in self.param_dict["PARAMETERS"]:
            ground_truth_dict.update(p)
        return sys_params_dict, ground_truth_dict

    def unpack_state_actions(self, x):
        state_action_dict = dict()
        global_index = 0
        for i in range(len(self.state)):
            state_action_dict[self.state[i]] = x[:,-1, global_index]
            global_index += 1
        for i in range(len(self.actions)):
            state_action_dict[self.actions[i]] = x[:,-1, global_index]
            global_index += 1
        return state_action_dict

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_n_layers, batch_size, self.rnn_hiden_dim).zero_().to(device)
        return hidden
    
    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2).mean()

    def weighted_huber_loss(self, input, target, weight, delta=1.0):
        error = input - target
        abs_error = torch.abs(error)
        
        quadratic = torch.minimum(abs_error, torch.tensor(delta).to(input.device))
        linear = abs_error - quadratic

        loss = 0.5 * quadratic ** 2 + delta * linear  # element-wise Huber loss

        if weight is not None:
            loss = loss * weight  # apply per-output weights

        return loss.mean()



class lePAVD(ModelBase):
    def __init__(self, param_dict, eval=False, csv_path='None'):
        self.eval_mode = eval
        self.csv_path = csv_path
        print("Eval mode: ", self.eval_mode)
        class GuardLayer(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                guard_output = create_module("DENSE", 
                                             param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"], 
                                             param_dict["MODEL"]["HORIZON"], 
                                             len(param_dict["PARAMETERS"]), 
                                             activation="Sigmoid")
                self.guard_dense = guard_output[0]
                self.guard_activation = guard_output[1]
                self.coefficient_ranges = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                self.coefficient_mins = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                for i in range(len(param_dict["PARAMETERS"])):
                    self.coefficient_ranges[i] = param_dict["PARAMETERS"][i]["Max"]- param_dict["PARAMETERS"][i]["Min"]
                    self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

            def forward(self, x):
                guard_output = self.guard_dense(x)
                guard_output = self.guard_activation(guard_output) * self.coefficient_ranges + self.coefficient_mins
                return guard_output
            
        super().__init__(param_dict, [GuardLayer(param_dict)], eval)

    def differential_equation(self, x, output, Ts=0.02):
        
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)

        # Steering and Throttle Inputs
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]

        # Tire Slip Angles
        alphaf = steering - torch.atan2(
            self.vehicle_specs["lf"] * state_action_dict["YAW_RATE"] + state_action_dict["VY"],
            torch.abs(state_action_dict["VX"])
        ) + sys_param_dict["Shf"]

        alphar = torch.atan2(
            self.vehicle_specs["lr"] * state_action_dict["YAW_RATE"] - state_action_dict["VY"],
            torch.abs(state_action_dict["VX"])
        ) + sys_param_dict["Shr"]

        # Load Transfer
        Fz_f = (self.vehicle_specs["mass"] * 9.81 * self.vehicle_specs["lr"] -
                self.vehicle_specs["h_cg"] * self.vehicle_specs["mass"] * state_action_dict["VX"]) / (
                self.vehicle_specs["lf"] + self.vehicle_specs["lr"])
        Fz_r = (self.vehicle_specs["mass"] * 9.81 * self.vehicle_specs["lf"] +
                self.vehicle_specs["h_cg"] * self.vehicle_specs["mass"] * state_action_dict["VX"]) / (
                self.vehicle_specs["lf"] + self.vehicle_specs["lr"])

        self.vehicle_specs["Fz0"] = self.vehicle_specs["mass"] * 9.81 / 4


        # Forces
        Frx = ((sys_param_dict["Cm1"] - sys_param_dict["Cm2"] * state_action_dict["VX"]) * throttle
            - sys_param_dict["Cr0"] - sys_param_dict["Cr2"] * state_action_dict["VX"]**2)

        # --- Front lateral with curvature E_f ---
        tBaf   = sys_param_dict["Bf"] * alphaf
        thetaf = torch.atan(tBaf - sys_param_dict["Ef"] * (tBaf - torch.atan(tBaf)))
        Ffy = (sys_param_dict["Svf"]
            + sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * thetaf)) \
            * (Fz_f / self.vehicle_specs["Fz0"])

        # --- Rear lateral with curvature E_r ---
        tBar   = sys_param_dict["Br"] * alphar
        thetar = torch.atan(tBar - sys_param_dict["Er"] * (tBar - torch.atan(tBar)))
        Fry = (sys_param_dict["Svr"]
            + sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * thetar)) \
            * (Fz_r / self.vehicle_specs["Fz0"])

        # Accelerations (Euler version)
        ax = (Frx - Ffy * torch.sin(steering)) / self.vehicle_specs["mass"] + \
            state_action_dict["VY"] * state_action_dict["YAW_RATE"]
        ay = (Fry + Ffy * torch.cos(steering)) / self.vehicle_specs["mass"] - \
            state_action_dict["VX"] * state_action_dict["YAW_RATE"]
        ayaw = (Ffy * self.vehicle_specs["lf"] * torch.cos(steering) -
                Fry * self.vehicle_specs["lr"]) / sys_param_dict["Iz"]

        dxdt = torch.stack([ax, ay, ayaw], dim=1) * Ts
        
    #     # if self.eval_mode:
    #     #     df = pd.read_csv(self.csv_path)
    #     #     # print(Frx)
    #     #     vx, vy, yaw_rate = (x[:, -1, :3] + dxdt)[0].cpu().detach().tolist()
    #     #     new_row = {
    #     #         'Vx': vx,
    #     #         'Vy': vy,
    #     #         'yaw_rate': yaw_rate,
    #     #         'Frx': Frx.cpu().detach().item(),
    #     #         'Ffy': Ffy.cpu().detach().item(),
    #     #         'Fry': Fry.cpu().detach().item(),
    #     #     }
    #     #     df.loc[len(df)] = new_row
    #     #     df.to_csv(self.csv_path, index=False)

    #     return x[:, -1, :3] + dxdt
    
        return x[:, -1, :3] + dxdt



# class BetterSlip(nn.Module):
#     """
#     Dynamic, load- and combined-slip–aware slip angles for single-track axles.
#     Implements:
#       - kinematic target (signed/clamped vx)
#       - relaxation length with load dependence: tau = sigma / |vx|
#       - optional combined-slip via kappa
#       - optional camber / compliance correction
#     """
#     def __init__(self, sigma0_f=2.0, sigma0_r=2.0, a_f=0.3, a_r=0.3,
#                  c_f=0.8, c_r=0.8,        # combined-slip strength
#                  k_gamma_f=0.0, k_gamma_r=0.0,
#                  k_comp_f=0.0,  k_comp_r=0.0,
#                  alpha_deg_cap=30.0):
#         super().__init__()
#         # make key scalars learnable (or set requires_grad=False to freeze)
#         self.sigma0_f = nn.Parameter(torch.tensor(float(sigma0_f)))
#         self.sigma0_r = nn.Parameter(torch.tensor(float(sigma0_r)))
#         self.a_f = nn.Parameter(torch.tensor(float(a_f)))
#         self.a_r = nn.Parameter(torch.tensor(float(a_r)))
#         self.c_f = nn.Parameter(torch.tensor(float(c_f)))
#         self.c_r = nn.Parameter(torch.tensor(float(c_r)))
#         self.k_gamma_f = nn.Parameter(torch.tensor(float(k_gamma_f)))
#         self.k_gamma_r = nn.Parameter(torch.tensor(float(k_gamma_r)))
#         self.k_comp_f  = nn.Parameter(torch.tensor(float(k_comp_f)))
#         self.k_comp_r  = nn.Parameter(torch.tensor(float(k_comp_r)))
#         self.alpha_cap = math.radians(alpha_deg_cap)

#         # persistent states (support batched shape)
#         self.register_buffer("alpha_f", torch.tensor(0.0))
#         self.register_buffer("alpha_r", torch.tensor(0.0))

#     def forward(self, vx, vy, w, delta, Shf, Shr, lf, lr, Ts,
#                 Fz_f, Fz_r, Fz0_f, Fz0_r,
#                 kappa_f=None, kappa_r=None,
#                 gamma_f=None, gamma_r=None,
#                 FyF_prev=None, FyR_prev=None,
#                 eps=1e-3):

#         # -- 1) Kinematic targets (signed/clamped vx) --
#         vx_eff = torch.sign(vx) * torch.clamp(vx.abs(), min=eps)
#         af_eq = delta - torch.atan2(lf * w + vy, vx_eff) + Shf
#         ar_eq = torch.atan2(lr * w - vy, vx_eff) + Shr

#         # -- 2) Relaxation length with load dependence --
#         load_f = torch.clamp(Fz_f / Fz0_f, min=0.05)
#         load_r = torch.clamp(Fz_r / Fz0_r, min=0.05)
#         sigma_f = self.sigma0_f * (load_f ** self.a_f)
#         sigma_r = self.sigma0_r * (load_r ** self.a_r)

        
#         # Guard τ against tiny values (avoid huge updates at near-zero speed)
#         tau_min = 1e-2  # ~10 ms; tune if needed
#         tau_f = torch.clamp(sigma_f / torch.clamp(vx.abs(), min=eps), min=tau_min)
#         tau_r = torch.clamp(sigma_r / torch.clamp(vx.abs(), min=eps), min=tau_min)

        
#         # (re)initialize buffers if shape changed — keep them as buffers
#         # seed
#         if self.alpha_f.ndim == 0 or self.alpha_f.shape != af_eq.shape:
#             with torch.no_grad():
#                 self.alpha_f = self.alpha_f.to(vx.device)
#                 self.alpha_r = self.alpha_r.to(vx.device)
#                 # resize & seed with current equilibrium (detached)
#                 self.alpha_f.resize_as_(af_eq).copy_(af_eq.detach())
#                 self.alpha_r.resize_as_(ar_eq).copy_(ar_eq.detach())


#         # DETACH previous state to break link to last graph
#         alpha_f_prev = self.alpha_f.detach()
#         alpha_r_prev = self.alpha_r.detach()

#         # Compute new (continuous) state for this step (gradients only through current vars)
#         alpha_f_new = alpha_f_prev + Ts * (af_eq - alpha_f_prev) / tau_f
#         alpha_r_new = alpha_r_prev + Ts * (ar_eq - alpha_r_prev) / tau_r

#         # -- 3) Combined-slip effect (optional) -- operate on *local* new variables
#         if kappa_f is not None: kappa_f = torch.clamp(kappa_f, -0.5, 0.5)
#         if kappa_r is not None: kappa_r = torch.clamp(kappa_r, -0.5, 0.5)

#         if kappa_f is not None:
#             alpha_f_new = torch.atan(torch.tan(alpha_f_new) /
#                                     torch.sqrt(1.0 + self.c_f * (kappa_f ** 2)))
#         if kappa_r is not None:
#             alpha_r_new = torch.atan(torch.tan(alpha_r_new) /
#                                     torch.sqrt(1.0 + self.c_r * (kappa_r ** 2)))

#         # ---- Ensure previous-force caches match shape/device; else ignore them ----
#         if FyF_prev is not None:
#             FyF_prev = FyF_prev.to(Fz_f.device)
#             if FyF_prev.numel() != Fz_f.numel():
#                 FyF_prev = None
#         if FyR_prev is not None:
#             FyR_prev = FyR_prev.to(Fz_r.device)
#             if FyR_prev.numel() != Fz_r.numel():
#                 FyR_prev = None

#         # -- 4) Camber / compliance (optional) -- still on *local* new variables
#         if gamma_f is not None:
#             alpha_f_new = alpha_f_new + self.k_gamma_f * gamma_f
#         if gamma_r is not None:
#             alpha_r_new = alpha_r_new + self.k_gamma_r * gamma_r
#         if FyF_prev is not None:
#             alpha_f_new = alpha_f_new + self.k_comp_f * FyF_prev / torch.clamp(Fz_f, min=eps)
#         if FyR_prev is not None:
#             alpha_r_new = alpha_r_new + self.k_comp_r * FyR_prev / torch.clamp(Fz_r, min=eps)

#         # Safety caps (still local)
#         alpha_f_out = torch.clamp(alpha_f_new, -self.alpha_cap, +self.alpha_cap)
#         alpha_r_out = torch.clamp(alpha_r_new, -self.alpha_cap, +self.alpha_cap)

#         # Store DETACHED copies back into the buffers IN-PLACE (preserves buffer registration)
#         with torch.no_grad():
#             self.alpha_f.copy_(alpha_f_out.detach())
#             self.alpha_r.copy_(alpha_r_out.detach())

#         return alpha_f_out, alpha_r_out



# class lePAVDIAC(ModelBase):
#     def __init__(self, param_dict, eval=False, csv_path='None'):
#         self.eval_mode = eval
#         self.csv_path = csv_path

#         class GuardLayer(nn.Module):
#             def __init__(self, param_dict):
#                 super().__init__()
#                 guard_output = create_module("DENSE", param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"], 
#                                              param_dict["MODEL"]["HORIZON"], len(param_dict["PARAMETERS"]), 
#                                              activation="Sigmoid")
#                 self.guard_dense = guard_output[0]
#                 self.guard_activation = guard_output[1]
#                 self.coefficient_ranges = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
#                 self.coefficient_mins = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
#                 for i in range(len(param_dict["PARAMETERS"])):
#                     self.coefficient_ranges[i] = param_dict["PARAMETERS"][i]["Max"]- param_dict["PARAMETERS"][i]["Min"]
#                     self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

#             def forward(self, x):
#                 guard_output = self.guard_dense(x)
#                 guard_output = self.guard_activation(guard_output) * self.coefficient_ranges + self.coefficient_mins
#                 return guard_output

        
#         super().__init__(param_dict, [GuardLayer(param_dict)], eval)


#         # ---- NEW: slip module + toggles for ablations ----
#         self.slip = BetterSlip(sigma0_f=2.0, sigma0_r=2.0, a_f=0.3, a_r=0.3,
#                                c_f=0.8, c_r=0.8, k_gamma_f=0.0, k_gamma_r=0.0,
#                                k_comp_f=0.0, k_comp_r=0.0, alpha_deg_cap=30.0)

#         self.use_combined_slip = True      # set False to ablate
#         self.use_camber_proxy  = True    # set True if you want roll->camber proxy
#         # --- Transient load smoothing (suspension-aware) ---
#         self.use_Fz_smoothing = True   # set False to ablate

#         # learnable positive time constants (seconds)
#         self._tauFz_f_raw = nn.Parameter(torch.tensor(0.08))  # ~80 ms to start
#         self._tauFz_r_raw = nn.Parameter(torch.tensor(0.10))  # ~100 ms to start

#         # running smoothed loads (buffers, shape set on first use)
#         self.register_buffer("Fz_f_smooth", torch.tensor(0.0))
#         self.register_buffer("Fz_r_smooth", torch.tensor(0.0))

        
#         # ADD THIS: learnable roll→camber gain (can start at 1.0 or 0.0)
#         self.k_gamma = nn.Parameter(torch.tensor(1.0))


#     def differential_equation(self, x, output, Ts=0.04):
#         sys, _ = self.unpack_sys_params(output)
#         s = self.unpack_state_actions(x)

#         # --- shorthand specs ---
#         m   = self.vehicle_specs["mass"]
#         g   = 9.81
#         lf  = self.vehicle_specs["lf"]
#         lr  = self.vehicle_specs["lr"]
#         hcg = self.vehicle_specs["h_cg"]

#         # --- inputs (cmd + fb) ---
#         steering = s["STEERING_FB"] + s["STEERING_CMD"]
#         throttle = s["THROTTLE_FB"] + s["THROTTLE_CMD"]

#         # --- base signals ---
#         vx = s["VX"]; vy = s["VY"]; w = s["YAW_RATE"]

#         # right after: vx = s["VX"]; vy = s["VY"]; w = s["YAW_RATE"]
#         B = vx.shape[0]
#         if getattr(self, "_prev_B", None) != B:
#             # drop cross-batch caches that depend on batch shape
#             self._FyF_prev = None
#             self._FyR_prev = None
#             # optional: reset slip buffers if batches are unrelated clips
#             self.slip.alpha_f.zero_()
#             self.slip.alpha_r.zero_()
#             self.Fz_f_smooth.zero_()
#             self.Fz_r_smooth.zero_()
#             self._prev_B = B

#         # --- kappa from wheel speeds (km/h -> m/s). If missing, returns None ---
#         v_fl = get_field(s, "WHEEL_FL", "wheel_fl")
#         v_fr = get_field(s, "WHEEL_FR", "wheel_fr")
#         v_rl = get_field(s, "WHEEL_RL", "wheel_rl")
#         v_rr = get_field(s, "WHEEL_RR", "wheel_rr")

#         if (v_fl is not None) and (v_fr is not None):
#             v_wf = 0.5 * (kmph_to_mps(v_fl) + kmph_to_mps(v_fr))
#         else:
#             v_wf = None

#         if (v_rl is not None) and (v_rr is not None):
#             v_wr = 0.5 * (kmph_to_mps(v_rl) + kmph_to_mps(v_rr))
#         else:
#             v_wr = None

#         eps = 1e-3
#         den = torch.sign(vx) * torch.clamp(vx.abs(), min=eps)

#         kappa_front = (v_wf - vx) / den if v_wf is not None else None
#         kappa_rear  = (v_wr - vx) / den if v_wr is not None else None
#         if not self.use_combined_slip:
#             kappa_front, kappa_rear = None, None

#         # --- camber proxy from roll (optional) ---
#         if self.use_camber_proxy:
#             roll = get_field(s, "ROLL", "roll")  # radians
#             if roll is not None:
#                 gamma_front = -self.k_gamma * roll   # flip sign later if needed
#                 gamma_rear  = -self.k_gamma * roll
#             else:
#                 gamma_front = gamma_rear = None
#         else:
#             gamma_front = gamma_rear = None

#         # --- longitudinal force (rear, RWD) ---
#         Frx = ((sys["Cm1"] - sys["Cm2"] * vx) * throttle
#             - sys["Cr0"] - sys["Cr2"] * (vx ** 2))

#         # --- load transfer via measured ax if present, else use Frx/m ---
#         ax_meas = get_field(s, "AX", "ax")
#         ax_long = ax_meas if ax_meas is not None else (Frx / m)

#         # --- per-axle static nominal loads (single-track) ---
#         Fz0_f = m * g * lr / (lf + lr)
#         Fz0_r = m * g * lf / (lf + lr)

#         # --- dynamic Fz with ax (guard > 0) ---
#         Fz_f = (m * g * lr - hcg * m * ax_long) / (lf + lr)
#         Fz_r = (m * g * lf + hcg * m * ax_long) / (lf + lr)
#         Fz_f = torch.clamp(Fz_f, min=0.05 * Fz0_f)
#         Fz_r = torch.clamp(Fz_r, min=0.05 * Fz0_r)

#         # --- NEW: transient load smoothing (suspension-aware) ---
#         if self.use_Fz_smoothing:
#             # positive time constants with a small floor for stability
#             tau_min = Ts  # 20 ms
#             tauF_f = torch.nn.functional.softplus(self._tauFz_f_raw) + tau_min
#             tauF_r = torch.nn.functional.softplus(self._tauFz_r_raw) + tau_min

#             # (re)seed buffers on first use / shape change
#             if self.Fz_f_smooth.ndim == 0 or self.Fz_f_smooth.shape != Fz_f.shape:
#                 with torch.no_grad():
#                     self.Fz_f_smooth = self.Fz_f_smooth.to(Fz_f.device)
#                     self.Fz_r_smooth = self.Fz_r_smooth.to(Fz_r.device)
#                     self.Fz_f_smooth.resize_as_(Fz_f).copy_(Fz_f.detach())
#                     self.Fz_r_smooth.resize_as_(Fz_r).copy_(Fz_r.detach())

#             # Euler low-pass:  Fz_s[k+1] = Fz_s[k] + (Ts/tau) * (Fz - Fz_s[k])
#             alpha_f = torch.clamp(Ts / tauF_f, max=1.0)
#             alpha_r = torch.clamp(Ts / tauF_r, max=1.0)

#             Fz_f_new = self.Fz_f_smooth + alpha_f * (Fz_f - self.Fz_f_smooth)
#             Fz_r_new = self.Fz_r_smooth + alpha_r * (Fz_r - self.Fz_r_smooth)

#             # outputs for this step
#             Fz_f_used = Fz_f_new
#             Fz_r_used = Fz_r_new

#             # store detached for next call
#             with torch.no_grad():
#                 self.Fz_f_smooth.copy_(Fz_f_new.detach())
#                 self.Fz_r_smooth.copy_(Fz_r_new.detach())
#         else:
#             Fz_f_used = Fz_f
#             Fz_r_used = Fz_r


#         # --- improved slip angles (dynamic, load-aware, combined-slip aware) ---
#         alphaf, alphar = self.slip(
#             vx=vx, vy=vy, w=w, delta=steering,
#             Shf=sys["Shf"], Shr=sys["Shr"], lf=lf, lr=lr, Ts=Ts,
#             Fz_f=Fz_f_used, Fz_r=Fz_r_used, Fz0_f=Fz0_f, Fz0_r=Fz0_r,
#             kappa_f=kappa_front, kappa_r=kappa_rear,
#             gamma_f=gamma_front, gamma_r=gamma_rear,
#             FyF_prev=getattr(self, "_FyF_prev", None),
#             FyR_prev=getattr(self, "_FyR_prev", None),
#         )

#         # --- lateral forces (Pacejka curvature) with per-axle Fz scaling ---
#         # Front
#         tBaf   = sys["Bf"] * alphaf
#         thetaf = torch.atan(tBaf - sys["Ef"] * (tBaf - torch.atan(tBaf)))
#         FyF_raw = sys["Svf"] + sys["Df"] * torch.sin(sys["Cf"] * thetaf)
#         Ffy = FyF_raw * (Fz_f_used / Fz0_f)

#         # Rear
#         tBar   = sys["Br"] * alphar
#         thetar = torch.atan(tBar - sys["Er"] * (tBar - torch.atan(tBar)))
#         FyR_raw = sys["Svr"] + sys["Dr"] * torch.sin(sys["Cr"] * thetar)
#         Fry = FyR_raw * (Fz_r_used / Fz0_r)

#         # --- planar accelerations (Euler) ---
#         ax = (Frx - Ffy * torch.sin(steering)) / m + vy * w
#         ay = (Fry + Ffy * torch.cos(steering)) / m - vx * w
#         ayaw = (Ffy * lf * torch.cos(steering) - Fry * lr) / sys["Iz"]

#         dxdt = torch.stack([ax, ay, ayaw], dim=1) * Ts

#         # store forces for optional compliance next step
#         self._FyF_prev = Ffy.detach()
#         self._FyR_prev = Fry.detach()

#         return x[:, -1, :3] + dxdt

class lePAVDIAC(ModelBase):
    def __init__(self, param_dict, eval_mode=False, csv_path='None'):
        self.eval_mode = eval_mode
        self.csv_path = csv_path

        class GuardLayer(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                guard_output = create_module("DENSE", param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"], 
                                             param_dict["MODEL"]["HORIZON"], len(param_dict["PARAMETERS"]), 
                                             activation="Sigmoid")
                self.guard_dense = guard_output[0]
                self.guard_activation = guard_output[1]
                self.coefficient_ranges = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                self.coefficient_mins = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                for i in range(len(param_dict["PARAMETERS"])):
                    self.coefficient_ranges[i] = param_dict["PARAMETERS"][i]["Max"]- param_dict["PARAMETERS"][i]["Min"]
                    self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

            def forward(self, x):
                guard_output = self.guard_dense(x)
                guard_output = self.guard_activation(guard_output) * self.coefficient_ranges + self.coefficient_mins
                return guard_output

        
        super().__init__(param_dict, [GuardLayer(param_dict)], eval_mode)


    def differential_equation(self, x, output, Ts=0.04):
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)

        # Inputs
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]

        # Slip angles (old kinematics)
        alphaf = steering - torch.atan2(self.vehicle_specs["lf"] * state_action_dict["YAW_RATE"] + state_action_dict["VY"],
                                        torch.abs(state_action_dict["VX"])) + sys_param_dict["Shf"]
        alphar = torch.atan2(self.vehicle_specs["lr"] * state_action_dict["YAW_RATE"] - state_action_dict["VY"],
                            torch.abs(state_action_dict["VX"])) + sys_param_dict["Shr"]

        # Longitudinal force (unchanged)
        Frx = ((sys_param_dict["Cm1"] - sys_param_dict["Cm2"] * state_action_dict["VX"]) * throttle
            - sys_param_dict["Cr0"] - sys_param_dict["Cr2"] * (state_action_dict["VX"] ** 2)) 


        # ---- Load transfer ----
        m   = self.vehicle_specs["mass"]
        g   = 9.81
        lf  = self.vehicle_specs["lf"]
        lr  = self.vehicle_specs["lr"]
        hcg = self.vehicle_specs["h_cg"]
        ax_long = Frx / m 
        

        # Dynamic axle loads (longitudinal transfer)
        Fz_f = (m * g * lr - hcg * m * ax_long) / (lf + lr)
        Fz_r = (m * g * lf + hcg * m * ax_long) / (lf + lr)

        
        Fz0 = self.vehicle_specs["mass"] * 9.81 / 4

        # Lateral forces (dynamic) with correct axle scaling (suspension aware)
        tBaf   = sys_param_dict["Bf"] * alphaf
        thetaf = torch.atan(tBaf - sys_param_dict["Ef"] * (tBaf - torch.atan(tBaf)))
        Ffy = (sys_param_dict["Svf"]
            + sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * thetaf)) * (Fz_f / Fz0)

        tBar   = sys_param_dict["Br"] * alphar
        thetar = torch.atan(tBar - sys_param_dict["Er"] * (tBar - torch.atan(tBar)))
        Fry = (sys_param_dict["Svr"]
            + sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * thetar)) * (Fz_r / Fz0)


        # Accelerations (unchanged)
        ax = (Frx - Ffy * torch.sin(steering)) / m + state_action_dict["VY"] * state_action_dict["YAW_RATE"]
        ay = (Fry + Ffy * torch.cos(steering)) / m - state_action_dict["VX"] * state_action_dict["YAW_RATE"]
        ayaw = (Ffy * lf * torch.cos(steering) - Fry * lr) / sys_param_dict["Iz"]

        dxdt = torch.stack([ax, ay, ayaw], dim=1) * Ts
        
        
        if self.eval_mode:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            # print(Frx)
            vx, vy, yaw_rate = (x[:, -1, :3] + dxdt)[0].cpu().detach().tolist()
            # print(vx, vy, yaw_rate)
            new_row = {
                'Vx': vx,
                'Vy': vy,
                'yaw_rate': yaw_rate,
                'Frx': Frx.cpu().detach().item(),
                'Ffy': Ffy.cpu().detach().item(),
                'Fry': Fry.cpu().detach().item(),
            }
            df.loc[len(df)] = new_row
            df.to_csv(self.csv_path, index=False)
        return x[:, -1, :3] + dxdt





    # def differential_equation(self, x, output, Ts=0.04):
    #     sys_param_dict, _ = self.unpack_sys_params(output)
    #     state_action_dict = self.unpack_state_actions(x)

    #     # Steering and Throttle Inputs
    #     steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
    #     throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]


    #     # Tire Slip Angles
    #     alphaf = steering - torch.atan2(
    #         self.vehicle_specs["lf"] * state_action_dict["YAW_RATE"] + state_action_dict["VY"],
    #         torch.abs(state_action_dict["VX"])
    #     ) + sys_param_dict["Shf"]

    #     alphar = torch.atan2(
    #         self.vehicle_specs["lr"] * state_action_dict["YAW_RATE"] - state_action_dict["VY"],
    #         torch.abs(state_action_dict["VX"])
    #     ) + sys_param_dict["Shr"]


        # Load Transfer
        # Fz_f = (self.vehicle_specs["mass"] * 9.81 * self.vehicle_specs["lr"] -
        #         self.vehicle_specs["h_cg"] * self.vehicle_specs["mass"] * state_action_dict["VX"]) / (
        #         self.vehicle_specs["lf"] + self.vehicle_specs["lr"])
        # Fz_r = (self.vehicle_specs["mass"] * 9.81 * self.vehicle_specs["lf"] +
        #         self.vehicle_specs["h_cg"] * self.vehicle_specs["mass"] * state_action_dict["VX"]) / (
        #         self.vehicle_specs["lf"] + self.vehicle_specs["lr"])

    #     self.vehicle_specs["Fz0"] = self.vehicle_specs["mass"] * 9.81 / 4

    #     # Forces
    #     Frx = ((sys_param_dict["Cm1"] - sys_param_dict["Cm2"] * state_action_dict["VX"]) * throttle
    #         - sys_param_dict["Cr0"] - sys_param_dict["Cr2"] * state_action_dict["VX"]**2)

    #     # --- Front lateral with curvature E_f ---
    #     tBaf   = sys_param_dict["Bf"] * alphaf
    #     thetaf = torch.atan(tBaf - sys_param_dict["Ef"] * (tBaf - torch.atan(tBaf)))
    #     Ffy = (sys_param_dict["Svf"]
    #         + sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * thetaf)) \
    #         * (Fz_f / self.vehicle_specs["Fz0"])

    #     # --- Rear lateral with curvature E_r ---
    #     tBar   = sys_param_dict["Br"] * alphar
    #     thetar = torch.atan(tBar - sys_param_dict["Er"] * (tBar - torch.atan(tBar)))
    #     Fry = (sys_param_dict["Svr"]
    #         + sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * thetar)) \
    #         * (Fz_r / self.vehicle_specs["Fz0"])


    #     # Accelerations (lePAVD)
    #     ax = (Frx - Ffy * torch.sin(steering)) / self.vehicle_specs["mass"] + \
    #         state_action_dict["VY"] * state_action_dict["YAW_RATE"]
    #     ay = (Fry + Ffy * torch.cos(steering)) / self.vehicle_specs["mass"] - \
    #         state_action_dict["VX"] * state_action_dict["YAW_RATE"]
    #     ayaw = (Ffy * self.vehicle_specs["lf"] * torch.cos(steering) -
    #             Fry * self.vehicle_specs["lr"]) / sys_param_dict["Iz"]

    #     dxdt = torch.stack([ax, ay, ayaw], dim=1) * Ts

    #     return x[:, -1, :3] + dxdt
    #     # if self.eval_mode:
    #     #     import pandas as pd
    #     #     df = pd.read_csv(self.csv_path)
    #     #     # print(Frx)
    #     #     vx, vy, yaw_rate = (x[:, -1, :3] + dxdt)[0].cpu().detach().tolist()
    #     #     # print(vx, vy, yaw_rate)
    #     #     new_row = {
    #     #         'Vx': vx,
    #     #         'Vy': vy,
    #     #         'yaw_rate': yaw_rate,
    #     #         'Frx': Frx.cpu().detach().item(),
    #     #         'Ffy': Ffy.cpu().detach().item(),
    #     #         'Fry': Fry.cpu().detach().item(),
    #     #     }
    #     #     df.loc[len(df)] = new_row
    #     #     df.to_csv(self.csv_path, index=False)
        
    #     # return x[:, -1, :3] + dxdt




string_to_lePAVD = {
    "lePAVD" : lePAVD,
    "lePAVD_iac" : lePAVDIAC,
}

string_to_LEdataset = {
    "lePAVD" : DeepDynamicsDataset,
    "lePAVD_iac" : DeepDynamicsDataset,
}

