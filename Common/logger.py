from fileinput import filename
import os
import numpy as np
from datetime import datetime
from abc import ABC
from Common import *
from .runtime_paths import HELVETICA_FONT_PATHS
import matplotlib.pyplot as plt
import imageio


class Logger(ABC):
    def __init__(self, ckpt_dir: str, log_dir: str, plot_dir: str):
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.plot_dir = plot_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        print(f"Log dir: {log_dir}")
        print(f"Checkpoint dir: {ckpt_dir}")
        print(f"Plot dir: {plot_dir}")

        self.log_dict = {}

    def _create_log_file(self, model_type):
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ending = f"{model_type}_{timestamp}"
        self.log_filename = f"{self.log_dir}/log_{self.ending}.csv"
        with open(self.log_filename, "w") as f:
            f.write("step,\n")  # header will be updated after first log

    def _log_step(self, step: int):
        """Call this at each logging step to write to CSV."""
        if step == 0 and self.log_dict:
            # Update header line with keys
            with open(self.log_filename, "r+") as f:
                lines = f.readlines()
                lines[0] = f"step, {', '.join(self.log_dict.keys())}\n"
                f.seek(0)
                f.writelines(lines)

        with open(self.log_filename, "a") as f:
            line = f"{step}, " + ", ".join(str(self.log_dict.get(k, '')) for k in self.log_dict.keys()) + "\n"
            f.write(line)

    @property
    def ckpt_filename(self):
        """Checkpoint filename with the current ending."""
        return f"{self.ckpt_dir}/ckpt_{self.ending}"

    
    def process_loss(self, loss):
        """
        Method to extract the outputs of the loss function you wish to log
        should return a dictionary of values to log.
        """
        if isinstance(loss, dict):
            return loss
        else:
            return {"loss": loss}
        
    def load_logs(self, filename):
        """
        Load logs from a CSV file.
        """
        logs_loaded = np.genfromtxt(filename, delimiter=",", names=True, dtype=None, encoding=None)
        logs_loaded = {name: logs_loaded[name] for name in logs_loaded.dtype.names}
        return logs_loaded

    def plot_logs(self, filenames=None, labels=None, fields=None):
        
        if filenames is None:
            filenames = [self.log_filename]

        if fields is None:
            logs_loaded = self.load_logs(filenames[0])
            #fields = ["loss", "loss_teacher_forcing", "loss_rollout", "lipschitz_loss_dynamics", "loss_observer", "loss_from_estimated", "lipschitz_loss_observer"]
            fields = list(logs_loaded.keys())
            #keys = list(logs_loaded.keys())
            # for key in keys:
            #     if "cost" in key:
            #         fields.append(key)



        for field in fields:
            for i, filename in enumerate(filenames):
                logs_loaded = self.load_logs(filename)
                # make any nans the same as their previous value
                #logs_loaded[field] = np.nan_to_num(logs_loaded[field], nan=np.nanmean(logs_loaded[field]))
                if labels is None:
                    plt.plot(logs_loaded["step"], logs_loaded[field], label=f"{i}")
                else:
                    plt.plot(logs_loaded["step"], logs_loaded[field], label=labels[i])

            plt.grid(which="both")
            plt.xlabel("Step")
            if "cost" in field:
                plt.ylabel("Cost")
            else:
                plt.ylabel("Loss")
            plt.yscale("log")
            plt.savefig(f"{self.plot_dir}/LOG_{field}.png", dpi=300)
            plt.clf()
            plt.close()

    def plot_rollouts(self, rollout_preds, rollout_true, mode="DYNAMICS"):
        """
        Plot the rollouts.
        """
        index_map = rollout_preds.index_map

        assert mode in ["DYNAMICS", "OBSERVER"], f"Invalid mode: {mode}. Choose either 'DYNAMICS' or 'OBSERVER'."

        for i, k in enumerate(index_map.keys()):

            predicted = getattr(rollout_preds, k).flatten()
            true = getattr(rollout_true, k).flatten()
            predicted = np.asarray(predicted)
            true = np.asarray(true)
            max_true = np.max(true)
            min_true = np.min(true)
            upper_bound = max_true + 0.1 * (max_true - min_true)
            lower_bound = min_true - 0.1 * (max_true - min_true)
            dt = 0.02
            max_time = (predicted.shape[0] + 1) * dt
            time = np.linspace(0.02, max_time, predicted.shape[0])
            plt.figure(figsize=(10, 5))
            plt.plot(time, predicted, linewidth=2, color="blue", alpha=0.5)
            plt.plot(time, true, linewidth=2, color="blue")
            plt.xlabel("Time (s)")
            plt.xlim([0.02-0.005, max_time+0.005])
            plt.ylim([lower_bound, upper_bound])
            plt.ylabel(k)
            plt.savefig(f"{self.plot_dir}/{mode}_ROLLOUT_{k}.png", dpi=300)
            plt.clf()
            plt.close()

    def save_video(self, frames, filename):
        """
        Save a video from frames.
        """
        name_prefix = f"VIDEO_{filename}"  # just the prefix, no directory
        existing_files = [f for f in os.listdir(self.plot_dir) if f.startswith(name_prefix)]
        n_files = len(existing_files)
        file_num = n_files + 1
        full_filename = os.path.join(self.plot_dir, f"{name_prefix}_{file_num}.mp4")
        imageio.mimwrite(full_filename, frames, fps=50)

    def save_blender_data(self, positions, quaternions, filename):
        """
        Save blender data from positions and quaternions.
        """
        name_prefix = f"BLENDER_{filename}"  # just the prefix, no directory
        existing_files = [f for f in os.listdir(self.plot_dir) if f.startswith(name_prefix)]
        n_files = len(existing_files)
        file_num = n_files + 1
        full_filename = os.path.join(self.plot_dir, f"{name_prefix}_{file_num}.npz")
        np.savez(full_filename, positions=positions, quaternions=quaternions)

    def save_mj_dataset(self, mj_dataset, filename):
        """
        Save blender data from positions and quaternions.
        """
        name_prefix = f"MJDATASET_{filename}"  # just the prefix, no directory
        existing_files = [f for f in os.listdir(self.plot_dir) if f.startswith(name_prefix)]
        n_files = len(existing_files)
        file_num = n_files + 1
        full_filename = os.path.join(self.plot_dir, f"{name_prefix}_{file_num}.npz")
        # get attribute names from mj_dataset
        attr_names = [attr for attr in dir(mj_dataset) if not attr.startswith("_") and not callable(getattr(mj_dataset, attr))]
        # create a dictionary to hold the data
        data_dict = {}
        for attr in attr_names:
            data_dict[attr] = getattr(mj_dataset, attr).array

        np.savez(full_filename, **data_dict)

    def save_tracking_error(self, tracking_error, filename):
        """
        Save tracking error data.
        """
        name_prefix = f"TRACKING_ERROR_{filename}"  # just the prefix, no directory
        existing_files = [f for f in os.listdir(self.plot_dir) if f.startswith(name_prefix)]
        n_files = len(existing_files)
        file_num = n_files + 1
        full_filename = os.path.join(self.plot_dir, f"{name_prefix}_{file_num}.npy")
        print(f"Tracking error: {tracking_error}")
        print(f"Tracking error: {tracking_error.shape}")
        np.save(full_filename, tracking_error)

    def plot_evaluation(self, filenames=None, model_names=None, labels=None, fields=None, aux_filenames=None):
        import numpy as np
        import scipy.stats as st
        from collections import defaultdict
        import matplotlib.pyplot as plt

        import matplotlib as mpl
        import matplotlib.font_manager as fm
        # font_path = "./OpenSans-Regular.ttf"
        font_paths = HELVETICA_FONT_PATHS
        # Register the font with Matplotlib
        fm.fontManager.addfont(font_paths[0])
        fm.fontManager.addfont(font_paths[1])
        fm.fontManager.addfont(font_paths[2])

        # Set it as the default font
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "pdf.fonttype": 42,  # TrueType for PDF compatibility
            "ps.fonttype": 42,
        })
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = "Helvetica"
        plt.rcParams["mathtext.it"] = "Helvetica:oblique"
        plt.rcParams["mathtext.bf"] = "Helvetica:bold"


        # def pastelize(rgb, factor=0.5):
        #     """Lighten and desaturate RGB color toward white by the given factor."""
        #     return tuple([1 - factor * (1 - c) for c in rgb])

        # base_tableau = plt.colormaps["tab10"].colors
        # pastel_tableau = [pastelize(c, factor=0.8) for c in base_tableau]
        # # pastel_tableau = pastel_tableau[1:] + [pastel_tableau[0]]  # move blue to end

        # plt.rcParams["axes.prop_cycle"] = plt.cycler(color=pastel_tableau)

        # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if filenames is None:
            filenames = [self.log_filename]

        if fields is None:
            logs_loaded = self.load_logs(filenames[0])
            fields = list(logs_loaded.keys())


        total = []
        total_lower = []
        total_upper = []

        plt.figure(figsize=(8, 5))
        trot_bwds = []
        trot_bwds_lower = []
        trot_bwds_upper = []
        trot_fwds = []
        trot_fwds_lower = []
        trot_fwds_upper = []
        trot_inplaces = []
        trot_inplaces_lower = []
        trot_inplaces_upper = []
        trot_lefts_raw = []
        trot_rights_raw = []
        steps = []

        for field in fields:
            for i, filename in enumerate(filenames):
                logs_loaded = self.load_logs(filename)  # dict with keys like 'epoch', 'field1', 'field2'...

                # group values by epoch
                data = defaultdict(list)
                for e, v in zip(logs_loaded["epoch"], logs_loaded[field]):
                    data[int(e)].append(float(v))


                # compute mean + 95% CI
                epochs, means, lowers, uppers = [], [], [], []
                for e in sorted(data.keys()):
                    arr = np.array(data[e])
                    # mean = arr.mean()
                    # sem = st.sem(arr)
                    # ci_low, ci_high = st.t.interval(0.95, len(arr) - 1, loc=mean, scale=sem)
                    # epochs.append(e)
                    # means.append(mean)
                    # lowers.append(ci_low)
                    # uppers.append(ci_high)
                    log_arr = np.log(arr)
                    mean = log_arr.mean()
                    sem = st.sem(log_arr)
                    ci_low, ci_high = st.t.interval(0.95, len(log_arr) - 1, loc=mean, scale=sem)
                    epochs.append(e)
                    means.append(np.exp(mean))
                    lowers.append(np.exp(ci_low))
                    uppers.append(np.exp(ci_high))

                label = f"{i}" if labels is None else labels[i]
                if "trot_bwd_cost" in field:
                    trot_bwds.append(means)
                    trot_bwds_lower.append(lowers)
                    trot_bwds_upper.append(uppers)
                    steps.append(epochs)
                if "trot_fwd_cost" in field:
                    trot_fwds.append(means)
                    trot_fwds_lower.append(lowers)
                    trot_fwds_upper.append(uppers)
                if "trot_in_place_cost" in field:
                    trot_inplaces.append(means)
                    trot_inplaces_lower.append(lowers)
                    trot_inplaces_upper.append(uppers)
                if "trot_left_cost" in field:
                    trot_lefts_raw.append(data)
                if "trot_right_cost" in field:
                    trot_rights_raw.append(data)

                plt.plot(epochs, means, label=label)
                # delta_lower = np.array(means) - np.array(lowers)
                # delta_upper = np.array(uppers) - np.array(means)

                # log_delta_lower = np.log10(delta_lower + 1e-8)  # add small constant to avoid log(0)
                # log_delta_upper = np.log10(delta_upper + 1e-8)
                # log_lowers = np.array(means) - log_delta_lower
                # log_uppers = np.array(means) + log_delta_upper
                plt.fill_between(epochs, lowers, uppers, alpha=0.2)
                #plt.fill_between(epochs, log_lowers, log_uppers, alpha=0.2, color=colors[i % len(colors)])

            # plt.grid(which="both")
            plt.xlabel("Step", fontsize=18)
            plt.ylabel("Task cost" if "cost" in field else "Loss", fontsize=18)
            # ticks fontsize
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True, which='both', color='white', linewidth=1.5)
            plt.tick_params(axis='both', which='both', length=0)

            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            # facecolor
            plt.gca().set_facecolor((0.95, 0.95, 0.95))
            plt.tight_layout()
            plt.yscale("log")
            
            # plt.ylim(bottom=0)
            plt.savefig(f"{self.plot_dir}/LOG_{field}.png", dpi=300)
            plt.clf()
            plt.close()


        # for i, filename in enumerate(filenames):
        #     logs_loaded = self.load_logs(filename)
        #     file_means = []
        #     for field in fields:
        #         if "cost" in field:
        #           # dict with keys like 'epoch', 'field1', 'field2'...
        #             # group values by epoch
        #             data = defaultdict(list)
        #             for e, v in zip(logs_loaded["epoch"], logs_loaded[field]):
        #                 data[int(e)].append(float(v))

        #             # compute mean + 95% CI
        #             epochs, means, lowers, uppers = [], [], [], []
        #             for e in sorted(data.keys()):
        #                 arr = np.array(data[e])
        #                 log_arr = np.log(arr)
        #                 mean = log_arr.mean()
        #                 sem = st.sem(log_arr)
        #                 ci_low, ci_high = st.t.interval(0.95, len(log_arr) - 1, loc=mean, scale=sem)
        #                 epochs.append(e)
        #                 means.append(np.exp(mean))
        #                 lowers.append(np.exp(ci_low))
        #                 uppers.append(np.exp(ci_high))

        #             file_means.append(means)
        #     total.append(file_means)
        # total = np.array(total)  # shape (n_files, n_fields, n_epochs)
        # # total_mean = np.mean(total, axis=1)  # shape (n_files, n_epochs)
        # log_total = np.log(total)
        # total_mean = np.exp(np.mean(log_total, axis=1))  # shape (n_files, n_epochs)
        # sem = st.sem(np.mean(log_total, axis=1), axis=1)
        # ci_low, ci_high = st.t.interval(0.95, log_total.shape[1] - 1, loc=np.mean(log_total, axis=1), scale=sem)
        # total_lower = np.exp(ci_low)
        # total_upper = np.exp(ci_high)

        # plt.figure(figsize=(8, 5))
        # for i in range(total_mean.shape[0]):
        #     label = f"{i}" if labels is None else labels[i]
        #     plt.plot(range(total_mean.shape[1]), total_mean[i], label=label, color=colors[i % len(colors)])
        #     plt.fill_between(range(total_mean.shape[1]), total_lower[i], total_upper[i], alpha=0.2, color=colors[i % len(colors)])
        # plt.xlabel("Updates")
        # plt.ylabel("Cumulative Cost")
        # # plt.yscale("log")
        # # plt.ylim
        # plt.savefig(f"{self.plot_dir}/LOG_TOTAL.png", dpi=300)
        # plt.clf()
        # plt.close()


        cost_fields = [f for f in logs_loaded.keys() if "cost" in f]
        plt.figure(figsize=(8, 5))
        total_costs = []
        total_costs_steps = []
        total_costs_lowers = []
        total_costs_uppers = []
        for i, filename in enumerate(filenames):
            logs_loaded = self.load_logs(filename)  # dict with keys like 'epoch', 'field1', 'field2'...
            # group summed costs by epoch
            data = defaultdict(list)
            for i in range(len(logs_loaded["epoch"])):
                epoch = int(logs_loaded["epoch"][i])
                # sum across all cost fields for this step
                total_cost = sum(float(logs_loaded[f][i]) for f in cost_fields)
                data[epoch].append(total_cost)

            # compute log-mean and 95% CI per epoch
            epochs, means, lowers, uppers = [], [], [], []
            for e in sorted(data.keys()):
                arr = np.array(data[e])
                log_arr = np.log(arr)
                mean = log_arr.mean()
                sem = st.sem(log_arr)
                ci_low, ci_high = st.t.interval(0.95, len(log_arr) - 1, loc=mean, scale=sem)
                
                epochs.append(e)
                means.append(np.exp(mean))
                lowers.append(np.exp(ci_low))
                uppers.append(np.exp(ci_high))

            # plot
            total_costs.append(means)
            total_costs_steps.append(epochs)
            total_costs_lowers.append(lowers)
            total_costs_uppers.append(uppers)
            plt.plot(epochs, means, label="Total cost per epoch", linewidth=2)
            plt.fill_between(epochs, lowers, uppers, alpha=0.2)

        plt.xlabel("Step", fontsize=18)
        # #plt.ylabel("Total cost", fontsize=18)
            # ticks fontsize
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, which='both', color='white', linewidth=1.5)
        plt.tick_params(axis='both', which='both', length=0)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
            # facecolor
        plt.gca().set_facecolor((0.95, 0.95, 0.95))
        plt.tight_layout()
        #plt.legend()
        plt.yscale("log")
        #plt.ylim((5, 20))
        plt.savefig(f"{self.plot_dir}/LOG_TOTAL_COST.png", dpi=300)
        plt.clf()
        plt.close()

        fields = ["lipschitz_loss_dynamics", "loss_rollout", "loss_rollout_observer"]
        observer_losses = []
        dynamics_losses = []
        lipschitz_losses = []
        steps = []
        if aux_filenames is not None:
            #plt.figure(figsize=(8, 5))
            # plot the lipschitz constants from aux files
            for field in fields:
                plt.figure(figsize=(8, 5))
                for aux_filename in aux_filenames:
                    aux_logs = self.load_logs(aux_filename)
                    if field == "loss_rollout_observer":
                        observer_loss = aux_logs[field]
                        observer_losses.append(observer_loss)
                        steps.append(aux_logs["step"])
                    if field == "loss_rollout":
                        dynamics_loss = aux_logs[field]
                        dynamics_losses.append(dynamics_loss)
                    if field == "lipschitz_loss_dynamics":
                        lipschitz_loss = aux_logs[field]
                        lipschitz_losses.append(lipschitz_loss)

                    if field == "loss_rollout":
                        # smooth the rollout loss with a moving average of window 10
                        window_size = 10
                        cumsum_vec = np.cumsum(np.insert(aux_logs[field], 0, 0)) 
                        aux_logs[field] = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
                        aux_logs["step"] = aux_logs["step"][window_size - 1:]

                    plt.plot(aux_logs["step"], aux_logs[field], label=f"Aux: {aux_filename}")

                plt.xlabel("Step", fontsize=18)
                y_label = "Dynamics Lipschitz bound" if field == "lipschitz_loss_dynamics" else "Rollout MAE"
                plt.ylabel(y_label, fontsize=18)
                # ticks fontsize
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(True, which='both', color='white', linewidth=1.5)
                plt.tick_params(axis='both', which='both', length=0)
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                    # facecolor
                plt.gca().set_facecolor((0.95, 0.95, 0.95))
                

                plt.yscale("log")
                plt.tight_layout()
                plt.savefig(f"{self.plot_dir}/LOG_{field}.png", dpi=300)
                plt.clf()
                plt.close()
                

        # put the last two in each list to the front of the list and reverse the order of the rest before concatenating
        if len(total_costs) >= 2:
            total_costs = [total_costs[-2], total_costs[-1]] + total_costs[:-2][::-1]
            total_costs_steps = [total_costs_steps[-2], total_costs_steps[-1]] + total_costs_steps[:-2][::-1]
            total_costs_lowers = [total_costs_lowers[-2], total_costs_lowers[-1]] + total_costs_lowers[:-2][::-1]
            total_costs_uppers = [total_costs_uppers[-2], total_costs_uppers[-1]] + total_costs_uppers[:-2][::-1]
        if len(dynamics_losses) >= 2:
            dynamics_losses = [dynamics_losses[-2], dynamics_losses[-1]] + dynamics_losses[:-2][::-1]
            observer_losses = [observer_losses[-2], observer_losses[-1]] + observer_losses[:-2][::-1]
            lipschitz_losses = [lipschitz_losses[-2], lipschitz_losses[-1]] + lipschitz_losses[:-2][::-1]
            steps = [steps[-2], steps[-1]] + steps[:-2][::-1]
        print(f"len of total costs: {len(total_costs)}, len of steps: {len(total_costs_steps)}")
        print(f"len of dynamics losses: {len(dynamics_losses)}, len of observer losses: {len(observer_losses)}, len of lipschitz losses: {len(lipschitz_losses)}")
        print(f"trot_bwds: {len(trot_bwds)}, trot_fwds: {len(trot_fwds)}, trot_inplaces: {len(trot_inplaces)}")
        # combine left trotting and right trotting costs into a lateral trotting cost
        lateral_trots = []
        lateral_trots_lower = []
        lateral_trots_upper = []
        for i in range(min(len(trot_lefts_raw), len(trot_rights_raw))):
            left_data = trot_lefts_raw[i]
            right_data = trot_rights_raw[i]
            combined_data = defaultdict(list)
            for e in set(left_data.keys()).union(set(right_data.keys())):
                left_vals = left_data.get(e, [0.0])
                right_vals = right_data.get(e, [0.0])
                combined_vals = [(l + r) / 2 for l, r in zip(left_vals, right_vals)]
                combined_data[e].extend(combined_vals)
            epochs, means, lowers, uppers = [], [], [], []
            for e in sorted(combined_data.keys()):
                arr = np.array(combined_data[e])
                log_arr = np.log(arr)
                mean = log_arr.mean()
                sem = st.sem(log_arr)
                ci_low, ci_high = st.t.interval(0.95, len(log_arr) - 1, loc=mean, scale=sem)
                
                epochs.append(e)
                means.append(np.exp(mean))
                lowers.append(np.exp(ci_low))
                uppers.append(np.exp(ci_high))
            lateral_trots.append(means)
            lateral_trots_lower.append(lowers)
            lateral_trots_upper.append(uppers)
        if len(trot_bwds) >= 2:
            trot_bwds = [trot_bwds[-2], trot_bwds[-1]] + trot_bwds[:-2][::-1]
            trot_bwds_lower = [trot_bwds_lower[-2], trot_bwds_lower[-1]] + trot_bwds_lower[:-2][::-1]
            trot_bwds_upper = [trot_bwds_upper[-2], trot_bwds_upper[-1]] + trot_bwds_upper[:-2][::-1]
        if len(trot_fwds) >= 2:
            trot_fwds = [trot_fwds[-2], trot_fwds[-1]] + trot_fwds[:-2][::-1]
            trot_fwds_lower = [trot_fwds_lower[-2], trot_fwds_lower[-1]] + trot_fwds_lower[:-2][::-1]
            trot_fwds_upper = [trot_fwds_upper[-2], trot_fwds_upper[-1]] + trot_fwds_upper[:-2][::-1]
        if len(trot_inplaces) >= 2:
            trot_inplaces = [trot_inplaces[-2], trot_inplaces[-1]] + trot_inplaces[:-2][::-1]
            trot_inplaces_lower = [trot_inplaces_lower[-2], trot_inplaces_lower[-1]] + trot_inplaces_lower[:-2][::-1]
            trot_inplaces_upper = [trot_inplaces_upper[-2], trot_inplaces_upper[-1]] + trot_inplaces_upper[:-2][::-1]
        if len(lateral_trots) >= 2:
            lateral_trots = [lateral_trots[-2], lateral_trots[-1]] + lateral_trots[:-2][::-1]
            lateral_trots_lower = [lateral_trots_lower[-2], lateral_trots_lower[-1]] + lateral_trots_lower[:-2][::-1]
            lateral_trots_upper = [lateral_trots_upper[-2], lateral_trots_upper[-1]] + lateral_trots_upper[:-2][::-1]
        print(f"len of trot_bwds: {len(trot_bwds)}, len of trot_fwds: {len(trot_fwds)}, len of trot_inplaces: {len(trot_inplaces)}, len of lateral trots: {len(lateral_trots)}")

        from matplotlib.ticker import MaxNLocator

        x_ticks = ["0", "$2.5\\times10^4$", "$5\\times10^4$", "$7.5\\times10^4$", "$10^5$"]

        alphas = [1.0, 0.8, 0.6, 0.4, 0.2]
        #fig, axs = plt.subplots(4, 2, figsize=(10, 16), constrained_layout=True) 
        fig, axs = plt.subplots(2, 4, figsize=(20.45, 8), constrained_layout=True)
        axs = axs.flatten()
        axs[0].set_title("Dynamics Lipschitz bound", fontsize=18, loc='left')
        axs[0].set_yscale("log")
        axs[0].set_xlabel("Step", fontsize=18)
        for i in range(len(lipschitz_losses)):
            if i > 1:
                axs[0].plot(steps[i], lipschitz_losses[i], linewidth=2, alpha=alphas[i-2])
            else:
                axs[0].plot(steps[i], lipschitz_losses[i], linewidth=2)

        axs[1].set_title("Total cost - 5 tasks", fontsize=18, loc='left')
        axs[1].set_yscale("log")
        axs[1].set_xlabel("Step", fontsize=18)
        for i in range(len(total_costs)):
            if i > 1:
                axs[1].plot(total_costs_steps[i], total_costs[i], linewidth=2, alpha=alphas[i-2])
            else:
                axs[1].plot(total_costs_steps[i], total_costs[i], linewidth=2)
            axs[1].fill_between(total_costs_steps[i], total_costs_lowers[i], total_costs_uppers[i], alpha=0.1)

        # tot_cost_smooth = total_costs[-3]
        # tot_cost_steps_smooth = total_costs_steps[-3]
        # # only steps > 95000
        # tot_cost_smooth = [c for s, c in zip(tot_cost_steps_smooth, tot_cost_smooth) if s >= 95000]
        # tot_cost_steps_smooth = [s for s in tot_cost_steps_smooth if s >= 95000]
        # min_total_cost = min(tot_cost_smooth)
        # min_index = tot_cost_smooth.index(min_total_cost)
        # print(f"Min total cost: {min_total_cost} at step {tot_cost_steps_smooth[min_index]}")

        axs[2].set_title("Dynamics rollout MAE", fontsize=18, loc='left')
        #axs[2].set_yscale("log")
        axs[2].set_xlabel("Step", fontsize=18)
        window_size = 15
        for i in range(len(dynamics_losses)):
            cumsum_vec = np.cumsum(np.insert(dynamics_losses[i], 0, 0)) 
            smooth_dynamics_losses = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            smooth_steps = steps[i][window_size - 1:]
            if i > 1:
                axs[2].plot(smooth_steps[::5], smooth_dynamics_losses[::5], linewidth=2, alpha=alphas[i-2])
            else:
                axs[2].plot(smooth_steps[::5], smooth_dynamics_losses[::5], linewidth=2)

        axs[3].set_title("Estimator rollout MAE", fontsize=18, loc='left')
        #axs[3].set_yscale("log")
        axs[3].set_xlabel("Step", fontsize=18)
        for i in range(len(observer_losses)):
            cumsum_vec = np.cumsum(np.insert(observer_losses[i], 0, 0)) 
            smooth_observer_losses = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            smooth_steps = steps[i][window_size - 1:]
            if i > 1:
                axs[3].plot(smooth_steps[::5], smooth_observer_losses[::5], linewidth=2, alpha=alphas[i-2])
            else:
                axs[3].plot(smooth_steps[::5], smooth_observer_losses[::5], linewidth=2)


        axs[4].set_title("Task cost - forward trot", fontsize=18, loc='left')
        #axs[4].set_title("Forward trotting", fontsize=18, loc='right')
        axs[4].set_yscale("log")
        axs[4].set_xlabel("Step", fontsize=18)
        for i in range(len(trot_fwds)):
            if i > 1:
                axs[4].plot(total_costs_steps[i], trot_fwds[i], linewidth=2, alpha=alphas[i-2])
            else:
                axs[4].plot(total_costs_steps[i], trot_fwds[i], linewidth=2)
            axs[4].fill_between(total_costs_steps[i], trot_fwds_lower[i], trot_fwds_upper[i], alpha=0.1)

        axs[5].set_title("Task cost - backward trot", fontsize=18, loc='left')
        # axs[5].set_title("Backward trotting", fontsize=18, loc='right')
        axs[5].set_yscale("log")
        axs[5].set_xlabel("Step", fontsize=18)
        for i in range(len(trot_bwds)):
            if i > 1:
                axs[5].plot(total_costs_steps[i], trot_bwds[i], linewidth=2, alpha=alphas[i-2])
            else:
                axs[5].plot(total_costs_steps[i], trot_bwds[i], linewidth=2)
            axs[5].fill_between(total_costs_steps[i], trot_bwds_lower[i], trot_bwds_upper[i], alpha=0.1)

        axs[6].set_title("Task cost - in-place trot", fontsize=18, loc='left')
        # axs[6].set_title("In-place trotting", fontsize=18, loc='right')
        axs[6].set_yscale("log")
        axs[6].set_xlabel("Step", fontsize=18)
        for i in range(len(trot_inplaces)):
            if i > 1:
                axs[6].plot(total_costs_steps[i], trot_inplaces[i], linewidth=2, alpha=alphas[i-2])
            else:
                axs[6].plot(total_costs_steps[i], trot_inplaces[i], linewidth=2)
            axs[6].fill_between(total_costs_steps[i], trot_inplaces_lower[i], trot_inplaces_upper[i], alpha=0.1)

        axs[7].set_title("Task cost - lateral trot (left + right)", fontsize=18, loc='left')
        #axs[7].set_title("Lateral trotting", fontsize=18, loc='right')
        axs[7].set_yscale("log")
        axs[7].set_xlabel("Step", fontsize=18)
        for i in range(len(lateral_trots)):
            if i > 1:
                axs[7].plot(total_costs_steps[i], lateral_trots[i], linewidth=2, alpha=alphas[i-2])
            else:
                axs[7].plot(total_costs_steps[i], lateral_trots[i], linewidth=2)
            axs[7].fill_between(total_costs_steps[i], lateral_trots_lower[i], lateral_trots_upper[i], alpha=0.1)


        for ax in axs:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(axis='both', which='both', labelsize=14)
            ax.set_xticks([0, 25000, 50000, 75000, 100000])
            ax.set_xticklabels(x_ticks)

        fig.align_labels()
        fig.align_titles()
        plt.savefig(f"{self.plot_dir}/LOG_AUXILIARY.png", dpi=400)
        plt.clf()
        plt.close()
