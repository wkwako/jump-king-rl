import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import time
import datetime
import traceback
import statistics

import static_variables
import JumpKingRL
import RecordingParser

class Analysis:
    # columns to keep, in this order (order doesn't matter for pandas access,
    # but this keeps written CSVs readable)
    KEEP_COLUMNS = [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "time/total_timesteps",
        "time/iterations",
        "time/fps",
        "time/time_elapsed",
        "train/entropy_loss",
        "train/clip_range",
        "train/n_updates",
        "train/learning_rate",
        "train/loss",
        "train/value_loss",
        "train/approx_kl",
        "train/policy_gradient_loss",
        "train/explained_variance",
        "train/clip_fraction",
    ]

    # columns that reset to 0 at the start of each retraining session
    # (because train_model_one_screen calls model.learn(reset_num_timesteps=True))
    # and need a running offset applied so they're monotonic across sessions
    CUMULATIVE_COLUMNS = ["time/total_timesteps", "train/n_updates", "time/iterations"]

    def __init__(self, model_folder):
        model_base_direc = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/"
        self.model_folder = model_folder
        self.model_dir = f"{model_base_direc}{model_folder}/"

    def _log_dir(self, screen_num):
        return os.path.join(self.model_dir, f"screen{screen_num}", f"ppo_screen_{screen_num}_log")

    def _training_csv_path(self, screen_num):
        return os.path.join(self.model_dir, f"screen{screen_num}", "training.csv")

    def combine_csvs(self, screen_num, write=True):
        """Concatenates every session's progress.csv for one screen, in
        chronological order. Applies a running offset to CUMULATIVE_COLUMNS
        so they don't reset at session boundaries. Skips empty session files.
        Writes the result to screen{N}/training.csv.
        """
        log_dir = self._log_dir(screen_num)
        if not os.path.isdir(log_dir):
            print(f"No log directory for screen {screen_num}: {log_dir}")
            return None
 
        session_dirs = sorted(
            d for d in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, d))
        )
 
        offsets = {c: 0.0 for c in self.CUMULATIVE_COLUMNS}
        frames = []
 
        for session in session_dirs:
            csv_path = os.path.join(log_dir, session, "progress.csv")
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                continue
 
            df = pd.read_csv(csv_path)
            cols_present = [c for c in self.KEEP_COLUMNS if c in df.columns]
            df = df[cols_present].copy()
 
            for col in self.CUMULATIVE_COLUMNS:
                if col in df.columns:
                    df[col] = df[col] + offsets[col]
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        offsets[col] = non_null.iloc[-1]
 
            frames.append(df)
 
        if not frames:
            print(f"No usable progress.csv files found for screen {screen_num}")
            return None
 
        combined = pd.concat(frames, ignore_index=True)
 
        if write:
            out_path = self._training_csv_path(screen_num)
            combined.to_csv(out_path, index=False)
            print(f"Wrote {out_path} ({len(combined)} rows from {len(frames)} session(s))")
 
        return combined

    def combine_all(self, write=True):
        """Runs combine_csvs for every screenN folder found in model_dir."""
        results = {}
        for entry in sorted(os.listdir(self.model_dir)):
            if not entry.startswith("screen"):
                continue
            suffix = entry[len("screen"):]
            if not suffix.isdigit():
                continue
            screen_num = int(suffix)
            results[screen_num] = self.combine_csvs(screen_num, write=write)
        return results

    def plot_screen_metric(self, screen_num, column, save=False, show=True):
        """Plots one column from screen{N}/training.csv against
        time/total_timesteps. Set save=True to write a PNG next to
        training.csv instead of (or in addition to) displaying it.
        """
        training_path = self._training_csv_path(screen_num)
        if not os.path.exists(training_path):
            print(f"No training.csv for screen {screen_num}. Run combine_csvs first.")
            return

        df = pd.read_csv(training_path)
        if column not in df.columns:
            print(f"Column '{column}' not found. Available columns: {list(df.columns)}")
            return

        x = df["time/total_timesteps"] if "time/total_timesteps" in df.columns else df.index

        plt.figure(figsize=(10, 5))
        plt.plot(x, df[column])
        plt.xlabel("Total timesteps")
        plt.ylabel(column)
        plt.title(f"Screen {screen_num}: {column} over training")
        plt.grid(alpha=0.3)

        if save:
            safe_name = column.replace("/", "_")
            out_path = os.path.join(self.model_dir, f"screen{screen_num}", f"{safe_name}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def train_no_learning(self, folder_name, screen, num_episodes=50):
        """Runs a screen's model for a fixed number of episodes with no
        training — used to estimate success rate, mean/std time-to-success,
        mean/std actions-to-success, and a fall-vs-timeout failure breakdown.

        wins/losses/fall_losses/action_timeout_losses are read directly off
        the env (never reset in reset(), so they safely accumulate across the
        whole batch). episode_timer/action_counter must travel through the
        info dict instead, since reset() zeroes them each episode and the
        VecEnv auto-resets the env internally as soon as done=True — by the
        time control returns here, those live attributes are already gone.
        """
        model_path = f"{self.model_dir}{folder_name}/ppo_screen_{screen}"
        screen_path = f"{self.model_dir}{folder_name}"
        if not os.path.exists(model_path + ".zip"):
            print(f"No model found for screen {screen}, stopping.")
            return

        jk = JumpKingRL.JumpKingRL(self.model_folder)
        model = jk.load_model(folder_name, screen=screen, only_agent=True)
        env = model.env.envs[0].env
        env.expected_screen = screen
        env.total_screen_actions = 0

        actual_screen = env.read_gamedata()["current_screen"]
        if actual_screen != screen:
            print(f"Warning: expected screen {screen}, got {actual_screen}. Teleporting...")
            env.teleport(screen)

        # eval runs start from a clean count regardless of anything the env
        # instance may have accumulated before this call
        env.wins = 0
        env.losses = 0
        env.fall_losses = 0
        env.action_timeout_losses = 0

        model.policy.set_training_mode(False)
        deterministic = screen not in static_variables.NONDETERMINISTIC_SCREENS

        success_times = []     # completed time-to-success intervals (mean-eligible)
        success_actions = []   # completed actions-to-success counts (mean-eligible)
        pending_time = 0.0     # time accumulated across consecutive failures, carried forward
        pending_actions = 0    # actions accumulated across consecutive failures, carried forward
        mean_success_time = None
        std_success_time = None
        mean_actions_to_success = None
        std_actions_to_success = None

        try:
            for episode in range(num_episodes):
                obs = model.env.reset()
                done = [False]
                info = {}
                while not done[0]:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, infos = model.env.step(action)
                    info = infos[0]

                pending_time += info.get("episode_timer", 0.0)
                pending_actions += info.get("actions", 0)

                if info.get("success"):
                    success_times.append(pending_time)
                    success_actions.append(pending_actions)
                    pending_time = 0.0
                    pending_actions = 0

            print(f"Screen {screen} eval complete: "
                f"{env.wins} win(s), {env.losses} loss(es) "
                f"({env.fall_losses} fall, {env.action_timeout_losses} timeout) "
                f"over {num_episodes} attempt(s).")

        except KeyboardInterrupt:
            print(f"Interrupted during eval on screen {screen}.")
            print(f"Partial results — {env.wins} win(s), {env.losses} loss(es).")

        finally:
            # computed here (not inside try) so a partial/interrupted run still
            # reports stats over whatever successes it did collect
            if success_times:
                mean_success_time = sum(success_times) / len(success_times)
                if len(success_times) >= 2:
                    std_success_time = statistics.stdev(success_times)
            if success_actions:
                mean_actions_to_success = sum(success_actions) / len(success_actions)
                if len(success_actions) >= 2:
                    std_actions_to_success = statistics.stdev(success_actions)

            if mean_success_time is not None:
                print(f"Mean time-to-success: {mean_success_time:.2f}s "
                    f"(n={len(success_times)})")
            else:
                print("No successful episodes — no time-to-success data.")

            self.write_stats(
                screen, env.wins, env.losses, env.fall_losses, env.action_timeout_losses,
                mean_success_time, std_success_time,
                mean_actions_to_success, std_actions_to_success,
                num_episodes
            )
            env.reset_keys()

        return {
            "wins": env.wins,
            "losses": env.losses,
            "fall_losses": env.fall_losses,
            "action_timeout_losses": env.action_timeout_losses,
            "success_times": success_times,
            "success_actions": success_actions,
            "mean_success_time": mean_success_time,
            "std_success_time": std_success_time,
            "mean_actions_to_success": mean_actions_to_success,
            "std_actions_to_success": std_actions_to_success,
        }


    def write_stats(self, screen, wins, losses, fall_losses, action_timeout_losses,
                    mean_success_time, std_success_time,
                    mean_actions_to_success, std_actions_to_success, target_episodes):
        path = os.path.join(self.model_dir, "eval_stats.json")

        if os.path.exists(path):
            with open(path, "r") as f:
                all_stats = json.load(f)
        else:
            all_stats = {}

        actual_episodes = wins + losses
        all_stats[str(screen)] = {
            "wins": wins,
            "losses": losses,
            "fall_losses": fall_losses,
            "action_timeout_losses": action_timeout_losses,
            "num_episodes": actual_episodes,
            "target_episodes": target_episodes,
            "win_percentage": wins / actual_episodes if actual_episodes > 0 else None,
            "mean_success_time": mean_success_time,
            "std_success_time": std_success_time,
            "mean_actions_to_success": mean_actions_to_success,
            "std_actions_to_success": std_actions_to_success,
        }

        with open(path, "w") as f:
            json.dump(all_stats, f, indent=2)

    def train_range(self, start_screen, end_screen, num_episodes, skip_screens=None):
        skip_screens = skip_screens or set()
        t0 = time.time()
        screens_to_eval = range(start_screen, end_screen+1)
        for screen in screens_to_eval:
            if screen in skip_screens:
                print(f"Skipping screen {screen} (already have good data)")
                continue
            try:
                name = f"screen{screen}"
                stats = self.train_no_learning(name, screen, num_episodes)
            except Exception as e:
                err_msg = f"Screen {screen} eval failed: {traceback.format_exc()}"
                error_log_path = f"{self.model_dir}/error_log.txt"
                with open(error_log_path, "a") as f:
                    f.write(f"\n[{datetime.datetime.now()}] {err_msg}")
                print(err_msg)
                continue

        t1 = time.time()
        print(f"Screens {start_screen} through {end_screen} completed in {round((t1-t0)/60, 4)} minutes.")

    def plot_histogram(self, screen):
        parser = RecordingParser.RecordingParser()
        records = parser.load_recording()
        records = parser.clean_actions(records)   # drops malformed rows (both-directions-held, implausible durations)
        by_screen = parser.split_recording_by_screen(records)
        actions = [a for state, a in by_screen[screen]]
        left_counts, right_counts, space_counts = parser.tally_actions(actions, threshold=0)  # raw, unsnapped

        capped_space_counts = {}
        for duration, count in space_counts.items():
            key = min(duration, 0.6)
            capped_space_counts[key] = capped_space_counts.get(key, 0) + count

        durations = sorted(capped_space_counts.keys())
        counts = [capped_space_counts[d] for d in durations]

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(durations, counts, width=0.04, color="steelblue", edgecolor="black")

        curated = static_variables.SCREEN_ACTION_MAPS[screen]["jumps"]  # [0.1, 0.15, 0.45, 0.5, 0.7]
        curated[-1] = 0.6
        for c in curated:
            ax.axvline(c, color="firebrick", linestyle="--", linewidth=1.2)

        ax.set_xlabel("Jump charge duration (s)")
        ax.set_ylabel("Count in recorded playthroughs")
        ax.set_title("Screen 17: recorded jump durations vs. curated action set")
        plt.tight_layout()
        plt.savefig("screen17_jump_durations.png", dpi=150)
        plt.show()

    def plot_agent_comparison(self,
        screen_num,
        model_base="C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/",
        left_folder="models",
        right_folder="models_rl_only",
        left_label="BC+RL",
        right_label="Pure RL",
        reward_col="rollout/ep_rew_mean",
        length_col="rollout/ep_len_mean",
        x_col="time/total_timesteps",
        save_path=None,
        show=True,
    ):
        """Plots a 2x2 comparison of two agents on the same screen.
    
        Layout: columns are agents (left_folder vs right_folder), rows are
        metrics (ep_rew_mean on top, ep_len_mean on bottom):
    
                        {left_label}      {right_label}
            ep_rew_mean  [top-left]        [top-right]
            ep_len_mean  [bottom-left]     [bottom-right]
    
        Y-axes are SHARED WITHIN EACH ROW so the across-column (agent-vs-agent)
        comparison sits on a common scale. Reward is deliberately NOT shared with
        length (different quantities). Note the two agents may use different reward
        functions (pure RL has proximity/platform shaping BC+RL lacks), so the
        reward row compares convergence SHAPE, not absolute height — say so in the
        caption. The length row is measured identically for both agents and is the
        honest efficiency comparison.
    
        Expects each folder to contain screen{N}/training.csv (the combined,
        offset CSV that Analysis.combine_csvs writes) — not a raw per-session
        progress.csv, or the x-axis will reset at session boundaries.
        """
        def load(folder):
            path = os.path.join(model_base, folder, f"screen{screen_num}", "training.csv")
            if not os.path.exists(path):
                print(f"Missing: {path}")
                return None
            return pd.read_csv(path)
    
        left = load(left_folder)
        right = load(right_folder)
    
        if left is None and right is None:
            print("Neither training.csv found — nothing to plot.")
            return
    
        # sharey='row' puts both agents on the same scale per metric so the
        # left-vs-right comparison is visually honest
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False, sharey="row")
    
        # columns: (label, dataframe); rows: (metric column, y-axis label)
        cols = [(left_label, left), (right_label, right)]
        rows = [(reward_col, "Mean episode reward"),
                (length_col, "Mean episode length (actions)")]
    
        for r, (metric, ylabel) in enumerate(rows):
            for c, (label, df) in enumerate(cols):
                ax = axes[r][c]
    
                if df is None:
                    ax.text(0.5, 0.5, f"No data\n({label})", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
    
                if metric not in df.columns:
                    ax.text(0.5, 0.5, f"'{metric}'\nnot in CSV", ha="center", va="center",
                            transform=ax.transAxes, color="firebrick")
                    continue
    
                x = df[x_col] if x_col in df.columns else df.index
                # marker='o' so short/sparse runs read as discrete logged points
                # rather than implying a resolution the data doesn't have
                ax.plot(x, df[metric], marker="o", markersize=3, linewidth=1.3)
                ax.grid(alpha=0.3)
    
                # metric label only on the leftmost column
                if c == 0:
                    ax.set_ylabel(ylabel)
                # agent title only on the top row
                if r == 0:
                    ax.set_title(label, fontsize=12, fontweight="bold")
                # compact tick labels ("50k" not "50000") and cap the number of
                # ticks so wide-range axes (e.g. pure RL out to 350k) don't collide
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
                ax.xaxis.set_major_formatter(
                    mticker.FuncFormatter(
                        lambda v, _: "0" if v == 0 else f"{v/1000:g}k"
                    )
                )
    
                # x label only on the bottom row
                if r == len(rows) - 1:
                    ax.set_xlabel("Total timesteps")
    
        fig.suptitle(f"Screen {screen_num}: {left_label} vs {right_label}",
                    fontsize=14, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig

    def plot_screen_curves(self,
        screen_num,
        model_type,
        model_base="C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/",
        folder="models",
        reward_col="rollout/ep_rew_mean",
        length_col="rollout/ep_len_mean",
        x_col="time/total_timesteps",
        normalize_reward=False,
        save_path=None,
        show=True,
    ):
        """Plots one screen's training curves as two side-by-side panels:
        ep_rew_mean (left) and ep_len_mean (right), both vs. total timesteps.
    
        Reads folder/screen{N}/training.csv (the combined, offset CSV that
        Analysis.combine_csvs writes) — not a raw per-session progress.csv.
    
        normalize_reward: if True, min-max scales the reward curve to [0, 1]
            via (r - min) / (max - min). Leave False for a single screen (you
            want the true magnitude). Set True only when you're overlaying
            MULTIPLE screens and want to compare convergence SHAPE rather than
            absolute reward height — different screens plateau at different
            magnitudes, so raw overlay would compare apples to oranges. Length
            is never normalized (it's already a comparable unit: actions).
        """
        path = os.path.join(model_base, folder, f"screen{screen_num}", "training.csv")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            return None
    
        df = pd.read_csv(path)
        x = df[x_col] if x_col in df.columns else df.index
    
        fig, (ax_rew, ax_len) = plt.subplots(1, 2, figsize=(13, 5))
    
        # --- reward panel ---
        if reward_col in df.columns:
            y = df[reward_col]
            ylabel = "Mean episode reward"
            if normalize_reward:
                lo, hi = y.min(), y.max()
                if hi > lo:  # avoid divide-by-zero on a flat curve
                    y = (y - lo) / (hi - lo)
                    ylabel = "Mean episode reward (min-max normalized)"
                else:
                    print(f"  Reward curve is flat for screen {screen_num}; "
                        f"skipping normalization.")
            ax_rew.plot(x, y, marker="o", markersize=3, linewidth=1.3)
            ax_rew.set_ylabel(ylabel)
        else:
            ax_rew.text(0.5, 0.5, f"'{reward_col}'\nnot in CSV", ha="center",
                        va="center", transform=ax_rew.transAxes, color="firebrick")
        ax_rew.set_xlabel("Total timesteps")
        ax_rew.set_title("Episode reward")
        ax_rew.grid(alpha=0.3)
    
        # --- length panel ---
        if length_col in df.columns:
            ax_len.plot(x, df[length_col], marker="o", markersize=3, linewidth=1.3)
            ax_len.set_ylabel("Mean episode length (actions)")
        else:
            ax_len.text(0.5, 0.5, f"'{length_col}'\nnot in CSV", ha="center",
                        va="center", transform=ax_len.transAxes, color="firebrick")
        ax_len.set_xlabel("Total timesteps")
        ax_len.set_title("Episode length")
        ax_len.grid(alpha=0.3)
    
        fig.suptitle(f"Screen {screen_num}: {model_type}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig
    
    def plot_screens_overlaid(self,
        screen_nums,
        model_base="C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/",
        folder="models",
        reward_col="rollout/ep_rew_mean",
        length_col="rollout/ep_len_mean",
        x_col="time/total_timesteps",
        normalize_reward=True,
        save_path=None,
        show=True,
    ):
        """Overlays several screens' curves on two shared panels — the figure
        that actually lets you compare BC+RL screens TO EACH OTHER.
    
        ep_rew_mean (left) and ep_len_mean (right), one line per screen, all on
        a common axis. reward defaults to min-max normalized here because screens
        converge to different reward magnitudes and you want to compare shape;
        set normalize_reward=False to see true magnitudes.
        """
        fig, (ax_rew, ax_len) = plt.subplots(1, 2, figsize=(13, 5))
    
        for screen_num in screen_nums:
            path = os.path.join(model_base, folder, f"screen{screen_num}", "training.csv")
            if not os.path.exists(path):
                print(f"Missing: {path} — skipping screen {screen_num}")
                continue
            df = pd.read_csv(path)
            x = df[x_col] if x_col in df.columns else df.index
    
            if reward_col in df.columns:
                y = df[reward_col]
                if normalize_reward:
                    lo, hi = y.min(), y.max()
                    if hi > lo:
                        y = (y - lo) / (hi - lo)
                ax_rew.plot(x, y, marker="o", markersize=2, linewidth=1.1,
                            label=f"Screen {screen_num}")
    
            if length_col in df.columns:
                ax_len.plot(x, df[length_col], marker="o", markersize=2, linewidth=1.1,
                            label=f"Screen {screen_num}")
    
        ax_rew.set_xlabel("Total timesteps")
        ax_rew.set_ylabel("Mean episode reward"
                        + (" (min-max normalized)" if normalize_reward else ""))
        ax_rew.set_title("Episode reward")
        ax_rew.grid(alpha=0.3)
        ax_rew.legend(fontsize=9)
    
        ax_len.set_xlabel("Total timesteps")
        ax_len.set_ylabel("Mean episode length (actions)")
        ax_len.set_title("Episode length")
        ax_len.grid(alpha=0.3)
        ax_len.legend(fontsize=9)
    
        fig.suptitle(f"{folder}: screens {', '.join(map(str, screen_nums))}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig

    def _seed_csv_path(self, seed_id):
        """training.csv for one seed, using the same layout as the rest of
        the class. seed_id is whatever the folder is named, e.g. '16a'."""
        return os.path.join(self.model_dir, f"screen{seed_id}", "training.csv")
 
    def seed_agreement_stats(
        self,
        seed_ids,
        x_col="time/total_timesteps",
        reward_col="rollout/ep_rew_mean",
        smooth_window=1,
        cutoff=None,
        verbose=True,
    ):
        """Single defensible 'all N seeds agree' statistic at a common timestep
        cutoff derived FROM THE DATA, not hard-coded.
 
        Why: when seeds stop under different rules (one deployment seed trained
        far longer than the rest), per-seed min-max normalization forces every
        curve to 1.0 at its own peak and manufactures agreement. Instead we:
          1. find each seed's last recorded timestep,
          2. take the MIN of those as the cutoff (every seed provably reached it),
          3. read each seed's reward at/just before that cutoff,
          4. report mean +/- std across seeds, raw and shared-scale normalized.
 
        cutoff=None derives the cutoff automatically (recommended). Pass an int
        only if you have a principled reason and know every seed reached it.
 
        Returns a dict of results.
        """
        seeds = []
        for sid in seed_ids:
            path = self._seed_csv_path(sid)
            if not os.path.exists(path):
                print(f"Missing: {path} — skipping seed {sid}")
                continue
            df = pd.read_csv(path)
            if x_col not in df.columns or reward_col not in df.columns:
                print(f"{path} missing '{x_col}' or '{reward_col}' — skipping")
                continue
            df = df[[x_col, reward_col]].dropna().sort_values(x_col)
            x = df[x_col].to_numpy(dtype=float)
            y = df[reward_col].to_numpy(dtype=float)
            if smooth_window and smooth_window > 1:
                y = pd.Series(y).rolling(smooth_window, min_periods=1).mean().to_numpy()
            if len(x) == 0:
                print(f"{path} has no usable rows — skipping")
                continue
            seeds.append({"id": sid, "x": x, "y": y, "last_x": float(x[-1])})
 
        if len(seeds) < 2:
            print("Need at least 2 seeds with usable data.")
            return None
 
        derived_cutoff = min(s["last_x"] for s in seeds)
        if cutoff is None:
            cutoff = derived_cutoff
        elif cutoff > derived_cutoff:
            print(f"WARNING: requested cutoff {cutoff:.0f} exceeds the shortest "
                  f"seed's last timestep {derived_cutoff:.0f}; falling back.")
            cutoff = derived_cutoff
 
        def reward_at(x, y, c):
            mask = x <= c
            return float(y[mask][-1]) if mask.any() else float("nan")
 
        for s in seeds:
            s["reward_at_cutoff"] = reward_at(s["x"], s["y"], cutoff)
 
        import numpy as np
        raw = np.array([s["reward_at_cutoff"] for s in seeds], dtype=float)
 
        # shared-scale normalization: one lo/hi for ALL seeds (over data up to
        # the cutoff), so we don't manufacture agreement the way per-seed does.
        all_y_upto = np.concatenate([s["y"][s["x"] <= cutoff] for s in seeds])
        lo, hi = float(all_y_upto.min()), float(all_y_upto.max())
        norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
 
        ids = [s["id"] for s in seeds]
        result = {
            "cutoff": cutoff,
            "derived_cutoff": derived_cutoff,
            "seed_ids": ids,
            "last_x_per_seed": {s["id"]: s["last_x"] for s in seeds},
            "raw_per_seed": dict(zip(ids, raw.tolist())),
            "norm_per_seed": dict(zip(ids, norm.tolist())),
            "raw_mean": float(np.mean(raw)),
            "raw_std": float(np.std(raw, ddof=1)),
            "raw_min": float(np.min(raw)),
            "raw_max": float(np.max(raw)),
            "raw_cv": float(np.std(raw, ddof=1) / np.mean(raw)) if np.mean(raw) != 0 else 0.0,
            "norm_mean": float(np.mean(norm)),
            "norm_std": float(np.std(norm, ddof=1)),
            "shared_lo": lo,
            "shared_hi": hi,
            "n_seeds": len(seeds),
        }
 
        if verbose:
            print(f"\n=== Seed agreement @ common cutoff {cutoff:.0f} timesteps "
                  f"(n={result['n_seeds']}) ===")
            print("Per-seed final timestep (shows the stopping-rule spread):")
            for s in seeds:
                mark = "  <- cutoff here" if s["last_x"] == derived_cutoff else ""
                print(f"  {str(s['id']):>6}: last step {s['last_x']:>9.0f}{mark}")
            print("\nPer-seed reward at cutoff:")
            for sid, r, n in zip(ids, raw, norm):
                print(f"  {str(sid):>6}: raw {r:8.2f}   shared-norm {n:5.3f}")
            print(f"\nRAW   mean +/- std : {result['raw_mean']:.2f} +/- {result['raw_std']:.2f}"
                  f"   (min {result['raw_min']:.2f}, max {result['raw_max']:.2f}, "
                  f"CV {result['raw_cv']*100:.1f}%)")
            print(f"NORM  mean +/- std : {result['norm_mean']:.3f} +/- {result['norm_std']:.3f}"
                  f"   (shared scale [{lo:.2f}, {hi:.2f}])")
            print("\nSuggested sentence:")
            print(f"  All {result['n_seeds']} seeds reached the deployment threshold on "
                  f"screen 16; at a common {cutoff:.0f}-timestep cutoff their mean "
                  f"episode reward was {result['raw_mean']:.1f} +/- "
                  f"{result['raw_std']:.1f} (n={result['n_seeds']}).")
 
        return result
 
    def timesteps_to_threshold(
        self,
        seed_ids,
        threshold,
        x_col="time/total_timesteps",
        reward_col="rollout/ep_rew_mean",
        smooth_window=1,
        require_sustained=True,
        verbose=True,
    ):
        """Timesteps each seed takes to first reach `threshold` mean episode
        reward — the stopping-rule-independent statistic for the screen-16 seed
        result. Reports mean +/- std of timesteps-to-threshold across seeds.
 
        threshold: raw ep_rew_mean bar you consider "playthrough-ready" (~160).
        require_sustained: if True, the crossing only counts if the reward stays
        >= threshold for the rest of the recorded curve (guards against a lone
        noisy spike counting as convergence). If a seed crosses only transiently,
        its first-crossing step is still reported but flagged.
 
        Returns a dict with per-seed crossing timesteps and the aggregate.
        """
        import numpy as np
 
        rows = []
        for sid in seed_ids:
            path = self._seed_csv_path(sid)
            if not os.path.exists(path):
                print(f"Missing: {path} — skipping seed {sid}")
                continue
            df = pd.read_csv(path)
            if x_col not in df.columns or reward_col not in df.columns:
                print(f"{path} missing columns — skipping")
                continue
            df = df[[x_col, reward_col]].dropna().sort_values(x_col)
            x = df[x_col].to_numpy(dtype=float)
            y = df[reward_col].to_numpy(dtype=float)
            if smooth_window and smooth_window > 1:
                y = pd.Series(y).rolling(smooth_window, min_periods=1).mean().to_numpy()
 
            at_or_above = y >= threshold
            if not at_or_above.any():
                rows.append({"id": sid, "cross_step": None, "sustained": False,
                             "max_reward": float(y.max())})
                continue
 
            first_idx = int(np.argmax(at_or_above))  # first True
            # sustained = stays >= threshold from first crossing to end
            sustained = bool(at_or_above[first_idx:].all())
            if require_sustained and not sustained:
                # first index from which it stays above for the remainder
                sustained_idx = None
                for i in range(len(y)):
                    if at_or_above[i:].all():
                        sustained_idx = i
                        break
                cross_step = float(x[sustained_idx]) if sustained_idx is not None else float(x[first_idx])
                sustained = sustained_idx is not None
            else:
                cross_step = float(x[first_idx])
 
            rows.append({"id": sid, "cross_step": cross_step, "sustained": sustained,
                         "max_reward": float(y.max())})
 
        crossed = [r for r in rows if r["cross_step"] is not None]
        steps = np.array([r["cross_step"] for r in crossed], dtype=float)
 
        result = {
            "threshold": threshold,
            "per_seed": {r["id"]: r["cross_step"] for r in rows},
            "sustained": {r["id"]: r["sustained"] for r in rows},
            "n_crossed": len(crossed),
            "n_total": len(rows),
            "mean_steps": float(np.mean(steps)) if len(steps) else float("nan"),
            "std_steps": float(np.std(steps, ddof=1)) if len(steps) > 1 else 0.0,
            "min_steps": float(np.min(steps)) if len(steps) else float("nan"),
            "max_steps": float(np.max(steps)) if len(steps) else float("nan"),
        }
 
        if verbose:
            print(f"\n=== Timesteps to threshold (reward >= {threshold}) ===")
            for r in rows:
                if r["cross_step"] is None:
                    print(f"  {str(r['id']):>6}: NEVER reached "
                          f"(max {r['max_reward']:.1f})")
                else:
                    flag = "" if r["sustained"] else "  (transient — did not stay above)"
                    print(f"  {str(r['id']):>6}: {r['cross_step']:>8.0f} steps{flag}")
            print(f"\n{result['n_crossed']}/{result['n_total']} seeds reached threshold.")
            if result["n_crossed"]:
                print(f"Timesteps-to-threshold: {result['mean_steps']:.0f} +/- "
                      f"{result['std_steps']:.0f}  "
                      f"(min {result['min_steps']:.0f}, max {result['max_steps']:.0f})")
                print("\nSuggested sentence:")
                print(f"  All {result['n_crossed']} seeds on screen 16 reached the "
                      f"deployment threshold ({threshold} mean episode reward), at "
                      f"{result['mean_steps']:.0f} +/- {result['std_steps']:.0f} "
                      f"timesteps (n={result['n_crossed']}).")
 
        return result
 
    def plot_seeds_with_cutoff(
        self,
        seed_ids,
        x_col="time/total_timesteps",
        reward_col="rollout/ep_rew_mean",
        smooth_window=1,
        normalize="none",     # "none" (raw, recommended), "shared", or "per_seed"
        cutoff=None,
        threshold=None,       # draw a horizontal line at this RAW reward (only when normalize="none")
        show_cutoff_line=True,
        title=None,
        save_path=None,
        show=True,
    ):
        """Redraws the seed overlay so the figure and the reported claim tell the
        same story.
 
        For the screen-16 seed result the honest picture is RAW reward
        (normalize="none") with a horizontal threshold line: every seed traces
        the same rise and crosses the deployment bar, which is the actual claim.
        Pass threshold=160 (or your bar) to draw it.
 
        normalize:
          "none"     - raw ep_rew_mean (recommended here; pair with threshold=)
          "shared"   - one lo/hi over all seeds (shape comparison, hides magnitude)
          "per_seed" - each curve to its own 1.0 (manufactures agreement; avoid)
 
        show_cutoff_line: the common-cutoff vertical line is only meaningful if
        you're reporting a reward-at-cutoff number. For the threshold framing you
        can turn it off so the eye goes to the threshold crossing instead.
        """
        import numpy as np
 
        loaded = []
        for sid in seed_ids:
            path = self._seed_csv_path(sid)
            if not os.path.exists(path):
                print(f"Missing: {path} — skipping seed {sid}")
                continue
            df = pd.read_csv(path)[[x_col, reward_col]].dropna().sort_values(x_col)
            x = df[x_col].to_numpy(dtype=float)
            y = df[reward_col].to_numpy(dtype=float)
            if smooth_window and smooth_window > 1:
                y = pd.Series(y).rolling(smooth_window, min_periods=1).mean().to_numpy()
            loaded.append((sid, x, y))
 
        if not loaded:
            print("No seeds loaded.")
            return None
 
        derived_cutoff = min(x[-1] for _, x, _ in loaded)
        if cutoff is None:
            cutoff = derived_cutoff
 
        if normalize == "shared":
            all_y = np.concatenate([y[x <= cutoff] for _, x, y in loaded])
            lo, hi = float(all_y.min()), float(all_y.max())
 
        fig, ax = plt.subplots(figsize=(8, 5))
        for sid, x, y in loaded:
            yy = y.copy()
            if normalize == "shared" and hi > lo:
                yy = (yy - lo) / (hi - lo)
            elif normalize == "per_seed":
                ylo, yhi = y.min(), y.max()
                if yhi > ylo:
                    yy = (y - ylo) / (yhi - ylo)
            ax.plot(x, yy, marker="o", markersize=2, linewidth=1.1, label=f"Screen {sid}")
 
        if show_cutoff_line:
            ax.axvline(cutoff, color="0.4", linestyle="--", linewidth=1)
            ax.text(cutoff, ax.get_ylim()[1], f" common cutoff\n {cutoff:.0f}",
                    va="top", ha="left", fontsize=8, color="0.3")
 
        # horizontal deployment-threshold line (raw scale only)
        if threshold is not None and normalize == "none":
            ax.axhline(threshold, color="crimson", linestyle=":", linewidth=1.3)
            ax.text(ax.get_xlim()[1], threshold, f"deployment threshold ({threshold:g}) ",
                    va="bottom", ha="right", fontsize=8, color="crimson")
 
        ax.set_xlabel("Total timesteps")
        ylabel = "Mean episode reward"
        if normalize == "shared":
            ylabel += " (shared-scale normalized)"
        elif normalize == "per_seed":
            ylabel += " (per-seed normalized)"
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"Seeds: {', '.join(map(str, seed_ids))}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
 
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

#screen = 0
#name = f"screen{screen}"

model_folder = "models"
analysis = Analysis(model_folder)

#analysis.train_range(start_screen=0, end_screen=0, num_episodes=500, skip_screens={})

#analysis = Analysis("models/screen16_seeds")
#analysis.seed_agreement_stats(["16a", "16b", "16c", "16d", "16e"])
#analysis.timesteps_to_threshold(["16a","16b","16c","16d","16e"], threshold=160)
#analysis.plot_seeds_with_cutoff(["16a","16b","16c","16d","16e"],normalize="none", threshold=160, show_cutoff_line=False, title="Screen 16: five seeds", save_path="images/16_seeds_raw.png")
 
#analysis.plot_screens_overlaid([0, 1, 2, 3, 4])

analysis.plot_screen_curves(screen_num=10, model_type="BC+RL", save_path=r"C:\Users\wkwak\Documents\CodingWork\Environments\workStuffPython\JumpKingRL\images\10_curves.png")

#analysis.plot_agent_comparison(screen_num=1, save_path=r"C:\Users\wkwak\Documents\CodingWork\Environments\workStuffPython\JumpKingRL\images\1_comparison.png")

#analysis.plot_histogram(17)

#analysis.combine_all()

#analysis.plot_screen_metric(0, "rollout/ep_rew_mean")

