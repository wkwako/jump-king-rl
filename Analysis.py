import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import datetime
import traceback
import statistics

import static_variables
import JumpKingRL

class Analysis:
    model_dir = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/models/"

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

    def __init__(self):
        pass

    def _log_dir(self, screen_num):
        return os.path.join(self.model_dir, f"screen{screen_num}", f"ppo_screen_{screen_num}_log")

    def _training_csv_path(self, screen_num):
        return os.path.join(self.model_dir, f"screen{screen_num}", "training.csv")

    def combine_csvs(self, screen_num, write=True):
        """Concatenates every session's progress.csv for one screen, in
        chronological order (session folders are named YYYYMMDD_HHMMSS, which
        sorts correctly as plain strings). Applies a running offset to the
        columns in CUMULATIVE_COLUMNS so they don't reset to 0 at each
        session boundary. Writes the result to screen{N}/training.csv.
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
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            cols_present = [c for c in self.KEEP_COLUMNS if c in df.columns]
            df = df[cols_present].copy()

            for col in self.CUMULATIVE_COLUMNS:
                if col in df.columns:
                    df[col] = df[col] + offsets[col]
                    # update offset for the next session using the last
                    # non-null value in this (already-offset) column
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        offsets[col] = non_null.iloc[-1]

            frames.append(df)

        if not frames:
            print(f"No progress.csv files found for screen {screen_num}")
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
        plt.xlabel("Total timesteps (cumulative across sessions)")
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

        jk = JumpKingRL.JumpKingRL()
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
                    f.write(f"\n[{datetime.now()}] {err_msg}")
                print(err_msg)
                continue

        t1 = time.time()
        print(f"Screens {start_screen} through {end_screen} completed in {round((t1-t0)/60, 4)} minutes.")
        
#screen = 0
#name = f"screen{screen}"
analysis = Analysis()

analysis.train_range(start_screen=15, end_screen=20, num_episodes=500)

#analysis.combine_all()

#analysis.plot_screen_metric(0, "rollout/ep_rew_mean")

