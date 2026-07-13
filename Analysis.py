import os
import time
import pandas as pd
import matplotlib.pyplot as plt

class Analysis:
    model_dir = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/models/"
 
    #columns to keep, in this order (order doesn't matter for pandas access, but this keeps written CSVs readable)
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

            if os.path.getsize(csv_path) == 0:
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

analysis = Analysis()
#analysis.combine_all()

analysis.plot_screen_metric(37, "rollout/ep_rew_mean")