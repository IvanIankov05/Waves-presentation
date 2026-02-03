import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def _load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None, sep=None, engine="python")

    if df.shape[1] < 3:
        raise ValueError("Expected at least 3 columns: time(s), C1(V), C2(V).")

    colmap = {}
    lower_cols = [str(c).strip().lower() for c in df.columns]
    for idx, name in enumerate(lower_cols):
        if name in {"t", "time", "time_s", "seconds", "s"}:
            colmap[df.columns[idx]] = "time_s"
        elif name in {"c1", "ch1", "channel1", "c1_v", "ch1_v"}:
            colmap[df.columns[idx]] = "c1_v"
        elif name in {"c2", "ch2", "channel2", "c2_v", "ch2_v"}:
            colmap[df.columns[idx]] = "c2_v"

    df = df.rename(columns=colmap)

    if "time_s" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time_s"})
    if "c1_v" not in df.columns:
        df = df.rename(columns={df.columns[1]: "c1_v"})
    if "c2_v" not in df.columns:
        df = df.rename(columns={df.columns[2]: "c2_v"})

    return df[["time_s", "c1_v", "c2_v"]]


def _sampling_rate(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Time column must be strictly increasing.")
    return 1.0 / np.median(dt)


def _fft(x: np.ndarray, fs: float):
    n = x.size
    win = np.hanning(n)
    xw = (x - np.mean(x)) * win
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(X) / (np.sum(win) / 2.0)
    return freqs, X, mag


def main():
    parser = argparse.ArgumentParser(
        description="FFT of C1 and C2 from a CSV file."
    )
    parser.add_argument("file", nargs="?", help="Path to CSV/TSV file.")
    parser.add_argument(
        "--max-freq",
        type=float,
        default=200000.0,
        help="Max frequency (Hz) to show in FFT plot.",
    )
    parser.add_argument(
        "--save-eq",
        help="Save equalizer H(f)=C1/C2 to an .npz file.",
    )
    parser.add_argument(
        "--apply-eq",
        help="Apply equalizer .npz to C2 and plot before/after vs C1.",
    )
    args = parser.parse_args()

    file_path = args.file
    if not file_path:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[
                ("Data files", "*.csv *.tsv *.txt"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()

    if not file_path:
        raise SystemExit("No file selected. Exiting.")

    df = _load_data(Path(file_path))
    time_s = df["time_s"].to_numpy(dtype=float)
    c1 = df["c1_v"].to_numpy(dtype=float)
    c2 = df["c2_v"].to_numpy(dtype=float)

    fs = _sampling_rate(time_s)

    if args.apply_eq:
        data = np.load(args.apply_eq)
        eq_freqs = data["freqs"]
        eq_H = data["H"]

        # Apply equalizer to C2
        f2, X2, _ = _fft(c2, fs)
        H = np.interp(f2, eq_freqs, np.real(eq_H), left=0.0, right=0.0) + 1j * np.interp(
            f2, eq_freqs, np.imag(eq_H), left=0.0, right=0.0
        )
        X_eq = X2 * H
        c_eq = np.fft.irfft(X_eq, n=c2.size)

        plt.figure(figsize=(10, 4))
        plt.plot(time_s, c1, label="C1 original")
        plt.plot(time_s, c2, label="C2 before EQ", alpha=0.7)
        plt.title("ASCII: Before Equalization")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(10, 4))
        plt.plot(time_s, c1, label="C1 original")
        plt.plot(time_s, c_eq, label="C2 after EQ", alpha=0.7)
        plt.title("ASCII: After Equalization")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    f1, X1, mag1 = _fft(c1, fs)
    f2, X2, mag2 = _fft(c2, fs)

    max_f = min(args.max_freq, f1[-1], f2[-1])
    mask1 = f1 <= max_f
    mask2 = f2 <= max_f

    plt.figure(figsize=(10, 4))
    plt.plot(f1[mask1], mag1[mask1], label="C1 FFT")
    plt.plot(f2[mask2], mag2[mask2], label="C2 FFT", alpha=0.8)
    plt.title("FFT Magnitude of C1 and C2")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()

    if args.save_eq:
        # Interpolate C2 FFT onto C1 frequency grid for stable ratio.
        X2i = np.interp(f1, f2, np.real(X2), left=0.0, right=0.0) + 1j * np.interp(
            f1, f2, np.imag(X2), left=0.0, right=0.0
        )
        eps = 1e-12
        H = X1 / (X2i + eps)
        np.savez(args.save_eq, freqs=f1, H=H)

        plt.figure(figsize=(10, 4))
        plt.plot(f1[mask1], np.abs(H)[mask1])
        plt.title("Equalizer |H(f)| = |C1/C2|")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

