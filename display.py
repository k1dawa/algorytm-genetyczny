import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

def load_histories(folder: str):
    files = glob.glob(f"{folder}/*hist*.csv")
    rows = []

    pattern = re.compile(
        r"Pc(?P<pc>[0-9.]+)_Pm(?P<pm>[0-9.]+)_N(?P<n>\d+)_sel(?P<selection>\w+)_cross(?P<crossover>\w+)_run(?P<run>\d+)"
    )

    for path in files:
        match = pattern.search(path)
        if not match:
            continue

        meta = match.groupdict()
        df = pd.read_csv(path)

        # dodanie metadanych
        df["pc"] = float(meta["pc"])
        df["pm"] = float(meta["pm"])
        df["n"] = int(meta["n"])
        df["selection"] = meta["selection"]
        df["crossover"] = meta["crossover"]
        df["run"] = int(meta["run"])

        rows.append(df)

    if not rows:
        raise RuntimeError("Brak plików historii zgodnych ze wzorcem.")

    return pd.concat(rows, ignore_index=True)


# ===================================================================================
# Wczytywanie summary
# ===================================================================================
def load_summary(path: str):
    df = pd.read_csv(path)
    return df


# ===================================================================================
# Wykres zmian najlepszego rozwiązania w czasie (średnia po runach)
# ===================================================================================
def plot_convergence(df: pd.DataFrame):
    df_group = df.groupby(["generation"]).agg(
        mean_best=("best_val", "mean"),
        std_best=("best_val", "std")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(df_group.index, df_group["mean_best"], label="Średni best_val")
    plt.fill_between(
        df_group.index,
        df_group["mean_best"] - df_group["std_best"].fillna(0),
        df_group["mean_best"] + df_group["std_best"].fillna(0),
        alpha=0.2,
    )

    plt.title("Zmiany najlepszego rozwiązania w czasie (średnia po runach)")
    plt.xlabel("Generacja")
    plt.ylabel("Best value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===================================================================================
# Porównania Pc, Pm, N, selection, crossover
# ===================================================================================
def plot_param_effect(df_summary: pd.DataFrame, param: str):
    plt.figure(figsize=(10, 6))
    groups = df_summary.groupby(param)["best_value"].mean()
    groups.plot(kind="bar")

    plt.title(f"Wpływ parametru {param} na najlepszy wynik")
    plt.ylabel("Średni best_value")
    plt.tight_layout()
    plt.show()


def plot_method_comparison(df_summary: pd.DataFrame, param: str):
    plt.figure(figsize=(10, 6))
    groups = df_summary.groupby(param)["best_value"].mean()
    groups.plot(kind="bar")

    plt.title(f"Porównanie metod: {param}")
    plt.ylabel("Średni best_value")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder = "results"
    summary_path = "results/plecak_runs_summary.csv"

    print("Wczytywanie historii...")
    df_hist = load_histories(folder)

    print("Wczytywanie summary...")
    df_summary = load_summary(summary_path)

    print("Rysowanie konwergencji...")
    plot_convergence(df_hist)

    print("Porównanie Pc...")
    plot_param_effect(df_summary, "Pc")

    print("Porównanie Pm...")
    plot_param_effect(df_summary, "Pm")

    print("Porównanie N...")
    plot_param_effect(df_summary, "N")

    print("Porównanie metod selekcji...")
    plot_method_comparison(df_summary, "selection")

    print("Porównanie metod krzyżowania...")
    plot_method_comparison(df_summary, "crossover")
