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

def load_summary(path: str):
    df = pd.read_csv(path)
    return df

def plot_result_change_with_generations(df: pd.DataFrame):
    df_group = df.groupby("generation").agg(
        mean_best=("best_val", "mean")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(df_group.index, df_group["mean_best"], label="Średnia najlepsza wartość")

    plt.title("Zmiany najlepszego rozwiązania w czasie")
    plt.xlabel("Generacja")
    plt.ylabel("Najlepsza wartość")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_param_effect(df_summary: pd.DataFrame, param: str):
    import matplotlib.pyplot as plt
    import numpy as np

    # agregacja: średnia i maksimum dla best_value
    agg = df_summary.groupby(param)["best_value"].agg(["mean", "max"])
    
    # przygotowanie danych do wykresu
    labels = agg.index.astype(str)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale("log")
    ax.bar(x - width/2, agg["mean"], width, label="Średnia najlepszego rozwiązania")
    ax.bar(x + width/2, agg["max"], width, label="Najlepsze rozwiązanie")

    ax.set_title(f"Wpływ parametru {param} na wyniki algorytmu")
    ax.set_ylabel("Wartość najlepszego rozwiązania")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(prop={'size': 6})
    plt.tight_layout()
    plt.show()

def plot_time_by_method(df_summary: pd.DataFrame):
    # etykieta: kombinacja selekcji i krzyżowania
    df_summary["method"] = df_summary["selection"] + " " + df_summary["crossover"]

    # uśrednienie po wszystkich runach i parametrach
    agg = df_summary.groupby("method")["time"].mean().sort_values()

    labels = agg.index
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.bar(x, agg.values)

    plt.title("Średni czas wykonania dla metod selekcji i krzyżowania")
    plt.ylabel("Czas [s]")
    plt.xticks(x, labels)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder = "results"
    summary_path = "results/plecak_runs_summary.csv"

    print("Wczytywanie historii...")
    df_hist = load_histories(folder)

    print("Wczytywanie summary...")
    df_summary = load_summary(summary_path)

    print("Rysowanie zmian wartości w czasie...")
    plot_result_change_with_generations(df_hist)

    print("Porównanie Pc...")
    plot_param_effect(df_summary, "Pc")

    print("Porównanie Pm...")
    plot_param_effect(df_summary, "Pm")

    print("Porównanie N...")
    plot_param_effect(df_summary, "N")

    print("Porównanie metod selekcji...")
    plot_param_effect(df_summary, "selection")

    print("Porównanie metod krzyżowania...")
    plot_param_effect(df_summary, "crossover")

    print("Rysowanie czasu wykonania metod...")
    plot_time_by_method(df_summary)