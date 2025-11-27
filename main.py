import time
import itertools
from random import Random
import pandas as pd
import numpy as np
from statistics import mean, median, stdev

BACKPACK_CSV = "problem plecakowy dane CSV tabulatory.csv"
BACKPACK_CAPACITY = 6404180

def load_backpack(csv_path):
    df = pd.read_csv(csv_path, sep="\t")
    # oczekujemy kolumn: 'Numer', 'Waga (kg)', 'Wartość (zł)'
    df = df.reset_index(drop=True)
    return df

# Przystosowanie
def fitness(individual, backpack_df, capacity):
    total_value = 0
    total_weight = 0
    for i, gene in enumerate(individual):
        if gene:
            total_value += int(backpack_df["Wartość (zł)"][i].replace(" ", ""))
            total_weight += int(backpack_df["Waga (kg)"][i].replace(" ", ""))
            if total_weight > capacity:
                return 0
    return total_value

# Generowanie populacji
def generate_population(pop_size, n_genes, rand: Random, backpack_df, capacity):
    population = []
    while len(population) != pop_size:
        random_individual = [1 if rand.random() < 0.5 else 0 for _ in range(n_genes)]
        fitness_value = fitness(random_individual, backpack_df, capacity)
        if fitness_value != 0 and fitness_value not in population:
            population.append(random_individual)
    return population

# Selekcja ruletkowa
def selection_roulette(population, backpack_df, capacity, rand: Random, num_of_chosen=1):
    fitnesses = [fitness(ind, backpack_df, capacity) for ind in population]
    total = sum(fitnesses)
    if total == 0:
        return [population[rand.randint(0, len(population)-1)] for _ in range(num_of_chosen)]
    chosen = []
    for _ in range(num_of_chosen):
        pick = rand.uniform(0, total)
        current = 0
        for ind, f in zip(population, fitnesses):
            current += f
            if current >= pick:
                chosen.append(ind)
                break
    return chosen

# Selekcja turniejowa
def selection_tournament(population, backpack_df, capacity, rand: Random, num_of_chosen=1, group_size=10):
    winners = []
    pop_size = len(population)

    for _ in range(num_of_chosen):
        group = rand.sample(population, min(group_size, pop_size))
        best = max(group, key=lambda ind: fitness(ind, backpack_df, capacity))
        winners.append(best)
    return winners

# Krzyżowanie jednopunktowe
def crossover_one_point(p1, p2, rand: Random):
    if len(p1) < 2:
        return p1[:], p2[:]
    point = rand.randint(1, len(p1)-1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

# Krzyżowanie dwupunktowe
def crossover_two_point(p1, p2, rand: Random):
    n = len(p1)
    if n < 3:
        return crossover_one_point(p1, p2, rand)
    a = rand.randint(1, n-2)
    b = rand.randint(a+1, n-1)
    c1 = p1[:a] + p2[a:b] + p1[b:]
    c2 = p2[:a] + p1[a:b] + p2[b:]
    return c1, c2

# Mutacja odwrócenia bitu
def mutate_bitflip(individual, pm, rand: Random):
    return [ (1 - g) if rand.random() < pm else g for g in individual ]

# Ocena populacji
def evaluate_population(population, backpack_df, capacity):
    fits = [fitness(ind, backpack_df, capacity) for ind in population]
    best_idx = int(np.argmax(fits))
    worst_idx = int(np.argmin(fits))
    return {
        "fitnesses": fits,
        "best_ind": population[best_idx],
        "best_val": fits[best_idx],
        "worst_ind": population[worst_idx],
        "worst_val": fits[worst_idx],
        "mean": float(np.mean(fits))
    }

# Przebieg ewolucji dla pojedynczego uruchomienia
# na eksperyment składa się 5 uruchomień z tymi samymi parametrami
def evolve(pop_size, n_genes, backpack_df, capacity, rand,
           T, Pc, Pm,
           selection_fn, cross_fn, mutation_fn):
    population = generate_population(pop_size, n_genes, rand, backpack_df, capacity)
    run_stats = []

    for gen in range(T):
        stats = evaluate_population(population, backpack_df, capacity)
        run_stats.append({
            "generation": gen,
            "best_val": stats["best_val"],
            "worst_val": stats["worst_val"],
            "mean": stats["mean"]
        })

        new_pop = []
        # dobieranie aż do wielkości populacji
        while len(new_pop) < pop_size:
            parents = selection_fn(population, backpack_df, capacity, rand, num_of_chosen=2)
            p1, p2 = parents[0][:], parents[1][:]
            # krzyżowanie z prawdopodobieństwem Pc
            if rand.random() < Pc:
                c1, c2 = cross_fn(p1, p2, rand)
            else:
                c1, c2 = p1, p2
            # mutacja
            c1 = mutation_fn(c1, Pm, rand)
            c2 = mutation_fn(c2, Pm, rand)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    # ostatnia ocena
    final_stats = evaluate_population(population, backpack_df, capacity)
    return {
        "final_best_ind": final_stats["best_ind"],
        "final_best_val": final_stats["best_val"],
        "final_worst_val": final_stats["worst_val"],
        "history": run_stats
    }

# Uruchomienie eksperymentów
def run_experiments(backpack_csv, backpack_capacity, out_prefix,
                    Pc_list, Pm_list, N_list, T,
                    selection_methods, crossover_methods,
                    mutation_method,
                    repeats = 5, seed_base = 12345):
    df = load_backpack(backpack_csv)
    n_genes = len(df)

    # mapping funkcji
    selection_map = {
        "roulette": selection_roulette,
        "tournament": selection_tournament
    }
    crossover_map = {
        "one_point": crossover_one_point,
        "two_point": crossover_two_point
    }
    mutation_map = {
        "bitflip": mutate_bitflip
    }

    # CSV nagłówki
    summary_rows = []
    run_id = 0

    # plan eksperymentów
    # wszystkie kombinacje parametrów
    combos = list(itertools.product(Pc_list, Pm_list, N_list,
                                    selection_methods, crossover_methods))

    for Pc, Pm, N, sel_name, cross_name in combos:
        sel_fn = selection_map[sel_name]
        cross_fn = crossover_map[cross_name]
        mut_fn = mutation_map[mutation_method]

        best_vals = []
        best_inds = []
        times = []
        # powtórzenia
        for r in range(repeats):
            run_id += 1
            rand = Random(seed_base + run_id)
            start = time.time()
            res = evolve(pop_size=N, n_genes=n_genes,
                         backpack_df=df, capacity=BACKPACK_CAPACITY,
                         rand=rand, T=T, Pc=Pc, Pm=Pm,
                         selection_fn=sel_fn, cross_fn=cross_fn, mutation_fn=mut_fn)
            elapsed = time.time() - start
            times.append(elapsed)
            best_vals.append(res["final_best_val"])

            # zapisz historię runu do CSV
            hist_df = pd.DataFrame(res["history"])
            hist_csv = f"{out_prefix}_hist_Pc{Pc}_Pm{Pm}_N{N}_sel{sel_name}_cross{cross_name}_run{r+1}.csv"
            hist_df.to_csv(hist_csv, index=False)

            best_inds.append(res["final_best_ind"])

            # dodatkowy zapis - najlepsze rozwiązanie i jego przedmioty
            summary_rows.append({
                "run_id": run_id,
                "Pc": Pc,
                "Pm": Pm,
                "N": N,
                "T": T,
                "selection": sel_name,
                "crossover": cross_name,
                "run": r+1,
                "best_value": res["final_best_val"],
                "worst_value": res["final_worst_val"],
                "time": elapsed
            })

        # statystyki dla kombinacji parametrów
        summary_stat = {
            "Pc": Pc, "Pm": Pm, "N": N, "T": T,
            "selection": sel_name, "crossover": cross_name,
            "runs": repeats,
            "best_mean": float(mean(best_vals)),
            "best_median": float(median(best_vals)),
            "best_min": float(min(best_vals)),
            "best_max": float(max(best_vals)),
            "time_mean_s": float(mean(times)),
            "best_solution": best_inds[int(np.argmax(best_vals))] 
        }
        print("Dane dla uruchomienia:", summary_stat)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{out_prefix}_runs_summary.csv", index=False)
    print("Wszystkie eksperymenty zakończone. Podsumowanie zapisane.")


if __name__ == "__main__":
    # Pc_list = [0.6, 0.8, 1.0]
    # Pm_list = [0.01, 0.05, 0.1]
    # N_list = [50, 100, 200]
    Pc_list = [0.6]
    Pm_list = [0.01, 0.05]
    N_list = [50]
    T = 100

    selection_methods = ["tournament"]
    crossover_methods = ["one_point", "two_point"]
    mutation_method = "bitflip"

    run_experiments(
        backpack_csv=BACKPACK_CSV,
        backpack_capacity=BACKPACK_CAPACITY,
        out_prefix="results/plecak",
        Pc_list=Pc_list, Pm_list=Pm_list, N_list=N_list, T=T,
        selection_methods=selection_methods,
        crossover_methods=crossover_methods,
        mutation_method=mutation_method,
        repeats=5,
        seed_base=123456
    )
