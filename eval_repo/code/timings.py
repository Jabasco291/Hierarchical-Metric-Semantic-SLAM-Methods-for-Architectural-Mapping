import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_stats(df, name):
    row = df[df["name"] == name]
    mean = row["mean[s]"].values[0]
    max = row["max[s]"].values[0]
    return mean, max

#one list per method
hydra_data = []
clio_data = []

#reads main timings data
for i in range(1, 6):
    hydra = pd.read_csv(f"../graphs/hydra/eval_runs_20260327_1838/run_{i}/timing_stats.csv")
    clio = pd.read_csv(f"../graphs/clio/eval_runs_20260328_1754/run_{i}/timing_stats.csv")
    
    h_f_mean, h_f_max = get_stats(hydra, "frontend/spin")
    h_b_mean, h_b_max = get_stats(hydra, "backend/spin")
    
    c_f_mean, c_f_max = get_stats(clio, "frontend/spin")
    c_b_mean, c_b_max = get_stats(clio, "backend/spin")
    
    hydra_data.append({"f_mean": h_f_mean, "f_max": h_f_max, "b_mean": h_b_mean, "b_max": h_b_max})
    clio_data.append({"f_mean": c_f_mean, "f_max": c_f_max, "b_mean": c_b_mean, "b_max": c_b_max})

#convert to dataframes
h_results = pd.DataFrame(hydra_data)
c_results = pd.DataFrame(clio_data)

#create total columns
h_results["total_mean"] = h_results["f_mean"] + h_results["b_mean"]
h_results["total_max"] = h_results["f_max"] + h_results["b_max"]

c_results["total_mean"] = c_results["f_mean"] + c_results["b_mean"]
c_results["total_max"] = c_results["f_max"] + c_results["b_max"]

h_results.index = h_results.index + 1
h_results.insert(0, 'Run', h_results.index)

c_results.index = c_results.index + 1
c_results.insert(0, 'Run', c_results.index)

c_results.index = c_results.index + 1
c_results.index.name = "Run"


csv_path = "../graphs/s-graphs/sgraphs_eval_20260426_1657/all_runs_timings.csv"

s_graphs_raw = pd.read_csv(csv_path)
s_graphs_raw.columns = [c.replace('[s]', '').strip() for c in s_graphs_raw.columns]
s_graphs_raw = s_graphs_raw.rename(columns={'name': 'Run'})
s_graphs_raw['Run'] = s_graphs_raw['Run'].str.split('_').str[-1]

# Final results dataframe
summary_list = []

# Tuple structure: (System, Stage, Mean Column, Max Column)
configs = [
    ("Hydra", "Frontend", h_results["f_mean"], h_results["f_max"]),
    ("Hydra", "Backend",  h_results["b_mean"], h_results["b_max"]),
    ("Hydra", "Total",    h_results["total_mean"], h_results["total_max"]),
    ("Clio",  "Frontend", c_results["f_mean"], c_results["f_max"]),
    ("Clio",  "Backend",  c_results["b_mean"], c_results["b_max"]),
    ("Clio",  "Total",    c_results["total_mean"], c_results["total_max"]),
    ("S-Graphs", "Backend", s_graphs_raw["mean"], s_graphs_raw["max"]), # Integrated here
]

for system, stage, mean_col, max_col in configs:
    m = mean_col.mean()
    s = mean_col.std()
    summary_list.append({
        "System": system,
        "Stage": stage,
        "Mean": m,
        "Std Dev": s,
        "Max Mean": max_col.mean(),
        "Max Std Dev": max_col.std(),
        "CV": (s / m) if m != 0 else 0
    })

summary_df = pd.DataFrame(summary_list)


# Printing section
print("--- S-Graphs Raw Data ---")
print(s_graphs_raw.round(3).to_string(index=False))
print("--- Raw Hydra Data ---")
print(h_results.round(3).to_string(index=False))
print("\n--- Raw Clio Data ---")
print(c_results.round(3).to_string(index=False))
print("\n--- Final Summary Statistics ---")
print(summary_df.round(3).to_string(index=False))

#create plot figure
plt.figure(figsize=(12, 5))

#plot average time for front,back, total over 5 runs
#get values
h_averages = h_results[["f_mean", "b_mean", "total_mean"]].mean()
c_averages = c_results[["f_mean", "b_mean", "total_mean"]].mean()
#store in dataframe
comparison_df = pd.DataFrame({
    "Hydra": h_averages.values, 
    "Clio": c_averages.values
}, index=["Frontend", "Backend", "Total"])
#plot
ax1 = plt.subplot(1, 3, 1)
comparison_df.plot(kind="bar", ax=ax1)
plt.title("Hydra vs Clio: Average Time for Processing Stages")
plt.ylabel("Mean Time (s)")
plt.xlabel("Processing Stage")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Combine your dataframes for plotting
combined_df = pd.DataFrame({
    'Hydra': h_results['total_mean'],
    'Clio': c_results['total_mean']
})

ax2 = plt.subplot(1, 3, 2)
sns.boxplot(data=combined_df, ax=ax2)
plt.ylabel('Mean Time (s)')
plt.title('Total Pipeline Time Distribution')

plot_df = comparison_df.loc[["Frontend", "Backend"]].T

ax3 = plt.subplot(1, 3, 3)
plot_df.plot(kind="bar", ax=ax3, stacked=True, edgecolor='black')
plt.title("Mean Pipeline Time Composition", fontsize=14, pad=15)
plt.ylabel("Mean Execution Time (s)", fontsize=12)
plt.xlabel("System", fontsize=12)
plt.xticks(rotation=0)  # Keep Hydra/Clio labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(title="Pipeline Stage")

plt.tight_layout()
# plt.savefig("Hydra-vs-Clio-Timings.png")
# plt.show()