import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/raw/CAPSTON/network_traffic.csv")


attack_counts = df['Attack'].value_counts().reset_index()
attack_counts.columns = ['Attack Type', 'Number of Samples']


total = attack_counts['Number of Samples'].sum()
attack_counts['Percentage (%)'] = (
    attack_counts['Number of Samples']  
    / total  
    * 100
).round(2)


summary = {
    'Attack Type': 'Total',
    'Number of Samples': total,
    'Percentage (%)': attack_counts['Percentage (%)'].sum().round(2)
}

attack_counts.loc[len(attack_counts)] = summary


print(attack_counts.to_string(index=False))


plot_df = attack_counts[attack_counts['Attack Type'] != 'Total']

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(plot_df['Attack Type'], plot_df['Number of Samples'])
ax.set_xlabel('Attack Type')
ax.set_ylabel('Number of Samples')
ax.set_title('Number of Samples per Attack Type')
plt.xticks(rotation=45, ha='right')


for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + total*0.005,
            f"{int(y)}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

