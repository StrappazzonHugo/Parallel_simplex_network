# libraries
import matplotlib.pyplot as plt
import pandas as pd

# Data
plt.rcParams.update({'font.size': 27})
fig = plt.figure(figsize=(17, 10))
r = [0, 1, 2, 3, 4, 5, 6, 7, 8]
raw_data = {'pivot rule': [0.65, 0.11, 0.24, 0.77, 0.63, 0.66, 0.47, 0.53, 0.31],
            'compute flow': [0.24, 0.14, 0.04, 0.03, 0.09, 0.07, 0.08, 0.27, 0.1],
            'update structures': [0.11, 0.75, 0.70, 0.05, 0.25, 0.13, 0.43, 0.18, 0.58],
            'other': [0.00, 0.00, 0.02, 0.15, 0.04, 0.14, 0.02, 0.02, 0.01]}
df = pd.DataFrame(raw_data)

# From raw value to percentage
totals = [i+j+k+l for i, j,
          k, l in zip(df['pivot rule'], df['compute flow'], df['update structures'], df['other'])]
pivot_rule = [i / j * 100 for i, j in zip(df['pivot rule'], totals)]
comput_flow = [i / j * 100 for i, j in zip(df['compute flow'], totals)]
update_struc = [i / j * 100 for i, j in zip(df['update structures'], totals)]
other = [i / j * 100 for i, j in zip(df['other'], totals)]

# plot
barWidth = 0.7
names = ('grid_long', 'grid_square', 'netgen_8', 'netgen_deg', 'netgen_lo_8', 'netgen_lo_sr',
         'netgen_sr', 'road graph',  'vision graph')
# Create green Bars
plt.bar(r, pivot_rule, color='#9d1f1f', edgecolor='white', width=barWidth, label="pivot rule")
# Create orange Bars
plt.bar(r, comput_flow, bottom=pivot_rule, color='#f6c681',
        edgecolor='white', width=barWidth, label="compute flow")
# Create blue Bars
plt.bar(r, update_struc, bottom=[i+j for i, j in zip(pivot_rule, comput_flow)],
        color='#4a7697', edgecolor='white', width=barWidth, label="update structures")

plt.bar(r, other, bottom=[i+j+k for i, j, k in zip(pivot_rule, comput_flow, update_struc)],
        color='#184a70', edgecolor='white', width=barWidth, label="other")

# Custom x axis
plt.xticks(r, names, fontsize = 16)
plt.xlabel("instances", fontsize = 27)
plt.ylabel("%", fontsize = 27)
plt.legend(loc='best', ncol=1, prop = { "size": 24 })
plt.savefig("fig.pdf", format="pdf")

# Show graphic
