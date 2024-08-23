import matplotlib.pyplot as plt


y1 = [0.76,	0.86, 0.87]
y2 = [1.2, 1.23, 1.09]
y3 = [0.99, 1.04, 0.86]
y4 = [1.01, 1.13, 1.16]
y5 = [0.94, 0.81, 0.57]
y6 = [0.87, 0.89, 0.79]
y7 = [1.1, 1.29, 1.18]
y75 = [0.65, 0.55, 0.41]
y8 = [0.79, 0.69, 0.5]
y9 = [0.77, 0.65, 0.47]
y10 = [0.91, 0.77, 0.53]
y11 = [0.95, 0.86, 0.63]

y = [1, 1, 1]
x = [2, 4, 8]

plt.rcParams.update({'font.size': 8})


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(7, 8))
axes[0][0].plot(x, y1)
axes[0][0].plot(x, y)
axes[0][0].grid()
axes[0][0].title.set_text('grid_long_20')

axes[0][1].plot(x, y2)
axes[0][1].plot(x, y)
axes[0][1].grid()
axes[0][1].title.set_text('grid_square_20')

axes[0][2].plot(x, y3)
axes[0][2].plot(x, y)
axes[0][2].grid()
axes[0][2].title.set_text('netgen_8_20')

axes[1][0].plot(x, y4)
axes[1][0].plot(x, y)
axes[1][0].grid()
axes[1][0].title.set_text('netgen_deg_12')

axes[1][1].plot(x, y5)
axes[1][1].plot(x, y)
axes[1][1].grid()
axes[1][1].title.set_text('netgen_lo_8_20')

axes[1][2].plot(x, y6)
axes[1][2].plot(x, y)
axes[1][2].grid()
axes[1][2].title.set_text('netgen_lo_sr_16')

axes[2][0].plot(x, y7)
axes[2][0].plot(x, y)
axes[2][0].grid()
axes[2][0].title.set_text('netgen_sr_16')

axes[2][1].plot(x, y75)
axes[2][1].plot(x, y)
axes[2][1].grid()
axes[2][1].title.set_text('road_path_06_FL')

axes[2][2].plot(x, y8)
axes[2][2].plot(x, y)
axes[2][2].grid()
axes[2][2].title.set_text('road_flow_07_TX')

axes[3][0].plot(x, y9)
axes[3][0].plot(x, y)
axes[3][0].grid()
axes[3][0].title.set_text('road_path_07_TX')

axes[3][1].plot(x, y10)
axes[3][1].plot(x, y)
axes[3][1].grid()
axes[3][1].title.set_text('vision_inv_05')

axes[3][2].plot(x, y11)
axes[3][2].plot(x, y)
axes[3][2].grid()
axes[3][2].title.set_text('vision_prop_05')
fig.tight_layout()
plt.savefig("speedups.pdf", format="pdf")
