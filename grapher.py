import matplotlib.pyplot as plt

# Data for the first line (EIT)
x1 = [196.1911607, 741.629354, 1555.546303, 1698.94634, 2339.822053, 2698.349361, 2982.682683, 3384.356804, 3833.611396]
y1 = [10, 73.06, 74.88, 59.96, 10, 10, 68.44, 10, 57.25]

# Data for the second line (EI)
x2 = [585.1824458, 1087.475639, 1673.962732, 1753.877393, 3435.567704, 5196.645483]
y2 = [72.22, 71.53, 73.58, 10, 74.03, 73.46]

# Plotting the lines
plt.plot(x1, y1, label='EIT', color='red')
plt.plot(x2, y2, label='EI', color='blue')

# Finding and marking maximum points with 'X' for EIT
max_x1 = x1[y1.index(max(y1))]
plt.scatter(max_x1, max(y1), color='red', marker='x', label='Max EIT')

# Finding and marking maximum points with 'X' for EI
max_x2 = x2[y2.index(max(y2))]
plt.scatter(max_x2, max(y2), color='blue', marker='x', label='Max EI')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Accuracy (%)')
plt.title('Comparison between EIT and EI on Optimizing CNN Image Recognition Model')
plt.legend()

# Displaying the graph
plt.grid(True)
plt.show()