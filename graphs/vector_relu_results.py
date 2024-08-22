import matplotlib.pyplot as plt
import numpy as np

# Sample data for the two models
input_dim = [16, 32, 64, 128, 256]
model     = [5, 10, 20, 40, 62]
simulator = [5, 10, 20, 40, 62]

# Set the width of the bars
bar_width = 0.35

# Create subplots
fig, ax = plt.subplots(2, 2, figsize = (10, 8))

# Create bar plots for the two models
hcem =ax[0,0].bar(np.arange(len(input_dim)), model, bar_width, label="HCEM")
sim =ax[0,0].bar(np.arange(len(input_dim)) + bar_width, simulator, bar_width, label="RISC-$V^2$")

# Add labels above each bar
for i in range(len(input_dim)):
   ax[0,0].text(i, model[i], str(model[i]), ha="center", va="bottom")
   ax[0,0].text(i + bar_width, simulator[i], str(simulator[i]), ha="center", va="bottom")


# Customize the plot
ax[0,0].set_xlabel("Number of elements")
ax[0,0].set_ylabel("Clock cycles")
ax[0,0].title.set_text("I. Reduced tensor size")
ax[0,0].set_xticks(np.arange(len(input_dim)) + bar_width / 2)
ax[0,0].set_xticklabels(input_dim)
ax[0,0].legend()

####################################################################################################################

# Sample data for the two models
input_dim = [17, 23, 50, 99, 210]
model     = [9, 9, 19, 34, 57]
simulator = [9, 9, 19, 34, 56]

hcem =ax[0,1].bar(np.arange(len(input_dim)), model, bar_width, label="HCEM")
sim =ax[0,1].bar(np.arange(len(input_dim)) + bar_width, simulator, bar_width, label="RISC-$V^2$")

# Add labels above each bar
for i in range(len(input_dim)):
   ax[0,1].text(i, model[i], str(model[i]), ha="center", va="bottom")
   ax[0,1].text(i + bar_width, simulator[i], str(simulator[i]), ha="center", va="bottom")


# Customize the plot
ax[0,1].set_xlabel("Number of elements")
ax[0,1].set_ylabel("Clock cycles")
ax[0,1].title.set_text("II. The tensor size is a not multiple of simulator vector width")
ax[0,1].set_xticks(np.arange(len(input_dim)) + bar_width / 2)
ax[0,1].set_xticklabels(input_dim)
ax[0,1].legend()
####################################################################################################################

# Sample data for the two models
input_dim = [1024, 1600, 3200, 6400]
model     = [ 194, 296, 568, 1118]
simulator = [ 194, 296, 568, 1118]

hcem =ax[1,0].bar(np.arange(len(input_dim)), model, bar_width, label="HCEM")
sim =ax[1,0].bar(np.arange(len(input_dim)) + bar_width, simulator, bar_width, label="RISC-$V^2$")

# Add labels above each bar
for i in range(len(input_dim)):
   ax[1,0].text(i, model[i], str(model[i]), ha="center", va="bottom")
   ax[1,0].text(i + bar_width, simulator[i], str(simulator[i]), ha="center", va="bottom")


# Customize the plot
ax[1,0].set_xlabel("Number of elements")
ax[1,0].set_ylabel("Clock cycles")
ax[1,0].title.set_text("III. Medium tensor size")
ax[1,0].set_xticks(np.arange(len(input_dim)) + bar_width / 2)
ax[1,0].set_xticklabels(input_dim)
ax[1,0].legend()
####################################################################################################################

# Sample data for the two models
input_dim = [16000, 24000, 32000]
model     = [2768, 4146, 5518]
simulator = [2768, 4146, 5518]

hcem =ax[1,1].bar(np.arange(len(input_dim)), model, bar_width, label="HCEM")
sim =ax[1,1].bar(np.arange(len(input_dim)) + bar_width, simulator, bar_width, label="RISC-$V^2$")

# Add labels above each bar
for i in range(len(input_dim)):
   ax[1,1].text(i, model[i], str(model[i]), ha="center", va="bottom")
   ax[1,1].text(i + bar_width, simulator[i], str(simulator[i]), ha="center", va="bottom")


# Customize the plot
ax[1,1].set_xlabel("Number of elements")
ax[1,1].set_ylabel("Clock cycles")
ax[1,1].title.set_text("IV. Large tensor size")
ax[1,1].set_xticks(np.arange(len(input_dim)) + bar_width / 2)
ax[1,1].set_xticklabels(input_dim)
ax[1,1].legend()
####################################################################################################################
plt.suptitle("Latency comparison between the HCEM and RISC-$V^2$ simulator for ReLU activation function")

# Show the plot
plt.tight_layout()
plt.show()
