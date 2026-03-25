import torch
import torch.nn as nn # this gives us nn.Module(), nn.Embedding(), nn.Linear()
import torch.nn.functional as F # this gives us relu()
import matplotlib.pyplot as plt
import seaborn as sns



class myNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.w1 = torch.tensor(1.43)
        self.b1 = torch.tensor(-0.61)

        self.w2 = torch.tensor(2.63)
        self.b2 = torch.tensor(-0.27)

        self.w3 = torch.tensor(-3.89)
        self.w4 = torch.tensor(1.35)


    def forward(self, input_values):
        # Here we run values through the neural network
        top_x_axis_values = (input_values * self.w1) + self.b1
        bottom_x_axis_values = (input_values * self.w2) + self.b2

        top_y_axis_values = F.relu(top_x_axis_values)
        bottom_y_axis_values = F.relu(bottom_x_axis_values)

        output_values = (top_y_axis_values * self.w3) + (bottom_y_axis_values * self.w4)

        return output_values
    

model = myNN()

# for Doses = 0.0, 0.5, 1.0 we create a tensor with the input doses
doses = torch.tensor([0.0, 0.5, 1.0])

model(doses)

# print(torch.round(model(doses), decimals=2))

#==================== Draw graph of the output from the top activation function relative to the input values ========

# we create a sequence of numbers between 0 and 1 (using torch.insplace()) to run through the neural network
input_doses = torch.linspace(start=0, end=1, steps=11)

# we multiply the doses by weight and add bias 
top_x_axis_values = (model.w1 * input_doses) + model.b1

# we run those x-axis values through the ReLu activation function
top_y_axis_values = F.relu(top_x_axis_values)


# We plot the data
# sns.set_theme(style="whitegrid")

# we draw individual points 
# sns.scatterplot(x=input_doses,
#                 y=top_y_axis_values,
#                 color="blue",
#                 s=200)

# We connect those points with a line
# sns.lineplot(x=input_doses,
#              y=top_y_axis_values,
#              color="blue",
#              linewidth=2.5)

# plt.xlabel("Dose")
# plt.ylabel("Upper ReLu Output")
# plt.show()

#==================== Draw graph of the output from the bottom activation function relative to the input values ========

bottom_x_axis_values = (model.w2 * input_doses) + model.b2

bottom_y_axis_values = F.relu(bottom_x_axis_values)

# sns.set_theme(style="whitegrid")
# sns.scatterplot(x=input_doses,
#                 y=bottom_y_axis_values,
#                 color="orange",
#                 s=200)

# sns.lineplot(x=input_doses,
#              y=bottom_y_axis_values,
#              color="orange",
#              linewidth=2.5)

# plt.xlabel("Dose")
# plt.ylabel("Bottom ReLu Output")
# plt.show()

#================== Draw graph of the output from the top and bottom activation function relative to the input values ========

# sns.set_theme(style="whitegrid")

# Top 
# sns.scatterplot(x=input_doses,
#                 y=top_y_axis_values,
#                 color="blue",
#                 s=200)

# sns.lineplot(x=input_doses,
#              y=top_y_axis_values,
#              color="blue",
#              linewidth=2.5)

# Bottom
# sns.scatterplot(x=input_doses,
#                 y=bottom_y_axis_values,
#                 color="orange",
#                 s=200)

# sns.lineplot(x=input_doses,
#              y=bottom_y_axis_values,
#              color="orange",
#              linewidth=2.5)

# plt.xlabel("Dose")
# plt.ylabel("ReLu Outputs")
# plt.show()

#================ We add the final weights in the neural network ==================

final_top_y_values = top_y_axis_values * model.w3
final_bottom_y_values = bottom_y_axis_values * model.w4

# We plot them to the graph

# sns.set_theme(style="whitegrid")
# Top 
# sns.scatterplot(x=input_doses,
#                 y=final_top_y_values,
#                 color="blue",
#                 s=200)

# sns.lineplot(x=input_doses,
#              y=final_bottom_y_values,
#              color="blue",
#              linewidth=2.5)

# Bottom
# sns.scatterplot(x=input_doses,
#                 y=bottom_y_axis_values,
#                 color="orange",
#                 s=200)

# sns.lineplot(x=input_doses,
#              y=bottom_y_axis_values,
#              color="orange",
#              linewidth=2.5)

# plt.xlabel("Dose")
# plt.ylabel("Final Bent Shapes for Top and Bottom")
# plt.show()

#================ We create the final bent shape adding the two bent shapes ==================

combined_bend_shape = final_top_y_values + final_bottom_y_values
sns.set_theme(style="whitegrid") 
sns.scatterplot(x=input_doses,
                y=combined_bend_shape,
                color="green",
                s=200)

sns.lineplot(x=input_doses,
             y=combined_bend_shape,
             color="green",
             linewidth=2.5)

plt.xlabel("Dose")
plt.ylabel("Final Bent Shape")
plt.show()