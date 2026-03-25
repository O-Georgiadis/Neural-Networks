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

doses = torch.tensor([0.0, 0.5, 1.0])

model(doses)

input_doses = torch.linspace(start=0, end=1, steps=11)

top_x_axis_values = (model.w1 * input_doses) + model.b1
top_y_axis_values = F.relu(top_x_axis_values)

bottom_x_axis_values = (model.w2 * input_doses) + model.b2
bottom_y_axis_values = F.relu(bottom_x_axis_values)

final_top_y_values = top_y_axis_values * model.w3
final_bottom_y_values = bottom_y_axis_values * model.w4

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