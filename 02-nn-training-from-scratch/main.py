import torch
import torch.nn as nn # this gives us nn.Module(), nn.Embedding(), nn.Linear()
import torch.nn.functional as F # this gives us relu()
from torch.optim import SGD # Stochastic Gradient Decent. 
import lightning as L 
from torch.utils.data import TensorDataset, DataLoader 
import matplotlib.pyplot as plt
import seaborn as sns


#================== We create the training data ========================

# those represent the x-axis coordinates 
training_inputs = torch.tensor([0.0, 0.5, 1.0])

# those represent the y_axis coordinates
training_label = torch.tensor([0.0, 1.0, 0.0])

# We put everything inside a DataLoader
training_dataset = TensorDataset(training_inputs, training_label)
dataloader = DataLoader(training_dataset)


#================== We create a Neural Network =====================

class myNN(L.LightningModule):
    def __init__(self):
        super().__init__()

        # we initialize parameters
        self.w1 = nn.Parameter(torch.tensor(0.06))
        self.b1 = nn.Parameter(torch.tensor(0.0))

        self.w2 = nn.Parameter(torch.tensor(3.49))
        self.b2 = nn.Parameter(torch.tensor(0.0))

        self.w3 = nn.Parameter(torch.tensor(-4.11))
        self.w4 = nn.Parameter(torch.tensor(2.74))

        self.loss = nn.MSELoss(reduction="sum")

    def forward(self, input_values):
        
        top_x_axis_values = (input_values * self.w1) + self.b1
        bottom_x_axis_values = (input_values * self.w2) + self.b2

        top_y_axis_values = F.relu(top_x_axis_values)
        bottom_y_axis_values = F.relu(bottom_x_axis_values)

        output_values = (top_y_axis_values * self.w3) + (bottom_y_axis_values * self.w4)

        return output_values

    
    def configure_optimizers(self): # Optimize weights using Stochastic Gradient Descent.
        # PyTorch doesnt have Gradient Descent. We will use Stochastic Gradient Descent (SGD) 
        # using the whole dataset rather than a random subset.
        return SGD(self.parameters(), lr=0.01) # lr: learning rate
    


    def training_step(self, batch, batch_idx): # calculates the loss
        inputs, labels = batch
        outputs = self.forward(inputs) # run inputs through the Neural Network 
        loss = self.loss(outputs, labels)  # calculate the difference between Observed and Predicted values 
        return loss


model = myNN()


#======================== Testing model before Training ==================================

# starting parameter values
for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=2))


input_doses = torch.linspace(start=0, end=1, steps=11) # a set of 11 evenly spaced values used to see how the model behaves across the whole input range.
print(input_doses)

output_values = model(input_doses) # equivalent to: output_values = model.forward(input_doses)
print(output_values)


#===================== We plot the untrained Neural Network =================================================

# sns.set_theme(style="whitegrid")
# sns.scatterplot(x=input_doses,
#                 y=output_values.detach().numpy(),
#                 color="green",
#                 s=200)

# sns.lineplot(x=input_doses,
#              y=output_values.detach().numpy(),
#              color="green",
#              linewidth=2.5)

# sns.scatterplot(x=training_inputs,
#                 y=training_label,
#                 color="orange",
#                 s=200)

# plt.xlabel("Dose")
# plt.ylabel("Effectiveness")
# plt.show()

#==================== Training the Weights and Biases in the Neural Network ==========================

model = myNN()

trainer = L.Trainer(max_epochs=500, # number of times to go through training data, changing the parameters each time
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False)

trainer.fit(model, train_dataloaders=dataloader)

# new values for parameters (weight and bias)
for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=2))

output_values = model(input_doses) 
print(torch.round(output_values, decimals=2))

#===================== We plot the Trained Neural Network =================================================

sns.set_theme(style="whitegrid")
sns.scatterplot(x=input_doses,
                y=output_values.detach().numpy(),
                color="green",
                s=200)

sns.lineplot(x=input_doses,
             y=output_values.detach().numpy(),
             color="green",
             linewidth=2.5)

sns.scatterplot(x=training_inputs,
                y=training_label,
                color="orange",
                s=200)

plt.xlabel("Dose")
plt.ylabel("Effectiveness")
plt.show()