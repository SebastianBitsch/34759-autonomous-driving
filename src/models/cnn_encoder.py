
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    
    """

    def __init__(self, in_features: int = 1, n_classes: int = 25) -> None:
        super(CNNEncoder, self).__init__()

        self.n_classes = n_classes

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, 128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten() # out: [batch, 32]
        )
        self.dense_block = nn.Sequential(
            nn.Linear(in_features=32, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes)
        )

    def forward(self, x) -> torch.Tensor:
        """ Forward pass """
        if (len(x.shape) == 3):
            x = x.unsqueeze(1) # in: [batch_size, 28, 28] -> out: [batch_size, 1, 28, 28]
        x = self.conv_block(x)
        x = self.dense_block(x)
        return x
    
    