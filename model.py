import torch.nn as nn


class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, 5, 1, 0),
        )
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # print('original:', x.shape)
        x = self.conv1(x)
        # print('conv1:', x.shape)
        x = self.conv2(x)
        # print('conv2:', x.shape)
        x = self.conv3(x)
        # print('conv3:', x.shape)
        x = x.view(x.size(0), -1)
        # print('view:', x.shape)
        x = self.fc1(x)
        # print('fc1:', x.shape)
        net_output = self.fc2(x)
        # print('fc2:', net_output.shape)
        return net_output
