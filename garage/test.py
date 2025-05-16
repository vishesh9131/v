import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, densenet121
from vit_pytorch import ViT

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Vanilla FFNN
class VanillaFFNN(nn.Module):
    def __init__(self):
        super(VanillaFFNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

vanilla_model = VanillaFFNN().to(device)

# Gaussian Layer
class GaussianLayer(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-5)

# FFNN with Gaussian
class FFNNWithGaussian(nn.Module):
    def __init__(self):
        super(FFNNWithGaussian, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.gaussian1 = GaussianLayer()
        self.fc2 = nn.Linear(128, 64)
        self.gaussian2 = GaussianLayer()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.gaussian1(x)
        x = torch.relu(self.fc2(x))
        x = self.gaussian2(x)
        x = self.fc3(x)
        return x

gaussian_model = FFNNWithGaussian().to(device)

# Reset optimizer to default settings
optimizer = optim.Adam(gaussian_model.parameters(), lr=0.001)

# Reset epochs to default
epochs = 5

# CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = SimpleCNN().to(device)

# RNN
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(28, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        h0 = torch.zeros(1, x.size(0), 128).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

rnn_model = SimpleRNN().to(device)

# LSTM
class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(28, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        h0 = torch.zeros(1, x.size(0), 128).to(device)
        c0 = torch.zeros(1, x.size(0), 128).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

lstm_model = SimpleLSTM().to(device)

# GRU
class SimpleGRU(nn.Module):
    def __init__(self):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(28, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        h0 = torch.zeros(1, x.size(0), 128).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

gru_model = SimpleGRU().to(device)

# ResNet
resnet_model = resnet18(pretrained=False)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)
resnet_model = resnet_model.to(device)

# DenseNet
densenet_model = densenet121(pretrained=False)
densenet_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, 10)
densenet_model = densenet_model.to(device)

# Transformer
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(28, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=2
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x

transformer_model = SimpleTransformer().to(device)

# Vision Transformer
vit_model = ViT(
    image_size=28,
    patch_size=7,
    num_classes=10,
    dim=128,
    depth=6,
    heads=8,
    mlp_dim=256,
    channels=1
).to(device)

def train_and_evaluate(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Train and evaluate each model
models = {
    "Vanilla FFNN": vanilla_model,
    "FFNN with Gaussian": gaussian_model,
    "CNN": cnn_model,
    "RNN": rnn_model,
    "LSTM": lstm_model,
    "GRU": gru_model,
    # "ResNet": resnet_model,
    # "DenseNet": densenet_model,
    "Transformer": transformer_model,
    "ViT": vit_model
}

for name, model in models.items():
    accuracy = train_and_evaluate(model, train_loader, test_loader)
    print(f"{name} Accuracy: {accuracy:.4f}")
