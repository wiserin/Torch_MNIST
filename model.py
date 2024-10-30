import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Задаем преобразования для нормализации изображений
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # нормализация к диапазону (-1, 1)
])

# Загрузка обучающего и тестового наборов
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Загрузка данных в DataLoader для мини-батчей
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Первый сверточный слой с 1 каналом на входе и 32 на выходе. Размер ядра свертки 3х3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Второй сверточный слой с 32 каналом на входе и 64 на выходе. Размер ядра свертки 3х3
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 2 полносвязных слоя NN и softmax
        self.fc = nn.Sequential(
            nn.Flatten(), # Выравниваем в плоский вектор
            nn.Linear(36864, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    # Прямой проход
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x


# Инициализация модели
model = CNNModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 5
for epoch in range(epochs):
    model.train()  # перевод в режим обучения
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # перенос данных на GPU

        optimizer.zero_grad()  # обнуление градиентов
        outputs = model(images)  # прямой проход
        loss = criterion(outputs, labels)  # вычисление ошибки
        loss.backward()  # обратный проход
        optimizer.step()  # шаг оптимизации

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


model.eval()  # перевод модели в режим оценки
correct = 0
total = 0

with torch.no_grad():  # отключение градиентов для оценки
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")