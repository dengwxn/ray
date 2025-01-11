import torch
import torch.nn as nn
import torch.optim as optim

from ..core.mp.resnet import resnet152

torch.manual_seed(998244353)

# Load pretrained ResNet50 model
model = resnet152(weights=True)

num_epochs = 2
# batch_size = 128
batch_size = 32

random_input = torch.randn(batch_size, 3, 224, 224)
random_targets = torch.randint(0, 1000, (batch_size,))

device = "cuda:0"
model = model.to(device)
random_input = random_input.to(device)
random_targets = random_targets.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()

    outputs = model(random_input)

    loss = criterion(outputs, random_targets)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == random_targets).sum().item()
        accuracy = 100 * correct / batch_size
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%"
        )

model.eval()
test_input = torch.randn(1, 3, 224, 224)
if torch.cuda.is_available():
    test_input = test_input.cuda()

with torch.no_grad():
    output = model(test_input)
    _, predicted_idx = torch.max(output.data, 1)
    print(f"\nTest prediction - Class index: {predicted_idx.item()}")
