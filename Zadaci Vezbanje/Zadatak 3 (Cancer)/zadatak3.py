import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed=42
torch.manual_seed(seed)
df = pd.read_csv("cancer.csv")
df = df.dropna()

#print(df['mortality'].value_counts()) -> Koristim f1

#Delim trening i test
train, test = train_test_split(df, test_size=0.2, random_state=seed)

#Dodeljujem trening i test
X_train = train.drop(['mortality'], axis=1)
y_train = train['mortality']

X_test = test.drop(['mortality'], axis=1)
y_test = test['mortality']

#Standardizacija
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Pretvaranje u tensore
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hiden_sizes, output_size):
        super().__init__()
        self.relu = nn.ReLU()

        sizes = [input_size] + hiden_sizes + [output_size]

        self.layers = nn.ModuleList([
            nn.Linear(sizes[i-1],sizes[i]) for i in range(1, len(sizes))
        ])

    def forward(self, x):
        out = x
        for i,layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out
    
input_size = X_train_tensor.shape[1] #3 age, year, nodes
hidden_sizes = [64]
num_classes = 2

model = MLPClassifier(input_size, hidden_sizes, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

num_epochs = 500
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Treniranje gotovo.")

# Evaluacija
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    f1 = metrics.f1_score(y_pred=predicted, y_true=y_test_tensor, average='micro')
    print(f'F1: {f1}')
