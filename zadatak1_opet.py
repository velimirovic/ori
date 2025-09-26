import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score # Import koji ćemo koristiti za izračunavanje krajnje metrike
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

seed = 42
torch.manual_seed(seed)
df = pd.read_csv('./data/train.csv')

df = df.dropna()

# 1 => Enkodiranje
le_z = LabelEncoder()
le_o = LabelEncoder()
le_p = LabelEncoder()

df['zvanje'] = le_z.fit_transform(df['zvanje'])
df['pol'] = le_p.fit_transform(df['pol'])
df['oblast'] = le_o.fit_transform(df['oblast'])

# One-Hot za oblast
#df = pd.get_dummies(df, columns=['oblast'], drop_first=True)  # drop_first=True štedi jednu kolonu


# 2 => Podela podataka
features = ['oblast','godina_doktor','godina_iskustva','pol','plata']
X = df[features]
y = df['zvanje']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# 3 => Standardizacija
numeric = ['godina_doktor', 'godina_iskustva', 'plata']

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric] = scaler.fit_transform(X_train[numeric])
X_test_scaled[numeric] = scaler.transform(X_test[numeric])

# 4 => Tenzori
X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 5 => Neuronska mreza
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.relu = nn.ReLU()

        sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = nn.ModuleList([
            nn.Linear(sizes[i-1], sizes[i]) for i in range(1, len(sizes))
        ])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out

# 6 => Model
input_size = X_train_tensor.shape[1]
hidden_sizes = [64]
output_size = 3

model = MLPClassifier(input_size, hidden_sizes, output_size)

# 7 => Loss i optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 8 => Treniranje
epochs = 100
print("Treniranje")
for epoch in range(epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Treniranje gotovo.")

# 9 => Evaluacija
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    f1 = f1_score(y_test_tensor, predicted, average='micro')
    print(f'F1: {f1}')