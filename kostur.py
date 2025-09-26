import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score

seed=42
torch.manual_seed(seed)
df = pd.read_csv('ime.csv')
df = df.dropna()
print(df['target'].value_counts())

# -(1)- Enkodiranje
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])

df = pd.get_dummies(df, columns=['oblast'], drop_first=True)

# -(2)- Podela na train i test
features = []
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# -(3)- Standardizacija (brojevi) i vektorizacija (tekst)
numeric = []
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric] = scaler.fit_transform(X_train[numeric])
X_test_scaled[numeric] = scaler.transform(X_test[numeric])


cv = CountVectorizer(stop_words='english')
X_train_bow = cv.fit_transform(X_train)
X_test_bow = cv.transform(X_test)

# -(4)- Tenzori
X_train_tensor = torch.tensor(X_train_scaled.toarray(), dtype=torch.float32) #Text
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32) #Numericki
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# -(5)- Neuronska mreza
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
    
# -(6)- Model
input_size = X_train_tensor.shape[1]
hidden_sizes = [64]
output_size = len(df['target'].unique())
model = MLPClassifier(input_size, hidden_sizes, output_size)

# -(7)- Loss & Optimizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# -(8)- Treniranje
print("Treniranje:")
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
print("Treniranje gotovo.")

# -(9)- Evaluacija
with torch.no_grad():
    model.eval()

    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    f1 = f1_score(y_test_tensor, predicted, average='micro') #.cpu()
    accur = accuracy_score(y_test_tensor, predicted)
    print(f'F1: {f1}')
    print(f'Accuracy: {accur}')