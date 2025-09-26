import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import torch.optim as optim

# za reproducibilnost
seed=42
torch.manual_seed(seed)
df = pd.read_csv('./data/south_park_train.csv')

df = df.dropna()

# 1 Enkodiranje
le = LabelEncoder()
df['Character'] = le.fit_transform(df['Character'])

# 2 Podela podataka
X = df['Line']
y = df['Character']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 3 BoW
cv = CountVectorizer(stop_words='english')
X_train_bow = cv.fit_transform(X_train)
X_test_bow = cv.transform(X_test)

# 4 Tensori
X_train_tensor = torch.tensor(X_train_bow.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_bow.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 5 Neuronska mreza

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
    
# 6 Model
input_size = X_train_tensor.shape[1]
hidden_sizes = [64]
output_size = df['Character'].nunique()

model = MLPClassifier(input_size, hidden_sizes, output_size)

# 7 Loss i optimizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8 Treniranje

num_epochs = 300
print("Treniranje..")
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 30 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Treniranje gotovo.")

# 9 Evaluacija
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _,predicted = torch.max(outputs, 1)
    acc = accuracy_score(y_test_tensor, predicted)
    f1 = f1_score(y_test_tensor, predicted, average='micro')
    print(f'F1: {f1}')
    print(f'Accuracy: {acc}')