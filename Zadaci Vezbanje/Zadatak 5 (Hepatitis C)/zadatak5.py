import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score

seed = 42
torch.manual_seed(seed)
df = pd.read_csv('hepatitisc.csv')
df = df.dropna()

#print(df['Category'].value_counts()) koristicu f1

#Enkodiranje
le_c = LabelEncoder()
le_s = LabelEncoder()

df['Category'] = le_c.fit_transform(df['Category'])
df['Sex'] = le_s.fit_transform(df['Sex'])

#Podela
features = ['Sex', 'Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
X = df[features]
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

#Bag of Words - vektorizaija teksta (kao standardizacija za numericke podatke)
#vectorizer = CountVectorizer(stop_words='english')
#X_train_bow = vectorizer.fit_transform(X_train)
#X_test_bow = vectorizer.transform(X_test)

#Standardizacija
numeric = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric] = scaler.fit_transform(X_train[numeric])
X_test_scaled[numeric] = scaler.transform(X_test[numeric])

#Tenzori
X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

#Neuronska mreza
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

#Model
input_size = X_train_tensor.shape[1] #broj karakteristika (9)
hidden_sizes = [64]
output_size = 5

model = MLPClassifier(input_size, hidden_sizes, output_size)

#Loss i Optimizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Treniranje
num_epochs = 500
print("Treniranje")
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Treniranje gotovo.")

#Evaluacija
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    f1 = f1_score(y_test_tensor, predicted, average='micro')
    print(f'F1: {f1}')