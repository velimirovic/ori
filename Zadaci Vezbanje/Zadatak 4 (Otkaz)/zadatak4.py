import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

seed = 42
torch.manual_seed(seed)
df = pd.read_csv("customer_churn.csv")

df = df.dropna()
#print(df['churn'].value_counts())

# 1 => Enkodiranje prvo
le_i = LabelEncoder()
le_v = LabelEncoder()
le_c = LabelEncoder()

df['international plan'] = le_i.fit_transform(df['international plan']) #moglo je i one-hot ali ovako ima vise smisla
df['voice mail plan'] = le_v.fit_transform(df['voice mail plan']) #moglo je i one-hot ali ovako ima vise smisla
df['churn'] = le_c.fit_transform(df['churn']) #y kolona uvek ide encodingom

# 2 => Podela podataka na train i test
features = ['international plan', 'voice mail plan', 'number vmail messages', 'total intl calls', 'total night calls', 'total day calls']

X = df[features]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# 3 => Standardizuj numericke
numeric = ['number vmail messages', 'total intl calls', 'total night calls', 'total day calls']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric] = scaler.fit_transform(X_train[numeric])
X_test_scaled[numeric] = scaler.transform(X_test[numeric])

# 4 => Konvertovanje u tensore
X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 5 => Pravimo klasu za neuronsku mrezu
class MLPClassifier(nn.Module):
    #Konstruktor (koliko ulaza ima, koliko skrivenih slojeva [64,32], koliko izlaza ima)
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.relu = nn.ReLU()

        #Kreiraj listu velicina slojeva
        sizes = [input_size] + hidden_sizes + [output_size]

        #Kreiraj listu linearnih slojeva
        self.layers = nn.ModuleList([
            nn.Linear(sizes[i-1], sizes[i]) for i in range(1, len(sizes))
        ])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            #ReLU na sve slojeve osim poslednjeg
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out
    
# 6 => Kreiraj model
input_size = X_train_tensor.shape[1] # broj karakteristika (6)
hidden_sizes = [64] # dva skrivena sloja sa po
num_classes = 2 

model = MLPClassifier(input_size, hidden_sizes, num_classes)

# 7 => Loss funkcija i optimizator
criterion = nn.CrossEntropyLoss() # Cross-Entropy Loss za klasifikaciju
optimizer = optim.Adam(model.parameters(), lr=0.05) # Adam optimizator 

# 8 => Treniranje
num_epochs = 1000
print("Treniranje..")

for epoch in range(num_epochs):
    #Forward
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    #Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Print svakih 100 epoha
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Treniranje gotovo.")


# 9 => Evaluacija
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    acc = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
    print(f'Accuracy: {acc}')
