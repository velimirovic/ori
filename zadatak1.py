import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score # Import koji ćemo koristiti za izračunavanje krajnje metrike
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Kostur
seed=42
torch.manual_seed(seed)
df = pd.read_csv("data/train.csv")

# Odbaci prazne redove
df = df.dropna()

#Da znam da li f1 ili accuracy
print("Distribucija zvanja:")
print(df['zvanje'].value_counts())

# Hijerarhijsko mapiranje zvanja
df['zvanje'] = df['zvanje'].map({'AsstProf': 0, 'AssocProf': 2, 'Prof': 1})

# One-hot encoding za oblast i pol (True/False)
df = pd.get_dummies(df, columns=['oblast', 'pol'], drop_first=True)

# Podela na trening i test skup
train, test = train_test_split(df, test_size=0.3, random_state=seed)

# Razdvajamo karakteristike (X) i ciljne vrednosti (Y)
X_train = train.drop('zvanje', axis=1)
y_train = train['zvanje']

X_test = test.drop('zvanje', axis=1)
y_test = test['zvanje']

# Standardizacija numerickih karakteristika (X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit_transform na trening skupu !!!
X_test_scaled = scaler.transform(X_test) # transform na test skupu !!!

# Konvertujem pandas (tabelu) u Pytorch tensore (brojke)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) # dtype=torch.long jer su klase (0,1,2)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long) # dtype=torch.long jer su klase (0,1,2)



# Pravimo klasu za neuronsku mrezu
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



#Kreiraj model
input_size = X_train_tensor.shape[1] # broj karakteristika (5)
hidden_sizes = [64, 32] # dva skrivena sloja sa po
num_classes = 3 # tri klase (0,1,2)
model = MLPClassifier(input_size, hidden_sizes, num_classes)

# Loss funkcija i optimizator
criterion = nn.CrossEntropyLoss() # Cross-Entropy Loss za klasifikaciju
optimizer = optim.Adam(model.parameters(), lr=0.05) # Adam optimizator 

#Treniranje
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


# Evaluacija
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    f1 = f1_score(y_pred=predicted, y_true=y_test_tensor, average='micro')
    print(f'F1: {f1}')
