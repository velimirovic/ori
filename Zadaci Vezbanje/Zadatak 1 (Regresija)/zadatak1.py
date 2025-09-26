# Zadatak 1 - Regresija sa neuronskom mrežom
# Predviđanje score-a na osnovu concentration-a

import pandas as pd  # Za rad sa CSV fajlovima i tabelama
import torch         # PyTorch - glavna biblioteka za neuronske mreže  
import torch.nn as nn        # Neuronski slojevi (Linear, ReLU, itd.)
import torch.optim as optim  # Optimizatori (ADAM)
from sklearn.model_selection import train_test_split  # Podela na train/test
from sklearn.preprocessing import StandardScaler      # Standardizacija podataka
from sklearn.metrics import mean_squared_error, r2_score  # Metrici za regresiju
import numpy as np   # Numerički proračuni

# Postavke za reproducibilnost - da svaki put dobijemo iste rezultate
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)    

df = pd.read_csv("math.csv")

#Prikazuje statistike o podacima, da znamo je l treba standardizacija ili ne
#print(df.describe())

df = df.dropna()
X = df[['concentration']].values
y = df['score'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Standardizacija 
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)  # fit_transform na trening!
X_test_scaled = scaler_x.transform(X_test)        # samo transform na test!

# Za regresiju cesto standardizujemo i y 
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()  # DODAJ .flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()        # DODAJ .flatten()

# Konvertuj podatke u PyTorch tensore
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Kreiraj klasu za neuronsku mrežu (regresija)
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.relu = nn.ReLU()
        
        # Kreiraj listu veličina slojeva
        sizes = [input_size] + hidden_sizes + [output_size]
        
        # Kreiraj listu linearnih slojeva
        self.layers = nn.ModuleList([
            nn.Linear(sizes[i-1], sizes[i]) for i in range(1, len(sizes))
        ])
    
    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # ReLU na sve slojeve osim poslednjeg
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out

input_size = X_train_tensor.shape[1]  # Broj ulaznih karakteristika
hidden_sizes = [64, 32]                # Broj neurona u skrivenim
output_size = 1                        # Jedan izlaz (score)

model = MLPRegressor(input_size, hidden_sizes, output_size)

criterion = nn.MSELoss() #MSE se koristi za regresiju !!!
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam optimizator

num_epochs = 1000
print("\nPočinje treniranje...")

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor.squeeze())  # 1D vs 2D problem!
    
    # Backward pass
    optimizer.zero_grad()  # Obriši gradijente
    loss.backward()        # Izračunaj gradijente  
    optimizer.step()       # Ažuriraj parametre
    
    # Prikaži progress svakih 200 epoha
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print("Treniranje završeno!")


#Evaluacija na test skupu
model.eval()
with torch.no_grad():  # Isključi računanje gradijenata
    # Predikcije na test skupu (skaliran)
    y_pred_scaled = model(X_test_tensor).squeeze()
    
    # Vrati predikcije u originalni opseg
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy().reshape(-1, 1)).flatten()
    
    # Izračunaj metrike
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n=== REZULTATI ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

# Prikaži nekoliko primera predikcija
print(f"\n=== PRIMERI PREDIKCIJA ===")
for i in range(min(8, len(y_test))):
    print(f"Concentration: {X_test[i][0]:.2f} | Stvarni: {y_test[i]:.2f} | Predviđen: {y_pred[i]:.2f}")