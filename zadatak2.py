import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import torch.optim as optim

# za reproducibilnost
seed=42
torch.manual_seed(seed)
df = pd.read_csv('./data/south_park_train.csv')

df = df.dropna()
print("Distribucija likova:")
print(df['Character'].value_counts())

#print(f"\nJedinstveni likovi: {df['Character'].unique()}")
#print(f"Broj likova: {df['Character'].nunique()}")
#print(df['Character'].value_counts())

# Enkodiranje karaktera u brojeve (Character → 0,1,2,3...)
label_encoder = LabelEncoder()
df['Character'] = label_encoder.fit_transform(df['Character'])

# Podela na train i test skup
X_train, X_test, y_train, y_test = train_test_split(df['Line'], df['Character'], test_size=0.2, random_state=seed)

#Bag of Words - vektorizaija teksta (kao standardizacija za numericke podatke)
vectorizer = CountVectorizer(stop_words='english')
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

#Kreiranje naivnog bajesa
nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)

# Predikcija na test skupu
y_pred = nb_model.predict(X_test_bow)

#Evaluacija prodela pre neuronske mreze
from sklearn.metrics import accuracy_score, f1_score, classification_report
print(f'Naive Bayes Accuracy: {accuracy_score(y_pred, y_test):.4f}')
print(f'Naive Bayes F1-score (macro): {f1_score(y_pred, y_test, average="macro"):.4f}')
print(f'Naive Bayes F1-score (weighted): {f1_score(y_pred, y_test, average="weighted"):.4f}')

#Neuronska mreza
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLPClassifier, self).__init__()
        self.relu = nn.ReLU()
        sizes = [input_size] + hidden_sizes + [num_classes]
        self.layers = nn.ModuleList([nn.Linear(sizes[i - 1], sizes[i]) for i in range(1, len(sizes))])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out

#Parametri
input_size = X_train_bow.shape[1]  # Broj karakteristika (veličina vokabulara)
hidden_sizes = [64]  # 🎯 SAMO 1 SLOJ! Minimal overfitting
num_classes = df['Character'].nunique()  # Broj klasa (likova)
mlp_model = MLPClassifier(input_size, hidden_sizes, num_classes)

# Konvertuj sparse matrix u dense i u tensore
X_train_tensor = torch.tensor(X_train_bow.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_bow.toarray(), dtype=torch.float32)

# Loss funkcija i optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)  # 🎯 BOLJE: Manji LR!

# Treniranje (manje epoha jer imamo više podataka)
num_epochs = 300  # 🎯 JOŠ VIŠE epoha!
print("Treniranje neuronske mreže...")

for epoch in range(num_epochs):
    outputs = mlp_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 25 == 0:  # 🎯 Prikaži progress češće
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluacija neuronske mreže
with torch.no_grad():
    mlp_model.eval()
    outputs = mlp_model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    #mlp_accuracy = accuracy_score(y_test, predicted)
    #print(f'Neuronska mreža Accuracy: {mlp_accuracy:.4f}')
    f1_macro = f1_score(y_pred=predicted, y_true=y_test, average='macro')
    f1_weighted = f1_score(y_pred=predicted, y_true=y_test, average='weighted')
    print(f'Neuronska mreža F1-score (macro): {f1_macro:.4f}')
    print(f'Neuronska mreža F1-score (weighted): {f1_weighted:.4f}')

# Kostur
# prikaži meru tačnosti
# from sklearn.metrics import accuracy_score
# print(f'Accuracy: {accuracy_score(y_pred, y_test)}')