import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import torch.nn as nn

# za reproducibilnost
seed=42
torch.manual_seed(seed)

# Ucitavanje podataka
df = pd.read_csv('sms_spam.tsv', sep='\t', header=None, names=['status', 'text'])
df = df.dropna()

#print(df['SPAM'].value_counts()) => ham 4825, spam 747

#Pretvaranje labela u brojeve
label_encoder = LabelEncoder()
df['status'] = label_encoder.fit_transform(df['status'])  # ham -> 0, spam -> 1

# Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['status'], test_size=0.2, random_state=seed)

# Vektorizaija teksta - Bag of Words
vectorizer = CountVectorizer(stop_words='english')
X_train_bow = vectorizer.fit_transform(X_train) # fit_transform na trening!
X_test_bow = vectorizer.transform(X_test) # samo transform na test!

nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)

y_pred = nb_model.predict(X_test_bow)

#Evaluacija modela pre neuronske mreze
print(f'Naive Bayes Accuracy: {accuracy_score(y_pred, y_test):.4f}')

#Neuronska mreza
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        self.relu = nn.ReLU()

        sizes = [input_size] + hidden_sizes + [num_classes]
        
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
    
input_size = X_train_bow.shape[1]  # Broj karakteristika iz Bag of Words
hidden_sizes = [64,32]  # Primer skrivenih slojeva
num_classes = 2

model = MLPClassifier(input_size, hidden_sizes, num_classes)

X_train_tensor = torch.tensor(X_train_bow.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_bow.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 500
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluacija na test skupu
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f'Neural Network Test Accuracy: {accuracy.item():.4f}')

# prikaži meru tačnosti
# from sklearn.metrics import accuracy_score
# print(f'Accuracy: {accuracy_score(y_pred, y_test)}')