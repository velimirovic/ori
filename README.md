Stanje mora definisati:

Da li je ciljano stanje
Koja stanja mogu da slede (prelazi stanja)
Funkciju poređenja sa drugim stanjima
Roditeljsko stanje (za rekonstrukciju putanje)

Za vođene pretrage dodatno:

Trenutnu cenu g(n)
Heuristiku h(n)

Algoritmi pretrage
Slepe pretrage
Koriste fiksnu strategiju odabira čvorova:
BFS (Prvi u širinu)

Struktura: red (queue)
Kompletna i optimalna
Visoka memorijska zahtevnost

DFS (Prvi u dubinu)

Struktura: stek (stack)
Nije optimalna, može biti nekompletna
Mala memorijska zahtevnost

IDFS (Iterativni DFS)

Kombinuje prednosti BFS i DFS
Postavlja ograničenje dubine koje postupno povećava

Vođene pretrage
Koriste heurističku funkciju:
UCS (Uniform-cost search)

f(n) = g(n) - bira stanja sa najmanjom dosadašnjom cenom
Kompletna i optimalna

Greedy Search

f(n) = h(n) - bira stanja najbliža cilju
Nije kompletna ni optimalna

A*

f(n) = g(n) + h(n) - kombinuje UCS i Greedy
Kompletna i optimalna (uz dopustivu i doslednu heuristiku)

Heuristike
Dopustiva heuristika: 0 ≤ h(n) ≤ h*(n) (ne precenjuje cenu do cilja)
Dosledna heuristika: h(n) ≤ c(n,m) + h(m) (razlika procena nije veća od stvarne cene)
Primeri: Manhattan rastojanje, Euklidsko rastojanje, maksimalno rastojanje po osi.
Ključne karakteristike

Kompletnost: garantuje pronalaženje rešenja ako postoji
Optimalnost: garantuje najbolje rešenje
Heuristika ne sme menjati rezultat - za promenu rezultata redefinišemo problem
Sve pretrage imaju istu strukturu, razlikuju se samo u select_state funkciji
