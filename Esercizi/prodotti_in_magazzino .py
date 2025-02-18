prodotti_in_magazzino = [
    {"nome": "Laptop Dell XPS 15", "prezzo": 1599.99, "categoria": "Elettronica", "quantita": 10, "stato": "Disponibile"},
    {"nome": "Smartphone Samsung Galaxy S23", "prezzo": 899.99, "categoria": "Elettronica", "quantita": 5, "stato": "Disponibile"},
    {"nome": "Cuffie Bose QuietComfort", "prezzo": 299.99, "categoria": "Elettronica", "quantita": 15, "stato": "Disponibile"}
]
print(type(prodotti_in_magazzino))
#RICORDATI CHE PRODOTTI MAGAZZINO E' UNA LISTA

def spacchettamento():
    for prodotto in prodotti_in_magazzino:
        if type(prodotto) == dict:
            print(prodotto)
            
            
spacchettamento()
#input_utente = input("Inserisci il nome del prodotto venduto: ")



"""
def ricerca(input_utente):
    if input_utente in prodotti_in_magazzino.values():
        print(prodotti_in_magazzino)
        
        
ricerca("")"""