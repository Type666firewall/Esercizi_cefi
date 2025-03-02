"""Esercizio 1
Creare una list comprehension che genera i quadrati dei numeri da 1 a 10."""

quadrati = [ x**2 for x in range(1,11)]

print(quadrati)

"""  
Esercizio 2
Utilizzare una list comprehension per convertire tutte le stringhe di una lista in maiuscolo.
"""

lista = ["ciao","addio","casa","Anto","jap"]

print([x.upper() for x in lista])

"""
Esercizio 3
Scrivere una list comprehension che estrae solo i numeri pari da una lista.
"""

import random


lista_random = [random.randrange(3, 189) for x in range(1,100)]
print(lista_random)
print([x for x in lista_random if x%2 == 0])

"""
Esercizio 4
Generare una lista delle lunghezze delle parole in una frase utilizzando list comprehensions.
"""

frase = "David Hume è stato un empirista Scozzese"

conto = [len(parola) for parola in frase.split()]
#split()	Splits the string at the specified separator, and returns a list

print(conto)


"""
Esercizio 5
Creare una list comprehension che include solo le stringhe di lunghezza superiore a 5 caratteri da una lista.
"""
lista = [parola for parola in frase.split() if len(parola)> 5]

print(lista)


"""
Esercizio 6
Utilizzare una list comprehension per creare una lista di tuple, ognuna contenente un numero e il suo quadrato."""

lista_di_tuple = [(x,x**2) for x in range(1,11)]

print(lista_di_tuple)


"""
Esercizio 7
Scrivere una list comprehension che filtra i numeri dispari di una lista e calcola il cubo di ciascuno.
"""
lista_cubi_dispari = [x**3 for x in range(1,11) if x%2!=0]

print(lista_cubi_dispari)


"""
Esercizio 8
Generare una lista di numeri interi casuali e utilizzare una list comprehension per selezionare 
solo quelli che sono multipli di 3.
"""

lista_multipli_3 = [x for x in lista_random if x%3 == 0]
#if type(x/3) == type(1)
#non funziona come previsto perché, in Python, l’operatore di divisione / restituisce sempre un float,

print(lista_multipli_3)



"""
Esercizio 9
Utilizzare una list comprehension per convertire una lista di gradi Fahrenheit in gradi Celsius.
"""
fahrenheit = [ x for x in range(32,133)]

celsius = [int((x-32)*5/9) for x in fahrenheit]

print(celsius)


"""
Esercizio 10
Creare una list comprehension che estrae le vocali da una stringa data.
"""

vocali_in_frase = [x for x in frase if x in "aeiouèàòùé"]

print(vocali_in_frase)