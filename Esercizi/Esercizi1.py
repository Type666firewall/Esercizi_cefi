
print("Ciao,",'\n',"mi chiamo",'\n',"Antonio", sep="")

anno_di_nascita = int(input("Inserisci il tuo hanno di nascità: "))

anno_corrente = int(input("inserisci l'anno corrente: "))

print("Hai", anno_corrente-anno_di_nascita, "anni")


#
lst = [100, "ciao", ["python" , "java"]]

lst1=(lst[2][0])

print(lst1[3:4])


lst = [-5,100,25,-1.5,200,55,]

print(lst[2:5:2])


#append aggiuge un solo elemento alla fine della lista, extend fonde due liste
lst = ["a", "b", "c", 2, 4, ]

del lst[0:-2]

print(lst)

lista = "12,34"

lista2 = lista.split(",",)

print(lista2,lista, sep="<-->")

print(len(lista)<len(lista2))

lista3 = ["A","b","C",]

lista3.sort()

print(lista3)

lista_copiata = lista3.copy()

lista3.append("D")

lista3.sort()

print(lista3,lista_copiata, sep=" ")


lista = ["elemento"]
i = 0

while i < 4:
    user_input = input("inserisci una stringa: ")

    if len(user_input) < 5:
        lista.append(user_input)
    
    else:
        lista.pop()
    
    i+=1

print(lista)

lista1 = ["A", "B", "C", "F", "D"]

if lista1.index("A"):
    print("A")
    
else:
    print(lista1.count("A"))
    
#la "A" ha indice 0 il primo if quindi darà 0 rendendo il bool False
#entrando quindi nell'else


