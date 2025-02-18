a = "*"
for x in range(1,5):
    print(a)
    a+="*"

for num in range(1,5):
   print("*"* num)

i= 0

while i < 4:
    i+=1
    print("*"* i )


lista = [1, 5, "b", "ciao", 4, (3, 4),]
#il risultato dei numeri interi nella lista --->10
somma = 0

for num in lista:
    if type(num) == int:
        somma+=num
    else:
        continue
print(somma)

i = 20.0

print(type(i))

while len(str(i)) > 10:
    i += 2
    
else:
    i//= 2

print(i)
