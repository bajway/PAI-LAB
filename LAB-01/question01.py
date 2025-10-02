number= int(input("Enter a number: "))
x= float(number)
y= str(number)
z= complex(number)

print(type(x))
print(type(y))
print(type(z))
temp= number
while(number>1):
    
    number= number-2
if(number==0):
    print("Even Number")
else:
    print("Odd Number")