size = int(input("Enter the size of list:"))
list1 = [] 
list2 = []
for i in range(0,size):
    value = input(f"Enter the {i+1} value of list 1:")
    list1.append(value)
for i in range(0,size):
    value = input(f"Enter the {i+1} value of list 2:")
    list2.append(value)
def Dictionary(list1,list2,size):
    myDictionary = {}
    for i in range(0,size):
        myDictionary[list1[i]] = list2[i]
    return myDictionary
print(Dictionary(list1,list2,size))

