digit= input("Enter a 4 digit number: ")
print("Original Number: ", digit)
print(len(digit))
swap= digit[-1]+digit[-3:-1]+digit[-len(digit)]
swap= digit[-1]+ digit[-2]+ digit[-3]+ digit[-len(digit)]
result = ""
for d in swap:
    result += str((int(d) + 7) % 10)

print("Final number:", result)
