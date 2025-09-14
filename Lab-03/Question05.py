def palindrome_check(string):
    
    lowerString = string.lower()
    length= len(lowerString)
    for i in range(0,length//2):
        if(lowerString[i]==lowerString[length-1-i]):
            return True
        else: return False


string = input("Enter a string:")
check=palindrome_check(string)
if(check):
    print("String is a palindrome")
else:
    print("String is not a palindrome")