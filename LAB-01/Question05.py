name= input("Enter Your Name: ")
birthYear=input("Enter your birth Year e.g(2005): ")
symbols= "@#%&*"
password= ""
password +=name[:3]
password += birthYear[-2:]
assciiValue= ord(name[0])
symbolIndex= assciiValue%len(symbols) #divided by len of symbols the first character of name

password= password+ symbols[symbolIndex]
print(password)