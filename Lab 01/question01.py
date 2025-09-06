# Create a program that checks whether a given string is a valid password:
# ● At least 8 characters long
# ● Contains both letters and digits
# ● Contains at least one special character (@, #, $, %)

password= input("Enter Your Password: ")
is_letter= False
is_length=False
is_digit=False
is_char=False
length= len(password)
for word in password:
    if(length>=8):
        is_length=True
    if(word.lower()>='a' and word.lower()<='z'):
        is_letter=True
    if(word>='0' and word<='9'):
        is_digit= True
    if(word=='@' or word=='#' or word=='$' or word=='%'):
        is_char=True    

if(is_digit and is_char and is_length and is_letter):
    print("You entered a valid Password")
else:
    print("You entered an invalid Password")