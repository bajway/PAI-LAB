def isVowelOrConsonant(stringg):
    stringg.lower()
    letter= stringg[-1]

    if letter in "aeiou":
        return True
    else: return False


stringg= input("Enter a String: ")
if(isVowelOrConsonant(stringg)):
    print("It is a Vowel")
else: 
    print("It is a Consonant")
