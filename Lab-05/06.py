def write_question():
    try:
        sentence = input("Enter a sentence: ")
        if sentence.strip().endswith('?'):
            with open("questions.txt", 'a') as f:
                f.write(sentence + "\n")
            print("Question saved.")
        else:
            print("Not a question.")
    except Exception as e:
        print("Error:", e)

write_question()
