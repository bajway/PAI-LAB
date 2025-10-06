try:
    filename = input("Enter file name: ")
    search = input("Word/Phrase to search: ")
    replace = input("Replace with: ")
    with open(filename, 'r') as f:
        content = f.read()
    content = content.replace(search, replace)
    with open(filename, 'w') as f:
        f.write(content)
    print("Replacement done.")
except FileNotFoundError:
    print("File not found.")
except Exception as e:
    print("Error:", e)
