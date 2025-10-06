try:
    filename = input("Enter file name: ")
    with open(filename, 'r') as f:
        content = f.read()
        chars = len(content)
        words = len(content.split())
        print("Characters:", chars)
        print("Words:", words)
except FileNotFoundError:
    print("File not found.")
except Exception as e:
    print("Error:", e)
