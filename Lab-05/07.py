def fix_file():
    try:
        with open("replacement_needed.txt", 'r') as f:
            content = f.read()
        corrected = ""
        for ch in content:
            if ch == 'x':  # suppose 'x' is wrong letter
                corrected += 'e'
            else:
                corrected += ch
        with open("replacement_needed.txt", 'w') as f:
            f.write(corrected)
        print("Correction done.")
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("Error:", e)

fix_file()
