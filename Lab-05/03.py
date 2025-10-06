try:
    list1 = input("Enter first list (comma separated): ").split(',')
    list2 = input("Enter second list (comma separated): ").split(',')
    if len(list1) != len(list2):
        raise ValueError("Lists must have same length.")
    data = {}
    for i in range(len(list1)):
        data[list1[i].strip()] = list2[i].strip()
    with open("dictionary.txt", 'w') as f:
        for k, v in data.items():
            f.write(f"{k}:{v}\n")
    print("Dictionary saved.")
except Exception as e:
    print("Error:", e)
