import json
try:
    data = {'Ali':23,'Saad':24,'Salman':15,'Shams':25,'Sadiq':46,'Hammad':23}
    with open("people.json", 'w') as f:
        json.dump(data, f)
    with open("people.json", 'r') as f:
        info = json.load(f)
    max_age = max(info.values())
    print("Max age:", max_age)
    print("People with max age:")
    for name, age in info.items():
        if age == max_age:
            print(name)
except Exception as e:
    print("Error:", e)
