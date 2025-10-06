try:
    with open("employee.txt", 'w') as f:
        name = input("Name: ")
        cnic = input("CNIC: ")
        age = input("Age: ")
        salary = input("Salary: ")
        f.write(f"Name:{name}\nCNIC:{cnic}\nAge:{age}\nSalary:{salary}\n")
    with open("employee.txt", 'a') as f:
        contact = input("Contact: ")
        f.write(f"Contact:{contact}\n")
    with open("employee.txt", 'r') as f:
        print(f.read())
except Exception as e:
    print("Error:", e)
