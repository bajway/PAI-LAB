def Employee(name, salary=10000):
    if(salary>0):
        tax= salary*0.02
        salary=salary-tax
        print("Employee Name: " + name)
        print("Salary after tax: ", salary)
    else:
        print("Enter Valid salary!!")


employeeName= input("Enter Employee name: ")
monthlySalary= float(input("Enter Salaray:"))
Employee(employeeName, monthlySalary)
Employee(employeeName)