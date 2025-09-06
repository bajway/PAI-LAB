# Create a student management system where you can:
# ● Add student name + marks in a dictionary
# ● Update marks
# ● Delete a student
# ● Find the topper

StudentRecord= {}

while True:
    print("****M E N U****", end="\n \n")
    print("1. Add a Student")
    print("2. Update Marks")
    print("3. Delete a Student")
    print("4. Find the Topper")
    print()
    
    choice= int(input("Select the operation: (1,2,3,4): (0 to exit) " ))
    if (choice==0):
        print("Program Terminated, GoodBye")
        break
    elif(choice==1):
        name= input("Enter student name: ")
        marks= float(input("Enter Marks: "))
        StudentRecord.update({name.lower(): marks})
        
    
    elif(choice==2):
        nameFind= input("Enter the name of student whom marks needs to be updated: ")
        if nameFind.lower() in StudentRecord:
            newMarks= float(input("Enter updated marks: "))
            StudentRecord[nameFind.lower()]= newMarks
        else:
            print("Student name not found!")
    elif(choice==3):
        nameFind= input("Enter the name of student you want to delete: ")
        if nameFind.lower() in StudentRecord:
            StudentRecord.pop(nameFind.lower())
            print("Record Deleted")
        else:
            print("Student Not Found!")
    elif(choice==4):
        max_score= max(StudentRecord.values())
        print("Topper Details: ")
        for name,marks in StudentRecord.items():
            if(marks==max_score):
                print(name, ":", marks, end="\n")
            else:
                print("No record Found")
    else:
        print("Invalid Choice!")