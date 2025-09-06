# Given a list of student records stored as tuples (name, marks), write a
# program to:
# ● Sort students by marks in descending order
# ● Print the top 3 students
import operator
student_data = [("zain", 60), ("ali", 95), ("omer", 50), ()]
student_data.sort(key=operator.itemgetter(1), reverse=True) #sort the second indexes i.e 1 in this case
for i in range(3):
    print(student_data[i])