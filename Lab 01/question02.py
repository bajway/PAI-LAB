# Implement matrix addition and multiplication using nested lists.
first_matrix = [[0 for _ in range(3)] for _ in range(3)]  
for x in range(0, 3):
    for y in range(0, 3):
        first_matrix[x][y] = int(input("Enter values of first matrix: "))

second_matrix = [[0 for _ in range(3)] for _ in range(3)]
for m in range(0, 3):
    for n in range(0, 3):
        second_matrix[m][n] = int(input("Enter values of second matrix: "))

sum_matrix = [[0 for _ in range(3)] for _ in range(3)]
for a in range(0, 3):
    for b in range(0, 3):
        sum_matrix[a][b] = first_matrix[a][b] + second_matrix[a][b]

print("Sum: ")
print(sum_matrix)

multi_matrix = [[0 for _ in range(3)] for _ in range(3)]
for p in range(0, 3):
    for q in range(0, 3):
        for r in range(0, 3):
            multi_matrix[p][q] += first_matrix[p][r] * second_matrix[r][q]

print("Multiplication: ")
print(multi_matrix)
