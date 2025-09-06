# Write a program that finds the pair of numbers in a tuple whose sum is closest
# to zero.
min_sum = 2**31 - 1
numbers = (-42, 87, 0, -99, 13, 64, -7, -25, 48, -88)
# Convert to list and sort
list_num = list(numbers)
list_num.sort()

size= len(list_num)
left= 0
right= size - 1
num1=0
num2 = 0

while left < right:
    sum_num =list_num[left]+ list_num[right]
    if abs(sum_num) < abs(min_sum):
        min_sum = sum_num
        num1, num2 = list_num[left], list_num[right]
    
    if sum_num < 0:
        left += 1
    elif sum_num > 0:
        right -= 1
    else: 
        break

print("Pair whose sum is closest to zero is (", num1, ",", num2, ")")
print("Sum= ", min_sum)
