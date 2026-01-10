def vowel_consonant(word):
    count_vowels  = 0
    count_consonants = 0
    for i in range(len(word)):
        if (word[i] == "A" or word[i] == "a") or (word[i] == "e" or word[i] == "E") or (word[i] == "i" or word[i] == "I") or (word[i] == "o" or word[i] == "O") or (word[i] == "u" or word[i] == "U"):
            
            count_vowels += 1
    count_consonants = len(word) - count_vowels
    return count_vowels,count_consonants
   
    
word = input("Enter any word you wish: ")
print("The number of vowels and consonants in the word: ",vowel_consonant(word))

def common_number_predictor(list_1, list_2):
    list_1 = list(list_1)
    list_2 = list(list_2)

    count_common_elements = 0
    for i in range(len(list_1)):
        for j in range(len(list_2)):
            if list_1[i] == list_2[j]:
                count_common_elements += 1
    return count_common_elements

n = int(input("Enter the size of list one: "))
m = int(input("Enter the size of list two: "))

list_1 = []
list_2 = []

for i in range(n):
    num_1 = input("Enter the numbers in list one: ")
    list_1.append(num_1)

for j in range(m):
    num_2 = input("Enter the numbers in list two: ")
    list_2.append(num_2)

print(
    "The number of common elements is: ",
    common_number_predictor(set(list_1), set(list_2))
)

def matrix_multiply(A, B, r1, c1, r2, c2):
    if c1 != r2:
        return None

    result = []
    for i in range(r1):
        row = []
        for j in range(c2):
            row.append(0)
        result.append(row)

    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                result[i][j] += A[i][k] * B[k][j]

    return result


r1 = int(input("Enter number of rows of matrix A: "))
c1 = int(input("Enter number of columns of matrix A: "))
r2 = int(input("Enter number of rows of matrix B: "))
c2 = int(input("Enter number of columns of matrix B: "))

A = []
for i in range(r1):
    row = []
    for j in range(c1):
        row.append(int(input()))
    A.append(row)

B = []
for i in range(r2):
    row = []
    for j in range(c2):
        row.append(int(input()))
    B.append(row)

result = matrix_multiply(A, B, r1, c1, r2, c2)

if result is None:
    print("Error: Matrix multiplication not possible")
else:
    print("Product of matrices A and B:")
    for row in result:
        print(row)

import random

def mean_median_mode():
    sum = 0
    numbers = []
    for i in range(100):
        num = random.randint(100, 150)
        numbers.append(num)
        sum += num
    mean = sum / 100
    numbers.sort()
    median = (numbers[49] + numbers[50]) / 2
    mode = numbers[0]
    max_count = 0
    for j in numbers:
        count = 0
        for k in numbers:
            if j == k:
                count += 1
        if count > max_count:
            max_count = count
            mode = j
    return mean, median, mode


print("(mean, median, mode) =", mean_median_mode())

def transpose_matrix():
    r = int(input("Enter rows: "))
    c = int(input("Enter columns: "))
    matrix = []
    for i in range(r):
        row = []
        for j in range(c):
            num = int(input("Enter number: "))
            row.append(num)
        matrix.append(row)
    transpose = []
    for i in range(c):
        row = []
        for j in range(r):
            row.append(matrix[j][i])
        transpose.append(row)
    print("Transpose:")
    for i in range(c):
        for j in range(r):
            print(transpose[i][j], end=" ")
        print()

transpose_matrix()

