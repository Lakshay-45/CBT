# Program to create upper triangular, lower triangular and pyramid pattern

# Taking user input to define the pattern length
while(True):
    try:
        n = int(input("Enter the value of length: "))
        break
    except:
        print("Please enter a valid integer value")

# Printing upper triangular pattern
print("Upper triangular:")
for i in range(n):
    print(" " * (i) + "*" * (n-i))

# Printing lower triangular pattern
print("Lower triangular:")
for i in range(n):
    print("*" * (i+1))

# Printing pyramid pattern
print("Pyramid:")
for i in range(n):
    print(" " * (n-i-1)+ "* " * (i+1))