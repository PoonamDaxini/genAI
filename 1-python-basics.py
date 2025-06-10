## declare and assign variable

age = 32
height = 5.1
name = "Krishna"
is_student = True

print("age: ", age)
print("height: ", height)
print("name: ", name)

# num1 = float(input("Enter first number: "))
# num2 = float(input("Enter second number: "))
# sum = num1 + num2
# diff = num1 - num2
# product = num1 * num2
# division = num1 / num2
# print("Difference: ", diff)
# print("Product: ", product)
# print("Division: ", division)
# print("Sum: ", sum)


first_set = {'xyz': 1, 'abc': 2, }
second_set = {'xyz': 1, 'abc': 2, 'pqr': 6}

print(set(second_set.items()) - set(first_set.items()))
print(set(first_set.items()) - set(second_set.items()))
