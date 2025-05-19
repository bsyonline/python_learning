a = 10
b = 0
try:
    c = a / b
except ZeroDivisionError as e:
    print(f"错误: {e}")


try:
    # code that may cause an exception
    open("file.txt", "r")
    c = a / b
except ZeroDivisionError as e1:
    # handle exception
    print(f"错误: {e1}")
except FileNotFoundError as e2:
    # handle exception
    print(f"错误: {e2}")


try:
    c = a / b
except (ZeroDivisionError, FileNotFoundError):
    # handle exception
    print("handle exception")




try:
    c = a / b
    print(c)
except ZeroDivisionError as error:
    print(error)
finally:
    # the code that always executes
    print('Finishing up.')


try:
    c = a / b
finally:
    # the code that always executes
    print('Finishing up.')


try:
    c = a / b
except ZeroDivisionError as error:
    print(error)
else:
    print(c)
finally:
    # the code that always executes
    print('Finishing up.')