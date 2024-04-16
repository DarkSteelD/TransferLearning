# Basic data types and operations
number = 10
floating_point = 10.5
text = "Hello, Python!"
my_list = [1, 2, 3, 4, 5]

# Control structures: If-Else
if number > 5:
    print("Number is greater than 5")
else:
    print("Number is 5 or less")

# Loop: For
for item in my_list:
    print(item)

# Loop: While
counter = 0
while counter < 5:
    print("Counter is", counter)
    counter += 1

# Function definition
def add_numbers(a, b):
    return a + b

# Function call
result = add_numbers(5, 3)
print("The sum is:", result)

# Basic Sorting Algorithm: Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Basic Searching Algorithm: Linear Search
def linear_search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1

# Testing bubble sort and linear search
sample_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = bubble_sort(sample_list.copy())
print("Sorted list:", sorted_list)

search_result = linear_search(sorted_list, 22)
print("Found at index:", search_result if search_result != -1 else "Not found")

import random

# Class definition for a simple Bank Account
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Added {amount} to the balance")

    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance -= amount
            print(f"Withdrew {amount} from the balance")
        else:
            print("Insufficient balance")

# More advanced sorting algorithms: QuickSort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[random.randint(0, len(arr) - 1)]
        less = [x for x in arr if x < pivot]
        equal = [x for x in arr if x == pivot]
        greater = [x for x in arr if x > pivot]
        return quick_sort(less) + equal + quick_sort(greater)

# MergeSort
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = merge_sort(arr[:mid])
        R = merge_sort(arr[mid:])
        
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr

# Binary Search Algorithm
def binary_search(arr, left, right, x):
    if right >= left:
        mid = left + (right - left) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binary_search(arr, left, mid - 1, x)
        else:
            return binary_search(arr, mid + 1, right, x)
    else:
        return -1

# File I/O and Exception Handling
def read_numbers_from_file(filename):
    try:
        with open(filename, 'r') as file:
            numbers = [int(line.strip()) for line in file.readlines()]
        return numbers
    except FileNotFoundError:
        print("File not found")
        return []
    except ValueError:
        print("File contains non-integer values")
        return []

# Main function to execute everything
def main():
    # Bank account demo
    account = BankAccount("John Doe", 100)
    account.deposit(50)
    account.withdraw(30)

    # Sorting
    arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = quick_sort(arr.copy())
    print("QuickSort result:", sorted_arr)
    sorted_arr = merge_sort(arr.copy())
    print("MergeSort result:", sorted_arr)

    # Searching
    index = binary_search(sorted_arr, 0, len(sorted_arr) - 1, 22)
    print("Binary Search found at index:", index if index != -1 else "Not found")

    # File operations
    numbers = read_numbers_from_file("numbers.txt")
    print("Numbers from file:", numbers)

if __name__ == "__main__":
    main()
import os
import socket
from functools import wraps

# Decorator for function execution time
def time_it(func):
    import time
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f}s")
        return result
    return wrapper

# Generator for Fibonacci numbers
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Advanced File Handling: Writing to one file and reading from another simultaneously
class FileTransformer:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination

    def __enter__(self):
        self.src_file = open(self.source, 'r')
        self.dest_file = open(self.destination, 'w')
        return self

    def transform(self):
        for line in self.src_file:
            processed = line.strip().upper()  # Example transformation
            self.dest_file.write(processed + "\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.src_file.close()
        self.dest_file.close()

# Socket programming: Simple echo server
def start_echo_server(port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', port))
    server_socket.listen(1)
    print("Server listening on port", port)
    
    client_socket, addr = server_socket.accept()
    print("Received connection from", addr)
    
    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            client_socket.sendall(data)  # Echo back to client
    finally:
        client_socket.close()
        server_socket.close()



if __name__ == "__main__":
    main()
