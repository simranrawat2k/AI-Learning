try:
    with open("random.txt", "r") as file:
        data = file.read()

    print(data)

except:
    print("File not found")