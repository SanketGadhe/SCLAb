def mcculloch_pitts_model():
    print("Enter the weights:")
    w1 = float(input("Weight 1 = "))
    w2 = float(input("Weight 2 = "))
    print("Enter the threshold value:")
    theta = float(input("Theta = "))

    y = [0, 0, 0, 0]
    x1 = [1, 1, 0, 0]
    x2 = [1, 0, 1, 0]
    z = [1, 0, 0, 0]

    con = True
    while con:
        zin = [x1[i] * w1 + x2[i] * w2 for i in range(4)]

        for i in range(4):
            if zin[i] >= theta:
                y[i] = 1
            else:
                y[i] = 0

        print("Output of net:", y)

        if y == z:
            con = False
        else:
            print("Network is not learning!")
            w1 = float(input("Weight w1 = "))
            w2 = float(input("Weight w2 = "))
            theta = float(input("Theta = "))

    print("McCulloch Pitts Model Function:")
    print("Weight of Neuron:")
    print(f"w1 = {w1}")
    print(f"w2 = {w2}")
    print("Threshold Value:")
    print(f"Theta = {theta}")

# Call the function
mcculloch_pitts_model()
