x = [0, 1, 4, 5, 2, 5 ,23, -3423, 0.8, 80000]
absmin = x[0]

for i in range(len(x)):
    absmin_new = x[i]
    if absmin_new < absmin:
        absmin = absmin_new

print(absmin)
