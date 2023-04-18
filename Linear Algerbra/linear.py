import numpy as np

ySum = 0
xSum = 0
xySum = 0
x2Sum = 0
a = np.array([[0,0,0,0]])
y = np.array([[0]])
print(a.shape)
print(y, '\n\n')
print('\n\n------------------------------- Estimated Price Algorithm Data -----------------------\n\n')
area = int(input("Land size(in sq. Feets): "))
noOfBedRooms = int(input("Number of Bedrooms: "))
noOfFloors = int(input("Number of Floors: "))
homeAge = int(input("Age Of Home: "))
price = int(input("Land prize(in 1000k): "))
a[0,:] = [area, noOfBedRooms, noOfFloors, homeAge]
y[0,:] = price
print(a)
print(y, '\n\n')

check = input('Wanna enter more data [y/n]: ')
if(check == 'y') :
    while(1) :
        area = int(input("Land size(in sq. Feets): "))
        noOfBedRooms = int(input("Number of Bedrooms: "))
        noOfFloors = int(input("Number of Floors: "))
        homeAge = int(input("Age Of Home: "))
        price = int(input("Land prize(in 1000k): "))
        y = np.append(y, [[price]], axis=0)
        a = np.append(a, [[area, noOfBedRooms, noOfFloors, homeAge]], axis=0)
        print(a)
        print(y, '\n\n')
        check = input('Wanna enter more data [y/n]: ')
        if(check == 'n') :
            break



print(a)
# for i in a :
#     print(i)
#     xSum += i[0]
#     ySum += i[1]
#     xySum += i[0]*i[1]
#     x2Sum += pow(i[0],2)

# s = a.size
# xMean = xSum/s
# yMean = ySum/s
# xyMean = xySum/s
# x2Mean = x2Sum/s

# c = (x2Mean*yMean - xMean*xyMean)/(x2Mean - pow(xMean, 2))
# m = (xyMean - xMean*yMean)/(x2Mean - pow(xMean, 2))

# y = np.transpose(y)
weights = np.linalg.solve(a,y)
print("Weigths : ", weights)


print("y = ", weights,"x")
# h = c + mx

while(1) :

    print('\n\n------------------------------- Estimated Price Algorithm -----------------------\n\n')
    area = int(input("Land size(in sq. Feets) : "))
    noOfBedRooms = int(input("Number of Bedrooms : "))
    noOfFloors = int(input("Number of Floors : "))
    homeAge = int(input("Age Of Home : "))
    print('\n');
    inputMatrice = np.array([area, noOfBedRooms, noOfFloors, homeAge]);
    estimatedPrice = np.dot(inputMatrice, weights)
    if(estimatedPrice[0] <= 0) :
        print("Can't estimate price with give data :(")
        break
    print("Estimated Price : ", estimatedPrice[0], '\n');
    check = input('Wanna estimate more {y/n}: ')
    if(check == 'n') :
        break

