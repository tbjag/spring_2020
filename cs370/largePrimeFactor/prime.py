import math
num = 600851475143


square = math.floor(math.sqrt(num))




def ifPrime(num):
    for i in range(2,num-1):
        if(num%i == 0):
            return True

