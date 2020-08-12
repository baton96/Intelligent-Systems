import math

import numpy as np


def trapezoid(val, bottomLeft, upperLeft, upperRight, bottomRight):
    if (val < bottomLeft):
        return 0
    elif (val < upperLeft):
        return (val - bottomLeft) / (upperLeft - bottomLeft)
    elif (val < upperRight):
        return 1.0
    elif (val < bottomRight):
        return (bottomRight - val) / (bottomRight - upperRight)
    else:
        return 0


def pi(val, bottomLeft, upperLeft, upperRight, bottomRight):
    if (val < (upperLeft + upperRight) / 2):
        return 1 / (1 + math.exp(k * ((bottomLeft + upperLeft) / 2 - val)))
    else:
        return 1 / (1 + math.exp(k * (val - (upperRight + bottomRight) / 2)))


def fuzzySet(bottomLeft, upperLeft, upperRight, bottomRight, stop, start=0):
    return [fun(i, bottomLeft, upperLeft, upperRight, bottomRight) for i in np.arange(start, stop, step)]


def tNorm(a, b):
    try:
        _ = iter(a)
    except:
        return min(a, b)  # Gödel–Dummett t-norm
    # return a*b#product t-conorm
    # return max(a+b-1,0)#Łukasiewicz t-norm
    # return min(a,b) if max(a,b)==1 else 0#drastic t-norm
    # return (a*b)/(2-(a+b-a*b))#Einstein t-norm
    # return (a*b)/(a+b-a*b)#Hamacher t-norm
    else:
        try:
            _ = iter(b)
        except:
            b = np.array([b] * len(a))
        return np.fmin(a, b)


# return np.multiply(a,b)#product t-conorm
# return [max(x+y-1,0) for x,y in zip(a,b)]#Łukasiewicz t-norm
# return [min(x,y) if max(x,y)==1 else 0 for x,y in zip(a,b)]#drastic t-norm
# return [(x*y)/(2-(x+y-x*y)) for x,y in zip(a,b)]#Einstein t-norm
# return [(x*y)/(x+y-x*y) for x,y in zip(a,b)]#Hamacher t-norm

def sNorm(a, b):
    try:
        _ = iter(a)
    except:
        return max(a, b)  # Gödel–Dummett t-norm
    # return a+b-a*b#product t-conorm
    # return min(a+b,1)#Łukasiewicz t-conorm
    # return max(a,b) if min(a,b)==0 else 1#drastic t-conorm
    # return (a+b)/(1+a*b)#Einstein t-conorm
    # return (a+b-2*a*b)/(1-a*b)#Hamacher t-conorm
    else:
        try:
            _ = iter(b)
        except:
            b = np.array([b] * len(a))
        return np.fmax(a, b)


# return [x+y-x*y for x,y in zip(a,b)]#product t-conorm
# return [min(x+y,1) for x,y in zip(a,b)]#Łukasiewicz t-conorm
# return [max(x,y) if min(x,y)==0 else 1 for x,y in zip(a,b)]#drastic t-conorm
# return [(x+y)/(1+x*y) for x,y in zip(a,b)]#Einstein t-conorm
# return [(x+y-2*x*y)/(1-x*y) for x,y in zip(a,b)]#Hamacher t-conorm


# defuzz = lambda agg: len(agg)-np.argmax(agg[::-1])-1#max of maximum
# defuzz = np.argmax#min of maximum
defuzz = lambda agg: np.mean(np.where(agg == max(agg)))  # mean of maximum

step = 0.1
k = 2.5
fun = trapezoid

tSolid = fuzzySet(0, 0, 4, 8, 12)
tSolidVal = fun(5, 0, 0, 4, 8)
tSolidClipped = tNorm(tSolid, tSolidVal)

tSoft = fuzzySet(4, 8, 12, 12, 12)
tSoftVal = fun(5, 4, 8, 12, 12)
tSoftClipped = tNorm(tSoft, tSoftVal)

wHeavy = fuzzySet(3, 6, 9, 9, 9)
wHeavyVal = fun(4.5, 3, 6, 9, 9)
wHeavyClipped = tNorm(wHeavy, wHeavyVal)

wLight = fuzzySet(0, 0, 0, 6, 9)
wLightVal = fun(4.5, 0, 0, 0, 6)
wLightClipped = tNorm(wLight, wLightVal)

gFirm = fuzzySet(2, 3, 4, 4, 4)
gMedium = fuzzySet(1, 2, 2, 3, 4)
gGentle = fuzzySet(0, 0, 1, 2, 4)

rule = tNorm(tSolidVal, wHeavyVal)
gFirmClipped = tNorm(gFirm, rule)

rule = sNorm(tNorm(tSolidVal, wLightVal), tNorm(tSoftVal, wHeavyVal))
gMediumClipped = tNorm(gMedium, rule)

rule = tNorm(tSoftVal, wLightVal)
gGentleClipped = tNorm(gGentle, rule)

aggregated = sNorm(sNorm(gFirmClipped, gMediumClipped), gGentleClipped)

print(defuzz(aggregated) * step)
