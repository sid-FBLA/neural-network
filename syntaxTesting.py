begString = '2012-01-'
for i in range (1, 20):
    endingDigit = str(i)
    if (i < 10) :
        endingDigit = '0' + endingDigit
    begString = begString + endingDigit
    print(begString)
    begString = '2012-01-'
