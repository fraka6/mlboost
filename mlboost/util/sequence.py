
def max_sequential_value(sequence, value=True):
    n = 0
    max_n = 0
    
    for el in sequence:
        if el==value:
            n+=1
            max_n = max(max_n, n)
        else:
            n=0
    return max_n

