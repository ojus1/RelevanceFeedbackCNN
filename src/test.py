def get_bound(i):
    i += 1
    if(i % 4 == 0):
        return (i-1, i+3)
    else:
        remainder = i % 4
        return (i - remainder, i + (4 - remainder - 1))

print(get_bound(0))
print(get_bound(4))
print(get_bound(2))
print(get_bound(11))
print(get_bound(24))
print(get_bound(67))