from ctypes import *

class Struc(Structure):
    _fields_ = [
        ('a', c_int),
    ]

s = Struc(a = 5)
p = pointer(s)

p.contents.a = 10
print(s.a)
p[0].a = 20
print(s.a)

s = c_int(5)
p = pointer(s)
p.contents
print(s)
p[0] = 20
print(s)
