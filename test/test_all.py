import unittest

# if you create a new test_xxx.py file, please put it here.
tests = [
"test_conditioning",
"test_mask",
"test_shape",
"test_conditioning",
"test_averaging",
]

for t in tests:
    exec "from %s import *" % t

if __name__ == "__main__":
    unittest.main()
