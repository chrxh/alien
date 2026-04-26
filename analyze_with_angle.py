expected = """
-
-
-
-
60.000000, 60.000000, 60.000000
300.000000
60.000000, 120.000000
300.000000
240.000000
-
300.000000, 180.000000, 240.000000
60.000000
300.000000, 180.000000
60.000000
300.000000
240.000000, 120.000000, 300.000000
60.000000
300.000000, 240.000000
180.000000, 300.000000
60.000000
120.000000, 60.000000
120.000000
-
60.000000, 180.000000, 120.000000
300.000000
60.000000, 180.000000, 60.000000
300.000000
"""

# Expected angle2 values
expected_lines = expected.strip().split('\n')

# BaseTable data (angle, r1, r2, r3, sign)
baseTable = [
    (60, -1, -1, -1, 0),     # n=1
    (60, -1, -1, -1, 0),     # n=2
    (120, -1, -1, -1, 0),    # n=3
    (-120, 2, 1, 0, 1),      # n=4
    (120, 3, -1, -1, -1),    # n=5
    (-120, 4, 0, -1, 1),     # n=6
    (-60, 5, -1, -1, -1),    # n=7
    (0, 5, -1, -1, -1),      # n=8
    (-120, -1, -1, -1, 0),   # n=9
    (120, 8, 5, 3, -1),      # n=10
    (-120, 9, -1, -1, 1),    # n=11
    (120, 10, 3, -1, -1),    # n=12
    (-120, 11, -1, -1, 1),   # n=13
    (-60, 12, -1, -1, -1),   # n=14
    (120, 12, 3, 2, -1),     # n=15
    (-120, 14, -1, -1, 1),   # n=16
    (0, 15, 2, -1, -1),      # n=17
    (120, 2, 1, -1, -1),     # n=18
    (60, 17, -1, -1, 1),     # n=19
    (0, 17, 16, -1, 1),      # n=20
    (0, 16, -1, -1, 1),      # n=21
]

print("Looking for pattern: angle2 = angle1 + angle + X")
print()

for n in range(4, min(22, len(expected_lines))):
    angle, r1, r2, r3, sign = baseTable[n-1]
    exp_line = expected_lines[n]
    if exp_line == '-':
        continue

    # Parse expected values
    exp_values = [float(x.strip()) for x in exp_line.replace(',', ' ').split()]

    # Calculate angle1 values
    angle1 = []
    angle1_base = []
    if r1 != -1:
        angle1.append(sign * 120)
        angle1_base.append(120)
    if r2 != -1:
        angle1.append(sign * 60)
        angle1_base.append(60)
    if r3 != -1:
        angle1.append(sign * 0)
        angle1_base.append(0)

    print(f"n={n}: angle={angle:4}, sign={sign:2}")

    if len(exp_values) == len(angle1):
        for i in range(len(exp_values)):
            # Check: angle2 = angle1 + angle + X
            x = exp_values[i] - angle1[i] - angle
            while x < -180:
                x += 360
            while x > 180:
                x -= 360

            print(f"  [{i}] base={angle1_base[i]:3}: angle1={angle1[i]:4.0f} angle2={exp_values[i]:4.0f} -> X (from angle1+angle+X) = {x:6.1f}")
    print()
