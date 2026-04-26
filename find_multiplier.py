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

expected_lines = expected.strip().split('\n')

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

base_angles = [120, 60, 0]

print("Checking: angle2 = angle1 + angle + base_angle[i] * M")
print()

for n in range(4, min(22, len(expected_lines))):
    angle, r1, r2, r3, sign = baseTable[n-1]
    exp_line = expected_lines[n]
    if exp_line == '-':
        continue

    exp_values = [float(x.strip()) for x in exp_line.replace(',', ' ').split()]

    angle1 = []
    indices = []
    if r1 != -1:
        angle1.append(sign * 120)
        indices.append(0)
    if r2 != -1:
        angle1.append(sign * 60)
        indices.append(1)
    if r3 != -1:
        angle1.append(sign * 0)
        indices.append(2)

    print(f"n={n}: angle={angle:4}, sign={sign:2}")

    if len(exp_values) == len(angle1):
        for j in range(len(exp_values)):
            i = indices[j]
            base = base_angles[i]

            # Calculate X from: angle2 = angle1 + angle + X
            x = exp_values[j] - angle1[j] - angle
            while x < -180:
                x += 360
            while x > 180:
                x -= 360

            # Calculate multiplier M from: X = base * M
            if base != 0:
                m = x / base
            else:
                m = 'N/A'

            print(f"  [idx={i} base={base:3}]: angle1={angle1[j]:4.0f} angle2={exp_values[j]:4.0f} X={x:6.1f} M={m}")
    print()
