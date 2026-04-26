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

print("Pattern analysis by connection index:")
print()

# Collect patterns by connection index
patterns_by_index = {0: [], 1: [], 2: []}

for n in range(4, min(22, len(expected_lines))):
    angle, r1, r2, r3, sign = baseTable[n-1]
    exp_line = expected_lines[n]
    if exp_line == '-':
        continue

    # Parse expected values
    exp_values = [float(x.strip()) for x in exp_line.replace(',', ' ').split()]

    # Calculate angle1 values (for requiredNodeId[0], [1], [2])
    angle1 = []
    angle1_base = []  # The base values (120, 60, 0)
    if r1 != -1:
        angle1.append(sign * 120)
        angle1_base.append(120)
    if r2 != -1:
        angle1.append(sign * 60)
        angle1_base.append(60)
    if r3 != -1:
        angle1.append(sign * 0)
        angle1_base.append(0)

    if len(exp_values) == len(angle1):
        for i in range(len(exp_values)):
            diff = exp_values[i] - angle1[i]
            while diff < -180:
                diff += 360
            while diff > 180:
                diff -= 360

            # Store pattern info
            patterns_by_index[i].append({
                'n': n,
                'angle': angle,
                'sign': sign,
                'angle1_base': angle1_base[i],
                'angle1': angle1[i],
                'angle2': exp_values[i],
                'diff': diff
            })

# Analyze patterns
for idx in [0, 1, 2]:
    print(f"\n=== Connection index {idx} ===")
    patterns = patterns_by_index[idx]
    if not patterns:
        continue

    for p in patterns:
        # Try: angle2 = angle1 + sign * X
        if p['sign'] != 0:
            x = p['diff'] / p['sign']
        else:
            x = 'N/A'

        print(f"n={p['n']:2} angle={p['angle']:4} sign={p['sign']:2} base={p['angle1_base']:3} angle1={p['angle1']:4.0f} angle2={p['angle2']:4.0f} diff={p['diff']:6.1f} diff/sign={x}")
