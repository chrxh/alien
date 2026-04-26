expected_str = """60.000000, 60.000000, 60.000000
300.000000
60.000000, 120.000000
300.000000
240.000000"""

baseTable = [
    (-120, 2, 1, 0, 1),      # n=4
    (120, 3, -1, -1, -1),    # n=5
    (-120, 4, 0, -1, 1),     # n=6
    (-60, 5, -1, -1, -1),    # n=7
    (0, 5, -1, -1, -1),      # n=8
]

lines = expected_str.split('\n')

print("Detailed analysis for n=4 through n=8:\n")

for i, line in enumerate(lines):
    n = i + 4
    angle, r1, r2, r3, sign = baseTable[i]
    exp_values = [float(x.strip()) for x in line.replace(',', ' ').split()]

    angle1 = []
    bases = []
    if r1 != -1:
        angle1.append(sign * 120)
        bases.append(120)
    if r2 != -1:
        angle1.append(sign * 60)
        bases.append(60)
    if r3 != -1:
        angle1.append(sign * 0)
        bases.append(0)

    print(f"n={n}: result.angle={angle:4}, angleSign={sign:2}")
    print(f"  angle1 values: {angle1}")
    print(f"  expected angle2: {exp_values}")

    # For each angle, show all tried formulas
    for j in range(len(exp_values)):
        print(f"  [{j}] angle1={angle1[j]:6.1f} -> angle2={exp_values[j]:6.1f}")
        print(f"      180 - angle1 = {(180 - angle1[j]) % 360:6.1f}")
        print(f"      angle - angle1 = {(angle - angle1[j]) % 360:6.1f}")
        print(f"      angle + angle1 = {(angle + angle1[j]) % 360:6.1f}")
        print(f"      -angle1 = {(-angle1[j]) % 360:6.1f}")
    print()
