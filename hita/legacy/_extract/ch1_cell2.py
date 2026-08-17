# Two-stage dataset design:
# 1) Before "that's not realistic": linearly separable points.
# 2) After that line: add symmetric noisy students to break separability.

separable_points = [
    # Fail class (left of boundary, z=study-exam < 0)
    (2, 3, 0),  # required fail example
    (4, 5, 0), (5, 6, 0),
    (1, 3, 0), (2, 4, 0), (4, 6, 0),
    (1, 4, 0), (3, 6, 0), (1, 6, 0),


    # Pass class (right of boundary, z > 0)
    (3, 2, 1),  # required pass example
    (5, 4, 1), (6, 5, 1),                    # z=+1 -> 3 pass
    (4, 2, 1), (6, 4, 1), (3, 1, 1),  # z=+2 -> 4 pass
    (4, 1, 1), (5, 2, 1), (6, 3, 1),  # z=+3 -> 4 pass
    (6, 2, 1),  # z=+4 (Scene 6b anchor)
    (6, 1, 1),
]

# Added noisy students (symmetric around z=0, on parallel lines)
# to break strict linear separability after "but that's not realistic".
noisy_symmetric_points = [
    # symmetric pairs on +/-1
    (2, 1, 0),  # z=+1 but fails
    (1, 2, 1),  # z=-1 but passes
    (3, 4, 1),  # z=-2 but fails
    (4, 3, 0),  # z=+2 but passes

    # symmetric pair on +/-2
    (3, 5, 1),  # z=+2 but passes
    (5, 3, 0),  # z=-2 but fails
]

realistic_points = separable_points + noisy_symmetric_points


def unpack_points(point_list):
    arr = np.array(point_list, dtype=float)
    s = arr[:, 0]
    e = arr[:, 1]
    labels = arr[:, 2].astype(int)
    diff = s - e
    return s, e, labels, diff


study_sep, exam_sep, y_sep, z_sep = unpack_points(separable_points)
study_real, exam_real, y_real, z_real = unpack_points(realistic_points)

xlim = (0, 7)
ylim = (0, 7)

midpoint_shift = (z_sep[y_sep == 0].max() + z_sep[y_sep == 1].min()) / 2.0

print(f"Separable students: {len(separable_points)}")
print(f"Realistic students: {len(realistic_points)}")
print("Required pass example present:", np.any((study_sep == 3) & (exam_sep == 2) & (y_sep == 1)))
print("Required fail example present:", np.any((study_sep == 2) & (exam_sep == 3) & (y_sep == 0)))
print("Threshold midpoint between classes (separable stage):", midpoint_shift)
