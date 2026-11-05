import unittest

from red_queen.validation import FULL_ONLY, build_checks, select_checks


class TestQuickSplit(unittest.TestCase):
    def test_full_only_ids_exist(self):
        ids = {label.split()[0] for label, _ in build_checks()}
        self.assertEqual(FULL_ONLY - ids, set())

    def test_split_is_a_partition_in_order(self):
        full = [label for label, _ in select_checks(quick=False)]
        quick = [label for label, _ in select_checks(quick=True)]
        self.assertEqual(full, [label for label, _ in build_checks()])
        self.assertEqual(len(full), 211)
        self.assertEqual(len(quick), 211 - len(FULL_ONLY))
        self.assertEqual(quick, [label for label in full if label.split()[0] not in FULL_ONLY])

    def test_quick_still_covers_every_milestone(self):
        quick_ids = {int(label.split()[0][1:]) for label, _ in select_checks(quick=True)}
        groups = {"M1": (1, 20), "M1.1": (21, 31), "M2": (32, 42), "M2.1": (43, 50), "M3": (51, 63),
                  "M3.2": (64, 73), "M3.3": (74, 82), "M3.4": (83, 90), "M3.5": (91, 100), "M3.6": (101, 108),
                  "M4": (109, 117), "M4.1": (118, 124), "M4.2": (125, 131), "M5": (132, 141), "M5.1": (142, 148),
                  "M5.2": (149, 155), "M5.3": (156, 162), "M5.4": (163, 169), "M6": (170, 176), "M7": (177, 183),
                  "M7.1": (184, 190), "M7.2": (191, 197), "M7.3": (198, 204), "M8": (205, 211)}
        missing = [m for m, (a, b) in groups.items() if not any(a <= i <= b for i in quick_ids)]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
