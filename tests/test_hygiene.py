import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# open(...).read() leaves the file to the garbage collector -> ResourceWarning on Python 3.14
PATTERN = re.compile(r"\bopen\([^)]*\)\.(read|readlines|write)\(")


class TestHygiene(unittest.TestCase):
    def test_no_unclosed_open_calls(self):
        offenders = []
        for folder in ("red_queen", "tests"):
            for name in sorted(os.listdir(os.path.join(ROOT, folder))):
                if name.endswith(".py"):
                    path = os.path.join(ROOT, folder, name)
                    with open(path, encoding="utf-8") as f:
                        for i, line in enumerate(f, 1):
                            if PATTERN.search(line) and not line.lstrip().startswith("#") and "PATTERN" not in line:
                                offenders.append(f"{folder}/{name}:{i}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
