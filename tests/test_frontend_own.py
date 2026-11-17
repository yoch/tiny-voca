"""Front-end selection tests for ``tiny_voca.text``."""

import unittest

from tiny_voca import text
from tiny_voca.text import frontend_fingerprint


class FrontEndSwitchTests(unittest.TestCase):
    def tearDown(self):
        text.set_frontend("espeak")

    def test_espeak_stays_the_default_and_keeps_its_identity(self):
        text.set_frontend("espeak")
        self.assertEqual(text.frontend(), "espeak")
        self.assertEqual(text.frontend_identity()["frontend_version"], 4)
        self.assertEqual(text.frontend_identity()["voice"], "fr")
        self.assertEqual(text.frontend_fingerprint(), frontend_fingerprint())
        with self.assertRaises(ValueError):
            text.set_frontend("festival")


if __name__ == "__main__":
    unittest.main()
