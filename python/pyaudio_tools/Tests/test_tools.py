__author__ = 'Adam Miller'

import unittest

class AudioTest(unittest.TestCase):

    def setUp(self):
        pass

    def assertListFloatEqual(self, list1, list2):
        if not len(list1) == len(list2):
            raise AssertionError("Lists differ in lenght. Cannot be equal")
        for i in range(len(list1)):
            try:
                self.assertLessEqual(abs(list1[i] - list2[i]), 1e-4)
            except AssertionError:
                err_str = "Lists differ on element" + str(i) + ": " + \
                    str(list1[i]) + " vs. " + str(list2[i])
                raise AssertionError(err_str)

    def tearDown(self):
        pass
