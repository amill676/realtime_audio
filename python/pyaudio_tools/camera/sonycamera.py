__author__ = 'Adam Miller'
import urllib


class SonyCamera(object):
    """
    Class for interacting with Sony
    """

    def __init__(self, url):
        """
        :param url: URL for connecting to camera
        """
        self._url = url
        self._loc =

    def connect(self):
        """
        Attempt to connect to specified URL
        :returns: True if connection succeeds, False otherwise
        """
        try:
            self._conn = urllib.urlopen(self._url)
        except IOError:
            return False
        return True

    def turn_around(self):





