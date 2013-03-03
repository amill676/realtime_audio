__author__ = 'adamjmiller'
import pyaudio


class AudioHelper(object):
    """
    Class for managing pyaudio devices. Provides extra tools
    for choosing, displaying, and managing avialable devices
    in conjunction with pyaudio
    """

    def __init__(self, pa):
        """
        @type pa pyaudio.PyAudio
        @param pa pyaudio.PyAudio object for use in determining devices
        """
        if not isinstance(pa, pyaudio.PyAudio):
            raise ValueError("pa must be a pyaudio.PyAudio object")
        self._pa = pa

    def get_device_names(self):
        """
        Returns a list with the names of the available devices. The
        devices position in the list indicates the device's index

        @returns list withe available device names
        """
        return [self._pa.get_device_info_by_index(i)['name']
                for i in range(self._pa.get_device_count())]

    def get_device_info_by_name(self, name):
        """
        Returns dictionary with device info for the device with
        the given name. Is case insensitive to 'name' input

        @param name name of the desired
        @returns dictionary with device info for the device with the
                given name if such a device exists. None otherwise
        """
        self._pa.get_device_info_by_index(2)
        devices = [self._pa.get_device_info_by_index(i) for i in range(self._pa.get_device_count())]
        for device in devices:
            if device['name'].lower() == name.lower():
                return device

    def display_devices(self):
        """
        Prints the avaiable devices to the console, with each name
        preceded by its associated device index
        """
        for i, name in enumerate(self.get_device_names()):
            print str(i) + ": " + name

    def get_input_device_from_user(self):
        print "Select an input device"
        print "======================"
        self.display_devices()
        chosen = False
        in_device = self.get_default_input_device_info()
        while not chosen:
            try:
                choice = int(raw_input())
                in_device = self.get_device_info_by_index(choice)
                chosen = True
            except IOError as e:
                print "Please enter a valid device index"
            except ValueError as e:
                print "Input must be a device index number"
        return in_device

    def get_device_count(self):
        """
        Wrapper for pyaudio.PyAudio method with same name
        """
        return self._pa.get_device_count()

    def get_default_input_device_info(self):
        """
        Wrapper for pyaudio.PyAudio method with same name
        """
        return self._pa.get_default_input_device_info()

    def get_default_output_device_info(self):
        """
        Wrapper for pyaudio.PyAudio method with same name
        """
        return self._pa.get_default_output_device_info()

    def get_device_info_by_index(self, index):
        """
        Wrapper for pyaudio.PyAudio method with same name
        """
        return self._pa.get_device_info_by_index(index)
