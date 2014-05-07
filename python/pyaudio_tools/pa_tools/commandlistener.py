import threading

class CommandListener(object):
  """
  Class for listening to and processing input given on the terminal.
  Will be listening for input from user, and will maintaing state
  information that is pollable by client code
  """
  def __init__(self):
    self._quit = False
    self._save_fig = False
    self._switch_beam = False
    self._poll_thread = threading.Thread(target=self._poll)

  def start_polling(self):
    self._poll_thread.start()

  def _poll(self):
    while True:
      read_in = raw_input()
      if read_in == "q":
        print "User has chosen to quit."
        self._quit = True
        break
      if read_in == "b":
        self._switch_beam = True
      if read_in == "s":
        self._save_fig = True
  
  def quit(self):
    # Don't reset quit
    return self._quit

  def savefig(self):
    return self._get_reset_flag('_save_fig')

  def switch_beamforming(self):
    return self._get_reset_flag('_switch_beam')

  def set_quit(self, val):
    self._quit = val

  def _get_reset_flag(self, flag):
    if self.__getattribute__(flag):
      self.__dict__[flag] = False
      return True
    return False
