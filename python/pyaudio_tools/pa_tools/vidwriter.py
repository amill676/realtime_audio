import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

class VidWriter(object):
  """
  Class for writing videos. Encapsulates the opencv interface, which on OS X
  is incredibly precarioius/annoying
  """
  default_fps = 25

  def __init__(self, outfile, width, height, fps=default_fps, fourcc=None):
    self.open(outfile, width, height, fps, fourcc)


  def _process_inputs(self, outfile, width, height, fps, fourcc):
    self._outfile = outfile
    self._width = width
    self._height = height
    self._fps = self._process_fps(fps)
    self._fourcc = self._process_fourcc(fourcc)

  def _process_fps(self, fps):
    valid_fps = [16, 20, 25, 30]
    fps = int(fps)
    if fps not in valid_fps:
      print "WARNING: %d is not among the tested fps values. May cause problems" % fps
    return fps


  def _process_fourcc(self, fourcc):
    default_fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
    if fourcc is None:
      return default_fourcc
    return fourcc

  def _setup_writer(self):
    return cv2.VideoWriter(self._outfile, self._fourcc, self._fps, (self._width, self._height))

  def write_image(self, image):
    """
    Write image to video file. 
    
    NOTE that the image should be in 
    (height, width) format. This is the standard orientation for
    an image, however it is the reverse of what will be returned
    by VideoCapture.read(). BE WARNED

    :param image: numpy array image. Should be 3 channels, with 'uint8' dtype
    """
    if image.shape[2] != 3:
      raise ValueError("Input image must have 3 dimensions -- one for each color")
    if image.shape[0] != self._height or image.shape[1] != self._width:
      raise ValueError("Input image must be has shape (height, width, 3)")
    if image.dtype != np.uint8:
      raise ValueError("Input image must have dtype uint8")
    self._writer.write(image)

  def write_file(self, filename):
    """
    Write a frame using a matrix stored as a file in string format.
    This file should have been written in format as from
    fig.canvas.print_rgba()
    :param filename: name of matrix file
    """
    with open(filename, 'r') as f:
      buf = f.read()
      mat = np.fromstring(buf, dtype=np.uint8)
      im = np.reshape(mat, (self._height, self._width, 4))
      self.write_image(im[:, :, :3])

  def _fig_to_array(self, fig):
    """
    Convert matplotlib figure to an image (numpy array)
    """
    fig.canvas.draw()  # Updates
    canvas = FigureCanvasAgg(fig)
    buf = canvas.tostring_rgb()
    w, h = canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(h, w, 3)

  def write_figure(self, figure):
    """
    Write out a matplotlib figure. Make sure this video writer was initialized
    with the correct dimensions
    """
    self.write_image(self._fig_to_array(figure))

  def open(self, outfile, width, height, fps=default_fps, fourcc=None):
    """
    Open a file and prepare for writing
    """
    self._process_inputs(outfile, width, height, fps, fourcc)
    self._writer = self._setup_writer()

  def close(self):
    """
    Will release the underlying VideoWriter object, saving the file
    """
    self._writer.release()
    self._writer = None
