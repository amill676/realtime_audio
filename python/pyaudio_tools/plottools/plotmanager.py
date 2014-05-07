import os

class PlotManager(object):
  """
  Class for managing matplotlib plots. Takes care of sizing and settings
  and provides ability to save figures 
  """
  def __init__(self, outfile_basename='plot', outfile_directory='figures'):
    self._file_base_name = outfile_basename
    self._directory = outfile_directory
    self._out_file_number = 0 # For counting to avoid replacing figures
    self._setup_savefig_settings()

  def savefig(self, fig):
    """
    Save figure to an image file. This will use the settings supplied at
    instantiation to name and locate the output file
    :param fig: matplotlib figure whose image will be saved
    """
    filename = self._get_fig_file_name()
    fig.savefig(filename, 
      facecolor=self._savefig_facecolor, 
      edgecolor=self._savefig_edgecolor
    )
    print "Figure saved to %s" % filename

  def _get_fig_file_name(self):
    if not os.path.exists('figures'):
        os.makedirs(self._directory)
    filename = self._directory + '/' + self._file_base_name + \
            str(self._out_file_number) + '.png'
    while os.path.exists(filename):
        self._out_file_number += 1
        filename = self._directory + '/' + self._file_base_name + \
                str(self._out_file_number) + '.png'
    return filename

  def _setup_savefig_settings(self):
    self._savefig_facecolor = 'white'
    self._savefig_edgecolor = 'none'
