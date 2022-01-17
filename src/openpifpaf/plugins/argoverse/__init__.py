import openpifpaf
from . import argoverse_plugin
def register():
    openpifpaf.DATAMODULES['argoverse'] = argoverse_plugin.ArgoversePlugin
