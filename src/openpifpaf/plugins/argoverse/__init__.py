import openpifpaf
from . import argoverse_plugin

def register():
    openpifpaf.DATAMODULES['bezier'] = argoverse_plugin.ArgoversePlugin
