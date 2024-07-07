'''
This is the server package. It contains the server code for the Birbs project.
'''

from .server import start_server
from .col_server_integration import COLServerIntegration

__all__ = ['start_server', 'COLServerIntegration']