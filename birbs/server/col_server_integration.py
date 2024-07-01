# Third party imports


# Custom imports
from birbs.col_filtering import COL
import birbs.server.server as server
import json

def get_my_peer_info():
    '''
    This function is for testing purposes.
    '''
    response = server.get_peers()
    peers = json.loads(response.data)
    my_peer_ip = peers["peers"][0]["IP"]
    my_peer_port = peers["peers"][0]["Port"]
    my_peer_hash = peers["peers"][0]["Hash"]

    col = COL(my_peer_ip, my_peer_port, my_peer_hash)
    col.start()