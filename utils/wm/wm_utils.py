from enum import Enum

from utils.wm.gs_provider import GsProvider
from utils.wm.tr_provider import TrProvider


class WmProviders(Enum):
    GS = GsProvider
    TR = TrProvider
