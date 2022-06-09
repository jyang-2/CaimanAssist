import pydantic
import typing
from pathlib import Path


class CaimanMocoMetrics(pydantic.BaseModel):
    """Class for metrics.json file in caiman_mc folder"""

    bord_px_rig: typing.List[int]
    bord_px_els: typing.List[int]
    bad_planes_rig: typing.Optional[typing.List[int]]
    bad_planes_els: typing.Optional[typing.List[int]]
    which_movie: typing.Literal['m_els', 'm_rig']
