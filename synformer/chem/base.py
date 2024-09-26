import abc
from typing import Literal, overload

import PIL.Image


class Drawable(abc.ABC):
    @overload
    def draw(self, size: int, svg: Literal[False]) -> PIL.Image.Image:
        ...

    @overload
    def draw(self, size: int, svg: Literal[True]) -> str:
        ...

    @abc.abstractmethod
    def draw(self, size: int, svg: bool) -> PIL.Image.Image | str:
        ...
