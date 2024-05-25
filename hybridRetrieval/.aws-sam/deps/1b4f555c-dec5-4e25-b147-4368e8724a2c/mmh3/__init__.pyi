# to use list, tuple, dict ... in Python 3.7 and 3.8
from __future__ import annotations

from typing import Protocol, Union, final

class IntArrayLike(Protocol):
    def __getitem__(self, index) -> int: ...

Hashable = Union[bytes, bytearray, memoryview, IntArrayLike]
StrHashable = Union[str, Hashable]

def hash(key: StrHashable, seed: int = 0, signed: bool = True) -> int: ...
def hash_from_buffer(key: StrHashable, seed: int = 0, signed: bool = True) -> int: ...
def hash64(key: StrHashable, seed: int = 0, x64arch:bool = True, signed: bool = True) -> tuple[int, int]: ...
def hash128(key: StrHashable, seed: int = 0, x64arch:bool = True, signed: bool = False) -> int: ...
def hash_bytes(key: StrHashable, seed: int = 0, x64arch: bool = True) -> bytes: ...

class Hasher:
    def __init__(self, seed: int = 0) -> None: ...
    def update(self, input: Hashable) -> None: ...
    def digest(self) -> bytes: ...
    def sintdigest(self) -> int: ...
    def uintdigest(self) -> int: ...
    def copy(self) -> Hasher: ...
    @property
    def digest_size(self) -> int: ...
    @property
    def block_size(self) -> int: ...
    @property
    def name(self) -> str: ...

@final
class mmh3_32(Hasher): ...

@final
class mmh3_x64_128(Hasher):
    def stupledigest(self) -> tuple[int, int]: ...
    def utupledigest(self) -> tuple[int, int]: ...

@final
class mmh3_x86_128(Hasher):
    def stupledigest(self) -> tuple[int, int]: ...
    def utupledigest(self) -> tuple[int, int]: ...
