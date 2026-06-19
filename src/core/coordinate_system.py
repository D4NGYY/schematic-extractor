from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec2:
    """Vettore 2D immutabile."""

    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)

    def distance_to(self, other: Vec2) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def normalized(self) -> Vec2:
        length = self.length()
        if length < 1e-12:
            return Vec2(0.0, 0.0)
        return self / length


class CoordinateSystem:
    """Converte coordinate tra sistemi comuni in EDA.

    KiCad usa "mils" (1/1000 inch) internamente, ma .kicad_sch esporta in mm.
    PDF usa "points" (1/72 inch).

    Conversioni supportate:
    - mm <-> mils (1 mil = 0.0254 mm)
    - mm <-> points (1 point = 1/72 inch = 0.352777... mm)
    - mils <-> points
    """

    MM_PER_MIL = 0.0254
    MM_PER_POINT = 25.4 / 72.0  # ≈ 0.352777...
    POINTS_PER_MM = 72.0 / 25.4
    Mils_PER_MM = 1 / 0.0254

    @classmethod
    def mm_to_mils(cls, mm: float) -> float:
        return mm / cls.MM_PER_MIL

    @classmethod
    def mils_to_mm(cls, mils: float) -> float:
        return mils * cls.MM_PER_MIL

    @classmethod
    def mm_to_points(cls, mm: float) -> float:
        return mm * cls.POINTS_PER_MM

    @classmethod
    def points_to_mm(cls, points: float) -> float:
        return points * cls.MM_PER_POINT

    @classmethod
    def mils_to_points(cls, mils: float) -> float:
        return cls.mm_to_points(cls.mils_to_mm(mils))

    @classmethod
    def points_to_mils(cls, points: float) -> float:
        return cls.mm_to_mils(cls.points_to_mm(points))
