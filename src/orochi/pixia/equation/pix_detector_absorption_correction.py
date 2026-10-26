"""Legacy compatibility shim for orochi.pixia.equation.detector_absorption."""

from .detector_absorption import (
    absorption_correction,
    detector_absorption,
    absorption,
)

__all__ = ["absorption_correction", "detector_absorption", "absorption"]