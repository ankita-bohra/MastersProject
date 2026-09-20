"""Reusable final RNQI arithmetic."""


def calculate_rnqi(connectivity, efficiency, hierarchy):
    """Calculate the RNQI equal-weight composite used by production."""
    return (float(connectivity) + float(efficiency) + float(hierarchy)) / 3.0
