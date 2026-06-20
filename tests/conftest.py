"""Shared test configuration.

Ensures torch.serialization.add_safe_globals is called before any module
import, preventing e3nn/checkpoint loading errors on PyTorch >= 2.6.
"""
import torch

torch.serialization.add_safe_globals([slice])
