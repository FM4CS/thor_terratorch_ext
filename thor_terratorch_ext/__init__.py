import logging

logger = logging.getLogger(__name__)
try:
    import thor  # noqa: F401

except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"ImportError: {e}")
    error_msg = (
        "THOR package not found. Please install it from https://github.com/fm4cs/thor"
    )
    raise ImportError(error_msg)

from .datasets import utils  # noqa: E402
from .models import thor_vit  # noqa: E402

__all__ = ["utils", "thor_vit"]
