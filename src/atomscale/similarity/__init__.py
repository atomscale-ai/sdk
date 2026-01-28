from .provider import SimilarityTrajectoryProvider

__all__ = [
    "SimilarityTrajectoryProvider",
    "aiter_poll_trajectory",
    "iter_poll_trajectory",
    "start_polling_trajectory_task",
    "start_polling_trajectory_thread",
]


def __getattr__(name: str):
    """Lazy import polling functions to avoid circular imports."""
    if name in (
        "aiter_poll_trajectory",
        "iter_poll_trajectory",
        "start_polling_trajectory_task",
        "start_polling_trajectory_thread",
    ):
        from . import polling

        return getattr(polling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
