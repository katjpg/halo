import logging

FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure logging if the root logger has no handlers."""
    logging.basicConfig(level=level.upper(), format=FORMAT)
