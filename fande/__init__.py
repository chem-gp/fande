import fande.models

import logging

logging.basicConfig(filename='fande.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info('Fande module imported')



try:
    from icecream import install
    install()
except ImportError:
    print("Icecream logger is not available")


__version__ = "0.0.0"

__all__ = [
    "modules",
    "__version__",
]
