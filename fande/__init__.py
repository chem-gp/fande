import fande.models

import logging

logging.basicConfig(filename='fande.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('Fande module imported')


__version__ = "0.0.0"

__all__ = [
    "modules",
    "__version__",
]
