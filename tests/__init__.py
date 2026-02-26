import warnings
import logging

# Suppress pervasive library-level warnings during test execution
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Disable verbose logging from underlying network/API libraries
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
