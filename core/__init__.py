import warnings

# Globally suppress ResourceWarnings from multiprocessing and other low-level libraries
# This is necessary for clean CLI and test output across different Python versions (esp. 3.9 on macOS)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
