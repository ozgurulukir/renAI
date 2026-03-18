import sys
from unittest.mock import MagicMock
sys.modules['pypdf'] = MagicMock()
sys.modules['pypdf.errors'] = MagicMock()
