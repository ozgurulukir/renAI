with open("tests/test_file_processor.py", "r") as f:
    content = f.read()

import re
# Clean up duplicate imports at bottom
content = re.sub(r'import sys\nfrom unittest\.mock import MagicMock\n\nsys\.modules\[\'PIL\'\] = MagicMock\(\)\nsys\.modules\[\'PIL\.Image\'\] = MagicMock\(\)\nsys\.modules\[\'PIL\.ImageEnhance\'\] = MagicMock\(\)\nsys\.modules\[\'PIL\.ImageFilter\'\] = MagicMock\(\)\nsys\.modules\[\'PIL\.ImageOps\'\] = MagicMock\(\)', '', content)

with open("tests/test_file_processor.py", "w") as f:
    f.write(content)
