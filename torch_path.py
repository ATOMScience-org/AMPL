#!/usr/bin/env python3
from pathlib import Path
import sys
import os
import shutil
import torch

p = Path(torch.__file__)
sys.stdout.write(str(p.parent))






