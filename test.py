import pytest
import sys
import os

# Uvozi teste iz obstoječe datoteke
sys.path.append(os.path.join(os.getcwd(), '.tests'))
from test_naloga2 import test_konvolucija, test_filtriraj_z_gaussovim_jedrom, test_filtriraj_sobel

# Funkcije so že uvožene, zato bodo avtomatsko zaznane s pytest
