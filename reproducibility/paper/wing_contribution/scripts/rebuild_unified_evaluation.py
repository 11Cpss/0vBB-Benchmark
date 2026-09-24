#!/usr/bin/env python3
"""Rebuild final paper artifacts; see ../evaluation/README.md for event evaluation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
from rebuild import main
if __name__=='__main__':main()
