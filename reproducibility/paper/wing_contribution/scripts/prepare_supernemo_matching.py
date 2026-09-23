#!/usr/bin/env python3
"""Validate reviewed illustration data, or prepare new data from an external event NPZ."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
from illustration import prepare_main
if __name__=='__main__':prepare_main()
