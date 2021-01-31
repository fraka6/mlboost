#!/usr/bin/env python
"""
Basic script to simulate user typing
Examples
echo "you are smart" | python type.py

"""
import os, sys
import time
for line in sys.stdin.readlines():
    for char in line:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.1)

