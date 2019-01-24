#!/bin/bash

# Format files
echo "[INFO] Format files"
find src/ -iname *.h -o -iname *.hpp -o -iname *.cpp | xargs clang-format -i
