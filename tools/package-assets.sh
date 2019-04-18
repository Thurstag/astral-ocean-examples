#!/bin/bash

# Copy data folder
rm -dRf assets_compiled
cp -r assets assets_compiled
echo "[INFO] Copy assets folder (assets_compiled)"

# Find shaders
files=$(find assets_compiled/ -iname *.frag -o -iname *.vert -o -iname *.comp)

# Compile shaders
for file in $files; do
	dir=$(dirname $file)
	filename=$(basename -- "$file")
	ext="${filename##*.}"

	glslangValidator -V "$file" -o "$dir/$ext.spv"
done
echo "[INFO] Compile shaders"

# Remove source files
for file in $files; do
	rm $file
done
echo "[INFO] Remove source files"
