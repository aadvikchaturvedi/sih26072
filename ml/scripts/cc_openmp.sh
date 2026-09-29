#!/bin/bash
# Compiler wrapper for building pysteps on macOS: Apple clang lacks -fopenmp, so map it to Homebrew libomp.
args=()
for a in "$@"; do
  if [ "$a" = "-fopenmp" ]; then args+=("-Xpreprocessor" "-fopenmp" "-I/opt/homebrew/opt/libomp/include" "-L/opt/homebrew/opt/libomp/lib" "-lomp"); else args+=("$a"); fi
done
exec /usr/bin/clang "${args[@]}"
