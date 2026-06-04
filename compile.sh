#!/usr/bin/env bash
set -euo pipefail
latexmk -pdf -outdir=build -interaction=nonstopmode main.tex
