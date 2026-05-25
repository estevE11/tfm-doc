#!/usr/bin/env bash
set -euo pipefail
latexmk -pdf -outdir=latex_build -interaction=nonstopmode main.tex
