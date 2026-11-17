# DNA Comparison Load Balancing Using OpenMP

This repository contains eight C++ sequence-alignment implementations, their
input datasets and recorded results, and a plot comparing the implementations'
runtime.

## Team

| Name | Registration number |
|---|---:|
| Nivas Ravate | 241080058 |
| Siddharth Pandey | 241080048 |

## Runtime comparison

The chart below summarizes the five runs recorded for each implementation.

[![Runtime comparison of all 8 implementations](https://siddharthpandey2211.github.io/dna-comparison-load-balancing-using-openmp/plot/runtime_comparison.png)](https://siddharthpandey2211.github.io/dna-comparison-load-balancing-using-openmp/)

The interactive project page is published with GitHub Pages.

## Project contents

- C++ implementations and the shared Needleman-Wunsch header
- Input datasets in `data/`
- Recorded benchmark results in `results/`
- Chart, source data, and plotting script in `plot/`

The benchmark dataset `data/chromosome21.fasta` identifies itself as the
GRCh38.p14 chromosome 21 reference assembly.
