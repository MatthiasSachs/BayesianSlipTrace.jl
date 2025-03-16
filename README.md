# Bayesian Modeling of Slip Traces

This repository contains Bayesian Mixture models for inference on slip trace data using the probabilistic modeling package `Turing.jl`.

When running the code for the first time, open a Julia REPL in the containing folder an execute the following commands:
```julia-repl
] activate .
] instantiate 
```
### Models

1. The file `/example/slip-trace-gaussian-mixture.jl` contains a work flow example of the Gaussian mixture model applied to synthetic data.
2. The file `/example/slip-trace-gaussian-mixture.jl` contains a work flow example of the Gaussian mixture model with an additional uniform mixture compenent (modelling noise observations) applied to synthetic data.

### ToDo:

- [ ] Implement suitable export functions (python -> hdf5 ?) for slip trace annotation data.
- [ ] Implement suitable import functions (hdf5 -> hdf5 ?) for slip trace annotation data.
- [ ] Run `sp_mix` and `sp_uinf_mix` models on data (i.e., replace the synthetic data in `slip-trace-gaussian-mixture.jl` and `slip-trace-uinfl-gaussian-mixture.jl` by real data.
