### Reproducing the experimental evaluations

The code in this repository can be used to reproduce the experimental evaluations seen in the paper

> Albin Heimerson, Johan Ruuskanen. "Extending Microservice Model Validity using Universal Differential Equations", to appear at IFAC WC 2023

The data is collected using the [FedApp sandbox](https://github.com/JohanRuuskanen/FedApp), though we provide recorded data used in the paper under the `data` folder.
The code for training and evaluation the models are run under `julia v1.8.5`.


#### Running the experiments

The cost-minimizing algorithm is implemented in `Julia` and  tested with `Julia v1.8.5`. For the plotting, it further requires `Python` with `matplotlib`. To activate the environment and install dependencies, copy the `ude_model/` folder to the home folder on the gateway and change to that directory. Start `julia` and type

```Julia
] activate .
] instantiate
```

[DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) was used to solve the fluid model, and [Forwardiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) for the automatic differentiation. For the neural networks we use [Flux.jl](https://github.com/FluxML/Flux.jl).


##### Training the model


##### Evaluating the model 

