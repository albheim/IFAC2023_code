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

The file `run_train.jl` can be called to train the model with varying configurations, e.g. from the `ude_model` folder call
```bash
julia --project run_train.jl --use_fluid --use_nn --mc --hl=2 --hu=20 --mi=500 --tag=test --reps=5
```
to run an experiment using the fluid model as base and extending it with the model constrained NN where the NN has 2 extra hidden layers in addition to the required 1, each hidden layer has 20 units and we run it for 500 training iterations over 5 repeated runs.

##### Evaluating the model 

Using the scripts `plotting.jl` and `plotting_series.jl` we can visualize the recorded data by inserting the filenames without the `data_` in the start or the `.csv` ending into the `tags` array.

They have been used interactively in the REPL, and for running as scripts some modifications would be needed.

We can also try to run the controller using a trained model to see how it will optimize the control parameters, and is done by running `run_opt.jl`. To change parameters of the run for this script you will have to edit the code, specifically the call to `train_point_and_eval_grid` in the start decides many of the interesting parameters.