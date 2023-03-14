using PyPlot, CSV

include("train_grid.jl")

p_train = 0.6

params = train_point_and_eval_grid(p_train, 0.1:0.2:1.0; ql_maxiters=500, use_fluid=true, use_nn=true, modelconstrained=true, hidden_units=20, hidden_layers=2, tag="test_opt2", seed=1)

losses = []
pss = []
function callback2(ps, l)
    push!(losses, l)
    push!(pss, softmax(ps)[1])
    @info ps l
    false
end

optf = OptimizationFunction(costfunc_ql, Optimization.AutoForwardDiff())

pw = [log(p_train/(1-p_train)), 0]
optprob1 = OptimizationProblem(optf, pw, params)
optsol1 = solve(optprob1, ADAM(0.1), maxiters=10, callback=callback2)

#optprob2 = OptimizationProblem(optf, optsol1.u, params)
#optsol2 = solve(optprob2, BFGS(), maxiters=10, callback=callback2)

p_target = softmax(optsol2)[1]

# Create a predict
ps = 0:0.05:1
pred = [
    begin
        eps = 1e-6
        pw = [log((p+eps)/(1-p+eps)), 0]

        # Solve using lambda for training point
        update_params!(params, pw, params.λ)

        sol = solve_both(params)

        xf = [sum(sol.u[end][params.M .== i], dims=1)[1] for i in 1:length(params.K)]
        p95 = p95_pred(params; xf)
        cost = costfunc_ql(pw, params) # Use predicted p95
        ql = sum(xf)
        (; p95, ql, cost)
    end
    for p in ps
]


basepath = joinpath(@__DIR__, "..", "data")
runid = "logs"

datatag = "fluid_only_retrain"
realdata = CSV.File(joinpath(basepath, runid, "data_$(datatag).csv"))

figure(32)
clf()
subplot(1, 3, 1)
plot(ps, map(x -> x.ql, pred), label="ql")
plot(ps, realdata.ql_data, label="ql")
subplot(1, 3, 2)
plot(ps, map(x -> x.p95, pred), label="p95")
plot(ps, realdata.p95_data, label="ql")
subplot(1, 3, 3)
plot(ps, map(x -> x.cost, pred), label="cost")
plot(ps, realdata.cost_data, label="ql")
plot(pss, losses, label="ql", marker="+")

gcf()


CSV.write(
    joinpath(basepath, runid, "optdata_p_200iter.csv"),
    (
        p = pss,
        cost = losses,
    );
    writeheader = true,
    header = [:p, :cost]
)

CSV.write(
    joinpath(basepath, runid, "data_fluid_nnfc_hu20_hl0_mi200_optrun.csv"),
    (
        p = ps,
        ql = map(x -> x.ql, pred),
        p95 = map(x -> x.p95, pred),
        cost = map(x -> x.cost, pred),
    );
    writeheader = true,
    header = [:p, :ql, :p95, :cost]
)
