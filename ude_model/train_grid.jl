include("autodiff.jl")

function fix_data(fname)
    data = CSV.File(fname)
    CSV.write(fname, (; iter=1:length(data)-1, dt=diff(data.t), loss=data.loss[1:end-1]); writeheader=true, header=[:iter, :dt, :loss])
end

function train_point_and_eval_grid(
    train_p, hist_p; 
    use_nn=true, use_fluid=true, fluid_net=false,
    hidden_units=20, hidden_layers=0, 
    flowconstrained=true, ql_maxiters=30, modelconstrained=false,
    tag="test", seed=37, kwargs...)

    (use_nn || use_fluid) || error("no model given")

    datafolder="logs"

    eps = 1e-8

    logfolder = joinpath(@__DIR__, "..", "data", datafolder)

    basename = "data_$(use_fluid ? "fluid_" : "")$(use_nn ? "nn$(fluid_net ? "fnet" : "$(flowconstrained ? "fc" : modelconstrained ? "mc" : "")_hu$(hidden_units)_hl$(hidden_layers)")_mi$(ql_maxiters)" : "")_p$(train_p)_hist$(hist_p)_$(tag)"

    datafile = joinpath(logfolder, basename * ".csv")

    global timingfile = joinpath(logfolder, basename * "_timingdata.csv")

    CSV.write(timingfile, []; writeheader=true, header=[:t, :loss])

    all_data = [get_data(logfolder, joinpath(logfolder, "traces", "sample$(round(Int, 20*p+1))")) for p in 0:0.05:1]

    hist_data = [([log((p+eps)/(1-p+eps)), 0], d.pop_class_mean, d.p95_data, d.λ) for (p, d) in zip(hist_p, all_data[round.(Int, 20 .* hist_p .+ 1)])]
        
    idx = round(Int, 20 * train_p) + 1
    w0 = [log((train_p+eps)/(1-train_p+eps)), 0]
    @time "Time for training" params = train_on_data(all_data[idx]; w0, hist_data, use_nn, use_fluid, fluid_net, hidden_units, hidden_layers, flowconstrained, modelconstrained, ql_maxiters, seed)

    fix_data(timingfile)

    CSV.write(datafile, []; writeheader=true, header=[:p1, :ql_i, :p95_i, :cost_i, :ql_t, :p95_t, :cost_t])

    for p in 0.0:0.05:1.0
        eps = 1e-6
        pw = [log((p+eps)/(1-p+eps)), 0]

        # Solve using lambda for each individual point
        update_params!(params, pw, all_data[round(Int, 20*p+1)].λ)

        sol = if use_fluid && use_nn
            solve_both(params)
        elseif use_fluid
            solve_fluid(params)
        else
            solve_nn(params)
        end

        xf = [sum(sol.u[end][params.M .== i], dims=1)[1] for i in 1:length(params.K)]
        p95_i = p95_pred(params; xf)
        cost_i = costfunc_ql(pw, params) # Use predicted p95
        ql_i = sum(xf)

        # Solve using lambda for training point
        update_params!(params, pw, all_data[round(Int, 20*train_p+1)].λ)

        sol = if use_fluid && use_nn
            solve_both(params)
        elseif use_fluid
            solve_fluid(params)
        else
            solve_nn(params)
        end
        xf = [sum(sol.u[end][params.M .== i], dims=1)[1] for i in 1:length(params.K)]
        p95_t = p95_pred(params; xf)
        cost_t = costfunc_ql(pw, params) # Use predicted p95
        ql_t = sum(xf)

        CSV.write(
            datafile,
            (
                p1=[p],
                ql_i=[ql_i],
                p95_i=[p95_i],
                cost_i=[cost_i],
                ql_t=[ql_t],
                p95_t=[p95_t],
                cost_t=[cost_t],
            ),
            append=true,
        )
    end
    @info basename
    params
end
