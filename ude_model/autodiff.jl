## ===== Loading common packages and functions =====
using ForwardDiff
using Flux, DiffEqSensitivity
using Optimization, OptimizationFlux, OptimizationOptimJL
using ComponentArrays
# TODO fix EMpht dev package

include(joinpath(@__DIR__, "src/QueueModelTracking.jl"))
include("data.jl")
include("util.jl")


function dx_fluid!(dx, x, params, t)
    (; A, W, M, K, p_smooth, λ) = params

    tmpq = zeros(eltype(x), length(K))
    tmpps = similar(x)
    for i in eachindex(M)
        tmpq[Int(M[i])] += x[i]
    end
    tmpq .= g_smooth.(tmpq, K, p_smooth)
    for i in eachindex(M)
        tmpps[i] = x[i] * tmpq[Int(M[i])]
    end

    mul!(dx, W', tmpps)
    mul!(dx, A, λ, true, true)
end
function dx_nn_base(x, params, t)
    re_ql(params.nnps_ql)([x; params.λ[1]; params.p[1]])
end
function dx_nn(x, params, t)
    dx = dx_nn_base(x, params, t)
    mul!(dx, params.A, params.λ, true, true)
    dx
end
function dx_both!(dx, x, params, t)
    dx_fluid!(dx, x, params, t) # Comment for only NN
    dx .+= dx_nn_base(x, params, t)
    return nothing
end

function solve_fluid(params)
    prob = ODEProblem(dx_fluid!, params.x0, params.tspan, params)
    solve(prob)
end
function solve_nn(params)
    prob = ODEProblem(dx_nn, params.x0, params.tspan, params)
    solve(prob)
end
function solve_both(params)
    prob = ODEProblem(dx_both!, params.x0, params.tspan, params)
    solve(prob)
end

# Function updating the parameters accordingly
function update_params!(params, pw, λ)
    params.p .= softmax(pw)
    for i in eachindex(params.p)
        params.P[Int(params.p_ind[2i-1]), Int(params.p_ind[2i])] = params.p[i]
    end
    params.W .= params.Ψ + params.B * params.P * params.A'
    params.λ = λ
end
function update_params(params, pw, λ)
    p = softmax(pw)
    P = Matrix{eltype(p)}(params.P)
    for i in eachindex(p)
        P[Int(params.p_ind[2i-1]), Int(params.p_ind[2i])] = p[i]
    end
    W = params.Ψ + params.B * P * params.A'
    (;params..., p, P, W, λ)
end

function loss_ql_nn(nnps_ql, params)
    params = ComponentArray(; params..., nnps_ql)
    loss = 0.0
    for (xavg, pw, λ) in zip(eachcol(params.class_hist), eachcol(params.w_hist), eachcol(params.λ_hist))
        update_params!(params, pw, λ)
        sol = solve_nn(params)
        final_state = sol[end]
        classlengths = zeros(eltype(final_state), length(xavg))
        for i in eachindex(params.N)
            classlengths[Int(params.N[i])] += final_state[i]
        end
        loss += sum(abs2, classlengths - xavg)
    end
    loss / prod(size(params.class_hist))
end
function loss_ql_both(nnps_ql, params)
    params = ComponentArray(; params..., nnps_ql)
    loss = 0.0
    for (xavg, pw, λ) in zip(eachcol(params.class_hist), eachcol(params.w_hist), eachcol(params.λ_hist))
        update_params!(params, pw, λ)
        sol = solve_both(params)
        final_state = sol[end]
        classlengths = zeros(eltype(final_state), length(xavg))
        for i in eachindex(params.N)
            classlengths[Int(params.N[i])] += final_state[i]
        end
        loss += sum(abs2, classlengths - xavg)
    end
    loss / prod(size(params.class_hist))
end

# function solveFluidModel(params; solver=Tsit5())
#     fluid_params = get_fluid_params(params)
# 
#     tspan = (0.0, 5.0)
#     x0 = zeros(T, length(M))
# 
#     prob = ODEProblem(dx_fluid!, x0, tspan, fluid_params)
#     sol = solve(prob, solver)
# 
#     return sol
# end

function dx_p95!(dx, x, W, t)
    mul!(dx, W', x)
end

function p95condition(u,t,integrator) 
    # Returned value should be positive normally, and cross 0 when we want something to happen
    sum(u) - 0.05
end

function solve_p95(params; xf)
    ps = g_smooth.(xf[Int.(params.M)], params.K[Int.(params.M)], params.p_smooth[Int.(params.M)])

    # Since Cr includes all classes, we can obtain the p95 matrices directly
    β = normalize(params.λ, 1)

    cb = ContinuousCallback(p95condition, terminate!)

    tspan = (0.0, 100.0)
    T = eltype(params.P)
    x0 = T.(params.A*β)
    
    prob = ODEProblem(dx_p95!, x0, tspan, (diagm(ps)*params.W))
    sol = solve(prob, Tsit5(), callback=cb)
    sol.t[end]
end

# Can be both calculated with closed form and as an ODE solution
function learn_p95(params)
    nq = length(params.K)
    n = length(params.p95_hist)
    xfs = Array{Float64,2}(undef, nq, n)
    diffs = Array{Float64,2}(undef, 1, n)

    for i in 1:n
        update_params!(params, params.w_hist[:, i], params.λ_hist[:, i])
        sol_xf = if params.use_fluid == 1
            solve_both(params)
        else
            solve_nn(params)
        end
        for j in 1:nq
            xfs[j, i] = sum(sol_xf.u[end][params.M .== j])
        end

        p95_pred = solve_p95(params; xf=xfs[:, i])

        diffs[1, i] = params.p95_hist[i] - p95_pred
    end

    net = Chain(
        Dense(nq, 10, elu),
        #Dense(10, 10, elu),
        Dense(10, 1),
    )
    ps, re = Flux.destructure(net)

    alpha = 0.01
    loss(nnps, p) = Flux.Losses.mse(re(nnps)(p[1]), p[2]) + alpha * sum(abs2, nnps)

    optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())

    @info "Running ADAM opt for p95"
    optprob = OptimizationProblem(optf, ps, (xfs, diffs))
    optsol = solve(optprob, ADAM(0.01), maxiters=100, callback=callback)

    #@info "Running BFGS opt for p95"
    #optprob = OptimizationProblem(optf, optsol.u, (xfs, diffs))
    #optsol = solve(optprob, BFGS(), maxiters=100, callback=callback)

    re(optsol.u)
end

# The cost function to the differentiated.
function costfunc_ql(pw, params)
    # Only update p, keep lambda
    params = update_params(params, pw, params.λ)
    # Solve the fluid model
    sol = if params.use_fluid == 1
        if params.use_nn == 1
            solve_both(params) 
        else
            solve_fluid(params)
        end
    else
        solve_nn(params)
    end

    # Calculate the queue length in each of the queues
    xf = [sum(sol.u[end][params.M .== i], dims=1)[1] for i in 1:length(params.K)]

    p95 = p95_pred(params; xf)

    if p95 > p95_lim
        return costFromP95(params.C_p, p95)
    else
        return costFromQL(params.C_q, xf)
    end
end
function p95_pred(params; xf) 
    if params.use_nn == 1
        solve_p95(params; xf) + only(re_p95(params.nnps_p95)(xf))
    else
        solve_p95(params; xf)
    end
end

function train_on_data(
    data; 
    w0=[2.2, 0], 
    hist_data=[], 
    use_nn=true,
    use_fluid=true,
    fluid_net=false,
    hidden_units,
    hidden_layers,
    flowconstrained,
    modelconstrained,
    ql_maxiters,
    seed=37,
)
    Random.seed!(seed)

    class_hist = hcat(
        map(x -> x[2], hist_data)...,
        data.pop_class_mean
    )
    w_hist = hcat(map(x -> x[1], hist_data)..., w0)
    p95_hist = hcat(map(x -> x[3], hist_data)..., data.p95_data)
    λ_hist = hcat(map(x -> x[4], hist_data)..., data.λ)
    params = ComponentArray(; 
        data.Ψ, 
        data.A, 
        data.B, 
        data.P, 
        data.λ, 
        data.M, 
        data.K, 
        data.p_smooth, 
        data.p_ind, 
        data.N, 
        data.C_p, 
        data.C_q, 
        class_hist, w_hist, p95_hist, λ_hist,
        use_nn, use_fluid,
        W=similar(data.Ψ),
        x0=zeros(length(data.N)),
        w=w0,
        p=softmax(w0),
        tspan=[0.0, 2.0], # Needs to be tuned on different apps
    )
    update_params!(params, w0, params.λ)

    params = if use_nn
        global re_ql
        nnps_ql, re_ql = create_net(size(params.A, 1), hidden_units, hidden_layers, flowconstrained, modelconstrained)

        optf = OptimizationFunction(use_fluid ? loss_ql_both : loss_ql_nn, Optimization.AutoForwardDiff())

        @info "Running ADAM opt for ql"
        optprob = OptimizationProblem(optf, nnps_ql, params)
        optsol = solve(optprob, ADAM(0.01), maxiters=ql_maxiters, callback=callback_log)

        #@info "Running BFGS opt for ql"
        #optprob = OptimizationProblem(optf, optsol.u, params)
        #optsol = solve(optprob, BFGS(), maxiters=ql_maxiters÷10, callback=callback_log)

        learned_params = ComponentArray(params; nnps_ql=optsol.u)
        p95_diff_net = learn_p95(learned_params)
        global re_p95
        nnps_p95, re_p95 = Flux.destructure(p95_diff_net)

        ComponentArray(learned_params; nnps_p95)
    else
        params
    end

    return params
end
