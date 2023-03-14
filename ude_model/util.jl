## ===== Parameters =====

# Cost per stationary mean queue length
cost_per_ql::Dict{String, Float64} = Dict{String, Float64}(
    "frontend" => 0,
    "backend-v1" => 3,
    "backend-v2" => 2,
    "backend-v3" => 1,
)

# Cost for violating the p95 limit
cost_per_p95::Float64 = 10.0

# Limit of p95 RT which we will not move past
p95_lim::Float64 = 0.55

# Number of phase states per PH distribution 
phases::Dict{String, Int} = Dict("ec"=>1, "m"=>3, "d"=>3)

# Number of processors per service
processors_per_service::Int = 4

# Printout verbosity
verbose::Bool = true

# Function for retrieving the cost from the queue lengths
function costFromQL(C_q, x)
    return dot(C_q, x)
end

# Function for retrieving the cost from the p95 estimation
function costFromP95(C_p, p95)
    return C_p * p95
end

function softmax(x)
    expx = exp.(x)
    return expx ./ sum(expx)
end

function create_net(n_phases, hidden_units, hidden_layers, flowconstrained, modelconstrained)
    net = Chain(
        Dense(n_phases + 2, hidden_units, elu),
        [Dense(hidden_units, hidden_units, elu) for i in 1:hidden_layers]...,
        Dense(hidden_units, n_phases),
        if flowconstrained
            (x -> x .- mean(x),)
        elseif modelconstrained
            (create_constrained_flow,)
        else
            ()
        end...,
    )
    Flux.destructure(net)
end

struct SplitCombineLayer{F}
    chain1::Chain
    chain2::Chain
    n1::Int
    combine::F
end
function (m::SplitCombineLayer)(x)
    #m.combine(m.chain1(x[1:m.n1]), m.chain2(x[m.n1+1:end]))
    m.combine(m.chain1(x), m.chain2(x))
end
Flux.@functor SplitCombineLayer
struct SkipLayer{F}
    chain::Chain
    combine::F
end
function (m::SkipLayer)(x)
    m.combine(m.chain(x), x)
end
Flux.@functor SkipLayer
function create_net_flowconserving2(params)
    n_phases = size(params.A, 1)
    net = SplitCombineLayer(
        Chain(
            Dense(2, 10, elu),
            Dense(10, n_phases*n_phases),
            x -> reshape(x, n_phases, n_phases),
            x -> x .- mean(x, dims=1),
        ),
        Chain(
            SkipLayer(
                Chain(
                    Dense(n_phases, 5, elu), # 5 is nq
                    Dense(5, n_phases, sigmoid),
                ),
                .*
            )
        ),
        2, *
    )
    Flux.destructure(net)
end
create_constrained_flow(x) = [
    -x[1]
    x[1] - x[2]
    x[2] - x[3]
    x[18] + x[24] - x[4]
    x[4] - x[5]
    x[5] - x[6]
    x[15] - x[7]
    x[7] - x[8]
    x[8] - x[9]
    x[21] - x[10]
    x[10] - x[11]
    x[11] - x[12]
    x[3]/3 - x[13]
    x[13] - x[14]
    x[14] - x[15]
    x[9] - x[16]
    x[16] - x[17]
    x[17] - x[18]
    x[3]*2/3 - x[19]
    x[19] - x[20]
    x[20] - x[21]
    x[12] - x[22]
    x[22] - x[23]
    x[23] - x[24]
]
function create_net_flowconserving3(params)
    n_phases = size(params.A, 1)
    net = Chain(
        Dense(n_phases+2, 10, elu),
        Dense(10, 10, elu),
        Dense(10, n_phases+1, softplus),
        create_constrained_flow,
    )
    Flux.destructure(net)
end

function callback(ps, l)
    t = time()
    println("$(t), $(l)")
    false
end

function callback_log(ps, l)
    t = time()
    println("$(t), $(l)")
    CSV.write(timingfile, (t=[t], loss=[l]), append=true)
    false
end

function phase2class(x, N)
    classlengths = zeros(eltype(x), maximum(Int, N))
    for i in eachindex(N)
        classlengths[Int(N[i])] += x[i]
    end
    return classlengths
end