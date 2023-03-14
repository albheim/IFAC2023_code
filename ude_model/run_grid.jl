using ArgParse, Distributed
import Pkg

include("train_grid.jl")

function run_distributed(; parsed_args...)
    rmprocs()
    addprocs(min(Sys.CPU_THREADS, parsed_args[:reps]))
    @info workers()
    @everywhere begin
        @eval import Pkg
        Pkg.activate($(Pkg.project().path))
        include($(joinpath(@__DIR__, "train_grid.jl")))
    end
    Distributed.pmap(1:parsed_args[:reps]) do i
        train_point_and_eval_grid(0.6, 0.1:0.2:1.0; parsed_args..., tag=parsed_args[:tag] * "_i$i", seed=i)
    end
end

function run_single(; parsed_args...)
    train_point_and_eval_grid(0.6, 0.1:0.2:1.0; parsed_args...)
end

function get_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--use_nn"
            action = :store_true
        "--use_fluid"
            action = :store_true
        "--flowconstrained", "--fc"
            action = :store_true
        "--modelconstrained", "--mc"
            action = :store_true
        "--fluid_net"
            action = :store_true
        "--hidden_units", "--hu"
            arg_type = Int
            default = 20
        "--hidden_layers", "--hl"
            arg_type = Int
            default = 0
        "--ql_maxiters", "--mi"
            arg_type = Int
            default = 30
        "--tag"
            arg_type = String
            default = "test"
        "--reps"
            arg_type = Int
            default = 1
    end
    return parse_args(s; as_symbols=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    parsed_args = get_args()
    if parsed_args[:reps] == 1
        run_single(; parsed_args...)
    else
        run_distributed(; parsed_args...)
    end
end

# Run like this from folder
# julia --project run_grid.jl --use_nn --use_fluid --fc --hu=30 --hl=1 --mi=200 --reps=5 --tag=mean_check