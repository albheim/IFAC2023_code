include("autodiff.jl")

eps = 1e-8
logfolder = joinpath(@__DIR__, "..", "data", "logs_grid_p05_no_edge")
for p in 0.0:0.05:1.0
    idx = round(Int, 20 * p) + 1
    tracefolder = joinpath(logfolder, "traces", "sample$(idx)")
    w0 = [log((p+eps)/(1-p+eps)), 0]
    find_next_param(logfolder, tracefolder, w0, "test_nn_grid_fluid_nn_full_p95")
end