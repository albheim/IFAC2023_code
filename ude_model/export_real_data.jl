include("data.jl")

datafolder="logs"
logfolder = joinpath(@__DIR__, "..", "data", datafolder)
all_data = [get_data(logfolder, joinpath(logfolder, "traces", "sample$(round(Int, 20*p+1))")) for p in 0:0.05:1]

CSV.write(joinpath(@__DIR__, "..", "data", datafolder, "data_real.csv"), 
    (
        p1=0:0.05:1,
        ql_data=map(data->sum(data.pop_queue_mean), all_data),
        p95_data=map(data->data.p95_data, all_data),
        cost_data=map(data->data.cost_data, all_data),
    ),
    writeheader=true, 
    header=[:p1, :ql_data, :p95_data, :cost_data],
)