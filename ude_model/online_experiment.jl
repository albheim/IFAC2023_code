include("autodiff.jl")

lock_file = joinpath(@__DIR__, "lock")
touch(lock_file)
last_time = mtime(lock_file)
while true
    printstyled("Waiting for updated values\n", bold=true, color=:green)
    while last_time == mtime(lock_file)
        sleep(0.5)
    end

    # In-parameters
    input_dict = YAML.load_file(joinpath(@__DIR__, "input.yaml"); dicttype=Dict{String, Any})
    logfolder = input_dict["logfolder"]
    tracefolder = input_dict["tracefolder"]
    w0 = input_dict["p0"]

    find_next_param(logfolder, tracefolder, w0, tag)
    touch(lock_file)
    last_time = mtime(lock_file)
end