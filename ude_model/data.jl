using JLD

function get_data(logfolder, tracefolder)
    verbose=true

    ## ===== Read data =====
    if verbose printstyled("Reading the data\n",bold=true, color=:green); end

    converted_fname = joinpath(tracefolder, "converted.jld")
    if isfile(converted_fname)
        @info "Found precomputed" converted_fname
        return load(converted_fname, "data_ntuple")
    end

    # Read the loadgenerator settings file
    loadgensettingsfiles = joinpath.(logfolder, filter(x -> occursin("settings", x), readdir(logfolder)))
    simSettings = Dict{String, Dict{String, Any}}()
    for file in loadgensettingsfiles
        lg = split(split(file, "settings_")[2], ".")[1]
        simSettings[lg] = YAML.load_file(file; dicttype=Dict{String, Any}) 
        simSettings[lg]["experimentTime"] = unix2datetime.(simSettings[lg]["experimentTime"])
    end

    # read the trace files
    df_pods, df_err, err_prct, ipname_bimap = readTraceData(tracefolder, simSettings, verbose=false)

    # report errors (if larger than some percentage?)
    if sum(values(err_prct)) > 0.0
        printstyled("Warning, errors detected in reading data\n", color=:red)
        for key in keys(err_prct)
            if err_prct[key] > 0.0
                printstyled("\t$key error: $(round(err_prct[key], digits=2)) %\n", color=:red)
            end
        end
    end

    # extract data into H objects
    data_H = traceData2Hr(df_pods)

    ## ===== Extract the queueing network topology =====
    if verbose printstyled("Extracting network topology\n",bold=true, color=:green); end

    # Extract classes and queue
    classes, ext_arrs,  queue_type = createClasses(data_H, simSettings)
    queues = unique(getindex.(classes, 1))

    # Boolean vectors for class containment in network, application 
    classInApp = ones(Bool, length(classes))
    for (k, class) in enumerate(classes)
        if queue_type[class[1]] == "ec"
            classInApp[k] = 0
        end
    end

    queueInApp = ones(Bool, length(queues))
    queueIsService = zeros(Bool, length(queues))
    for (k, queue) in enumerate(queues)
        if queue_type[queue] == "ec"
            queueInApp[k] = 0
        elseif queue_type[queue] == "m"
            queueIsService[k] = 1
        end
    end

    # Retrieve important queuing network parameters
    Cq = [sum([qc == q for (qc, _, _) in classes]) for q in queues]
    S = zeros(Int64, length(classes))
    queue_disc = Dict{T_qn, String}()
    queue_servers = Dict{T_qn, Int}()
    for (i, (q, _, _)) = enumerate(classes)
        S[i] = phases[queue_type[q]]
    end
    for q in queues
        queue_disc[q] = (queue_type[q] == "m" ? "PS" : "INF")
        queue_servers[q] = (queue_type[q] == "m" ? processors_per_service :  typemax(Int64))
    end
    M, Mc, N = getPhasePos(S, length(queues), Cq)

    ## ===== Extract queueing network states and variables =====
    if verbose printstyled("Extracting network states and variables\n",bold=true, color=:green); end

    # Calculate arrival/departure times for each class in each sim. 
    ta_class, td_class, ri_class = getArrivalDeparture(data_H, simSettings, classes, queue_type)

    @assert all(vcat(td_class...) .>= vcat(ta_class...))
    tw_class = td_class - ta_class

    # Extract path of requests over classes
    paths_class, paths_err = getAllPaths(ri_class, ta_class, td_class, 
        classes, ext_arrs, queue_type)

    nbr_p_err = length(values(paths_err))
    nbr_p = length(values(paths_class))
    if nbr_p_err > 0
        printstyled("Warning, path extraction error\n", color=:red)
        printstyled("\terr: $(round(nbr_p_err / (nbr_p_err + nbr_p) * 100, digits=2)) %\n",
            color=:red)
    end

    # Extract populations for queues and classes
    pop_class = getQueueLengths.(ta_class, td_class, start_time=minimum(minimum(ta_class)))
    pop_queue = Vector{Matrix{Float64}}(undef, length(queues))
    for i = 1:length(queues)
        pop_queue[i] = addQueueLengths(pop_class[Mc .== i])
    end

    # Calculate mean queue length in both classes and queues
    pop_class_mean = getQueueLengthAvg.(pop_class)
    pop_queue_mean = getQueueLengthAvg.(pop_queue)

    # Calcualte utilization and optimal smoothing values
    util = zeros(length(queues))
    p_smooth_opt = zeros(length(queues))
    for (i, q) in enumerate(queues)
        util[i] = getUtil(pop_queue[i], queue_servers[q])
        p_smooth_opt[i] = getOptimPNorm(pop_queue_mean[i], util[i], queue_servers[q])
    end

    # Calculate the service times
    ts_class = getServiceTimes(ta_class, td_class, queue_servers, classes) 

    # Fit PH distribution to classes
    if verbose printstyled("Fitting PH dist\n", bold=true, color=:magenta); end
    ph_vec = Vector{EMpht.PhaseType}(undef, length(classes))
    for (i, (q, _, _)) = enumerate(classes)
        if verbose println("\t$i / $(length(classes))"); end
        ph_vec[i] = fitPhaseDist(filtOutliers(ts_class[i], ϵ=0, α=0.99, β=10), 
                phases[queue_type[q]], max_iter=200, verbose=false)
    end

    # Extract arrival rates
    w_a = getExternalArrivals(data_H, ext_arrs, classes)
    λ = map(x -> x > 0 ? x : 0, w_a) ./ 
        (maximum(maximum(td_class)) - minimum(minimum(ta_class)))

    # Extract current routing probability matrix
    P = zeros(length(classes), length(classes))
    class_idx_dict = classes2map(classes)
    w = getClassRoutes(data_H, classes, queue_type)
    for (i, (q, n, u)) in enumerate(classes)
        w_tmp = w_a[i] < 0 ? w_a[class_idx_dict[(q, n, 1)]] : 0
        if (sum(w[i, :]) + w_tmp) > 0
            P[i, :] = w[i, :] ./ (sum(w[i, :]) + w_tmp)
        end
    end

    # Get the indexes of p for all backends
    depart = (ipname_bimap["frontend"], "/detect/", 1)
    backends = sort(collect(filter(x -> occursin("backend", x), keys(ipname_bimap))))
    p_idx = Int[]
    for backend in backends
        arrives = ((depart[1], ipname_bimap[backend]), (depart[2], "/detect/"), 1)
        append!(p_idx, (findfirst(c -> c == depart, classes),  findfirst(c -> c == arrives, classes)))
    end

    # Extract the fluid model parameters
    Ψ, A, B = getStackedPHMatrices(ph_vec)
    K = [queue_servers[q] for q in queues]

    # The p95 response times according to the data
    ta_cr, td_cr, _ = getArrivalDeparture(paths_class, classes)
    tw_cr = td_cr - ta_cr
    p95_data = findQuantileData(tw_cr, 0.95)

    violating = p95_data > p95_lim # Use p95_data instead for soft cost

    # Extract cost vectors
    C_p = cost_per_p95
    C_q = zeros(length(queues))
    for (i, q) in enumerate(queues)
        if haskey(ipname_bimap, q)
            C_q[i] = get(cost_per_ql, ipname_bimap[q], 0)
        end
    end

    # The current cost
    cost_data = p95_data > p95_lim ? costFromP95(C_p, p95_data) : costFromQL(C_q, pop_queue_mean)

    data_ntuple = (; Ψ, A, B, P, λ, M, K, p_smooth=p_smooth_opt, p_ind=p_idx, N, C_p, C_q, pop_queue_mean, pop_class_mean, pop_queue, pop_class, cost_data, violating, p95_data, Mc)

    @save converted_fname data_ntuple
    return data_ntuple
end