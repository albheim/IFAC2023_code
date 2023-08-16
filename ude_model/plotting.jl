using PyPlot, CSV

basepath = joinpath(@__DIR__, "..", "data")
runid = "logs"

datatag = "real"
realdata = CSV.File(joinpath(basepath, runid, "data_$(datatag).csv"))

tags = [
    "fluid_nn_hu20_hl2_mi1500_longer",
    "nn_hu20_hl2_mi1500_longer",
    "fluid_nnmc_hu20_hl2_mi1500_longer",
    "fluid_nnmc_hu20_hl2_mi1500_longer",
    "fluid_nnmc_hu20_hl2_mi500_fix_mc",
    #"fluid_nn_hu40_hl3_mi1000_long_and_big",
    #"nn_hu40_hl3_mi1000_long_and_big",
    #"fluid_nnmc_hu40_hl3_mi500_long_and_big",
    #"fluid_nn_hu20_hl2_mi500_v1_i1",
    #"fluid_nnfc_hu20_hl2_mi500_v1_i1",
    #"fluid_nnmc_hu20_hl2_mi500_v1_i1",
    #"nn_hu20_hl2_mi500_v1_i1",
    "fluid_nnmc_hu10_hl1_mi100_test_ouput",
    "fluid_v1",
]
datas = [CSV.File(joinpath(basepath, runid, "data_$(tag).csv")) for tag in tags]

p95_lim = 0.55
cost_per_p95 = 10.0

begin
    figure(32)
    clf()
    subplot(1, 3, 1)
    plot(realdata.p1, realdata.ql_data, "*C1--")
    #plot(realdata.p1, realdata.ql_train, "C0")
    for i in eachindex(datas)
        plot(datas[i].p1, datas[i].ql_i, "+:")
    end
    xlim([0, 1])
    #ylim([0.9*minimum(ql_data), 1.1*maximum(ql_data)])
    title("Total queue length")
    #legend()
    subplot(1, 3, 2)
    plot(realdata.p1, realdata.p95_data, "*C1--")
    #plot(realdata.p1, realdata.p95_train, "C0")
    for i in eachindex(datas)
        plot(datas[i].p1, datas[i].p95_i, "+:")
    end
    plot([0, 1], [p95_lim, p95_lim], "k--")
    title("p95 response time")
    xlim([0, 1])
    #ylim([0.9*minimum(p95_data), 1.1*maximum(p95_data)])
    #legend()
    subplot(1, 3, 3)
    # Use violating to plot this nicer?
    plot(realdata.p1, realdata.cost_data, "*C1--", label="data")
    #plot(realdata.p1, realdata.cost_train, "C0", label="fluid")
    for i in eachindex(datas)
        plot(datas[i].p1, datas[i].cost_i, "+:", label=tags[i])
    end
    xlim([0, 1])
    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Total cost \n (what we minimize over)")
    legend()
end

gcf()

begin
    figure(32)
    clf()
    subplot(1, 3, 1)
    plot(realdata.p1, realdata.ql_data, "*C1--")
    #plot(realdata.p1, realdata.ql_train, "C0")
    for i in eachindex(datas)
        plot(datas[i].p1, datas[i].ql_t, "+C$(i+2):")
    end
    xlim([0, 1])
    #ylim([0.9*minimum(ql_data), 1.1*maximum(ql_data)])
    title("Total queue length")
    #legend()
    subplot(1, 3, 2)
    plot(realdata.p1, realdata.p95_data, "*C1--")
    #plot(realdata.p1, realdata.p95_train, "C0")
    for i in eachindex(datas)
        plot(datas[i].p1, datas[i].p95_t, "+C$(i+2):")
    end
    plot([0, 1], [p95_lim, p95_lim], "k--")
    title("p95 response time")
    xlim([0, 1])
    #ylim([0.9*minimum(p95_data), 1.1*maximum(p95_data)])
    #legend()
    subplot(1, 3, 3)
    # Use violating to plot this nicer?
    plot(realdata.p1, realdata.cost_data, "*C1--", label="data")
    #plot(realdata.p1, realdata.cost_train, "C0", label="fluid")
    for i in eachindex(datas)
        plot(datas[i].p1, datas[i].cost_t, "+C$(i+2):", label=tags[i])
    end
    xlim([0, 1])
    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Total cost \n (what we minimize over)")
    legend()
end

gcf()

savefig(joinpath(basepath, runid, "plot_$(tag).png"))

## Plotting time series

basepath = joinpath(@__DIR__, "..", "data")
runid = "logs"

tags = [
    "fluid_nn_hu20_hl2_mi1500_longer",
    "nn_hu20_hl2_mi1500_longer",
    "fluid_nnmc_hu20_hl2_mi1500_longer",
    "fluid_nn_hu40_hl3_mi1000_long_and_big",
    "nn_hu40_hl3_mi1000_long_and_big",
    "fluid_nnmc_hu40_hl3_mi500_long_and_big",
    "fluid_nnmc_hu20_hl2_mi500_fix_mc",
    "fluid_nnmc_hu20_hl2_mi1500_fix_mc",
    #"fluid_nn_hu20_hl2_mi500_v1_i1",
    #"fluid_nnfc_hu20_hl2_mi500_v1_i1",
    #"fluid_nnmc_hu20_hl2_mi500_v1_i1",
    "nn_hu20_hl2_mi500_v1_i1",
]
    
datas = [CSV.File(joinpath(basepath, runid, "data_$(tag)_timingdata.csv")) for tag in tags]

begin
    figure(23424, figsize=(10.0, 8.0))
    clf()
    subplot(2, 1, 1)
    for i in eachindex(datas)
        semilogy(datas[i].iter, datas[i].dt, "C$(i):", label=tags[i])
    end
    #xlim([0, 1])
    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Time per gradient step")
    legend()
    subplot(2, 1, 2)
    for i in eachindex(datas)
        semilogy(datas[i].iter, datas[i].loss, "C$(i):")
    end
    #xlim([0, 1])
    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Loss")
end

gcf()


# if not yet done
begin
    figure(23424, figsize=(10.0, 8.0))
    clf()
    for i in eachindex(datas)
        semilogy(datas[i].loss, "C$(i):")
    end
    #xlim([0, 1])
    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Loss")
    legend()
end

gcf()

# foreach(tags) do tag
#     fname = joinpath(basepath, runid, "data_$(tag)_timingdata.csv")
#     fix_data(fname)
# end