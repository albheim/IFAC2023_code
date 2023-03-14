using PyPlot, CSV

basepath = joinpath(@__DIR__, "..", "data")
runid = "logs"

datatag = "fluid_only_retrain"
realdata = CSV.File(joinpath(basepath, runid, "data_$(datatag).csv"))

tags = [
    #"fluid_only_06_nohist", 
    #"fluid_nn_06_nohist", 
    #"fluid_nn_06_hist_1to65", 
    #"fluid_only_06_hist_01_step_02_test_nnfluidflags",
    #"nn_only_06_hist_01_step_02_test_nnfluidflags",
    "fluid_nnfc_hu20_hl0_p0.6_hist0.1:0.2:0.9_test_new",
    #"fluid_nnfc_hu20_hl0_mi300_p0.6_hist0.1:0.2:0.9_normal_long",
    #"fluid_nnfc_hu10_hl5_mi300_p0.6_hist0.1:0.2:0.9_deep_small_long",
    #"fluid_nnfc_hu20_hl0_mi30_p0.6_hist0.1:0.2:0.9_normal",
    "nn_hu30_hl2_mi300_p0.6_hist0.1:0.2:0.9_test_only_large_nn",
    #"fluid_nn_06_hist_01_step_02_test_fancynet",
    #"fluid_nn_06_hist_01_step_02_test_simple",
    #"fluid_nn_06_hist_01_step_02_largenet_more_train",
    #"fluid_nn_03_hist_0_03", 
    #"fluid_only_07_nohist", 
    #"fluid_only_03_nohist", 
    "fluid_nn_hu20_hl0_mi500_p0.6_hist0.1:0.2:0.9_for_paper_v1",
    "nn_hu20_hl0_mi500_p0.6_hist0.1:0.2:0.9_for_paper_v1",
    "fluid_nnfc_hu20_hl0_mi500_p0.6_hist0.1:0.2:0.9_for_paper_v1",
    "fluid_nnmc_hu20_hl0_mi500_p0.6_hist0.1:0.2:0.9_for_paper_v1",
    "fluid_nn_hu20_hl2_mi500_p0.6_hist0.1:0.2:0.9_for_paper_v1",
    "nn_hu20_hl2_mi500_p0.6_hist0.1:0.2:0.9_for_paper_v1",
    "fluid_nnfc_hu20_hl2_mi500_p0.6_hist0.1:0.2:0.9_for_paper_v1",
    "fluid_nnmc_hu20_hl2_mi500_p0.6_hist0.1:0.2:0.9_for_paper_v1",
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
    "fluid_nnmc_hu20_hl2_mi500_p0.6_hist0.1:0.2:0.9_v1_i1",
    "fluid_nnmc_hu20_hl2_mi500_p0.6_hist0.1:0.2:0.9_v1_i2",
    "fluid_nnmc_hu20_hl2_mi500_p0.6_hist0.1:0.2:0.9_v1_i3",
    "fluid_nnmc_hu20_hl2_mi500_p0.6_hist0.1:0.2:0.9_v1_i4",
    "fluid_nnmc_hu20_hl2_mi500_p0.6_hist0.1:0.2:0.9_v1_i5",
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