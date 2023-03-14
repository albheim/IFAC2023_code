using PyPlot, CSV, Statistics

basepath = joinpath(@__DIR__, "..", "data")
runid = "logs_grid_p05_no_edge"

tags = [
    #"fluid_nnfc_hu10_hl0_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    #"fluid_nn_hu10_hl0_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    #"nn_hu10_hl0_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    #"fluid_nnfc_hu30_hl1_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    #"fluid_nn_hu30_hl1_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    #"nn_hu30_hl1_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    "fluid_nnfc_hu20_hl0_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    "fluid_nn_hu20_hl0_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    "nn_hu20_hl0_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    "fluid_nnfc_hu20_hl1_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    "fluid_nn_hu20_hl1_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
    "nn_hu20_hl1_mi200_p0.6_hist0.1:0.2:0.9_mean_check",
]
datas = [
    begin
        reps = []
        while isfile(joinpath(basepath, runid, "data_$(tag)_i$(length(reps)+1)_timingdata.csv"))
            push!(reps, CSV.File(joinpath(basepath, runid, "data_$(tag)_i$(length(reps)+1)_timingdata.csv")))
        end
        @assert length(reps)>0 "no file found for tag=$(tag)"
        reps
    end for tag in tags
]
for i in eachindex(datas)
    loss_mean = mean.(zip(map(d->d.loss, datas[i])...))
    CSV.write(joinpath(basepath, runid, "data_$(tags[i])_avgdata.csv"), (; iter=1:length(loss_mean), loss=loss_mean); writeheader=true, header=[:iter, :loss])
end

begin
    figure(23424)
    clf()
    subplot(2, 1, 1)
    for i in eachindex(datas)
        timings = zip(map(d->d.dt, datas[i])...)
        semilogy(mean.(timings), "C$(i)", label=tags[i]*"_mean")
        #semilogy(std.(timings), "C$(i):", label=tags[i]*"_std")
    end
    #xlim([0, 1])
    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Time per gradient step")
    legend()
    subplot(2, 1, 2)
    for i in eachindex(datas)
        losses = zip(map(d->d.loss, datas[i])...)
        semilogy(mean.(losses), "C$(i)")
        #semilogy(std.(losses), "C$(i):")
    end
    #xlim([0, 1])
    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Loss")
end
gcf()

begin
    figure(23124)
    clf()
    subplot(2, 1, 1)
    for i in eachindex(datas)
        timings = map(d->d.dt, datas[i])
        semilogy(timings[1], "C$(i)", label=tags[i])
        for x in timings[2:end]
            semilogy(x, "C$(i)")
        end
    end
    #xlim([0, 1])
    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Time per gradient step")
    legend()
    subplot(2, 1, 2)
    for i in eachindex(datas)
        losses = map(d->d.loss, datas[i])
        semilogy(losses[1], "C$(i)", label=tags[i])
        for x in losses[2:end]
            semilogy(x, "C$(i)")
        end
    end
    #xlim([0, 1])
    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Loss")
end
gcf()