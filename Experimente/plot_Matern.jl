using JLD2
# using TuePlots
using Statistics
using CairoMakie
using LaTeXStrings

CairoMakie.activate!(type = "svg")

loaded_results = load("./Experimente/out/on-model_results_matern12.jld2")

include("./plot_theme.jl")


WIDTH, HEIGHT = FULL_WIDTH, 1.35FULL_HEIGHT


fig = begin

    grid_plot = Figure(;
        size=(WIDTH, HEIGHT),
        figure_padding=1,
    )

    rmse_axes = []
    covdist_axes = []
    legend_handles = []
    for (plot_i, cur_lx) in enumerate(loaded_results["spatial_lengthscale_list"])
        push!(
            rmse_axes,
            Axis(
                grid_plot[1, plot_i],
                xticks=round.(Int, LinRange(1, loaded_results["nval_list"][end], 5)),
                xtrimspine=true,
                ytrimspine=(true, false),
                topspinevisible = false,
                rightspinevisible = false,
                xgridvisible = false,
                ygridvisible = false,
                xticklabelsvisible=false,
                aspect=1.5,
                # title="vs. KF mean",
                titlesize=BASE_FONTSIZE - 3,
                titlealign=:left,
                titlegap=0.0,
            )
        )
        Label(grid_plot[1, plot_i, Top()], L"\ell_x = %$cur_lx", padding=(0, 0, 0, 0), fontsize=BASE_FONTSIZE-1)

        #spectrum_legendhandle = lines!(
        #    rmse_axes[end],
        #    # loaded_results["nval_list"],
        #    cumsum(loaded_results["eval_results"][plot_i]]["kf_cov"]) / sum(loaded_results["eval_results"][plot_i]["kf_cov"]),
        #    color=:grey80,
        #    linewidth=1.5,
        #)


        band!(
            rmse_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]] - 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]] + 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]],
            color=(COLORS[2], 0.4)
        )
        _enkfline = scatterlines!(
            rmse_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]],
            label="EnKF",
            marker=:rect,             color=COLORS[2]
        )


        band!(
            rmse_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]] - 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]] + 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]],
            color=(COLORS[1], 0.4)
        )
        _etkfline = scatterlines!(
            rmse_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]],
            label="ETKF",
                        color=COLORS[1]
        )

        _rrkfline = scatterlines!(
            rmse_axes[end],
            loaded_results["nval_list"],
            loaded_results["eval_results"][plot_i]["rrkf"]["rmse_to_kf"],
            label="SKF",
            marker=:diamond,             color=COLORS[3], #zorder=10
        )
        lastpoint_legend_handle = CairoMakie.scatter!(
            rmse_axes[end],
            loaded_results["nval_list"][end:end],
            loaded_results["eval_results"][plot_i]["rrkf"]["rmse_to_kf"][end:end],
            label="SKF",
            marker=:diamond,
            color=COLORS[4],
            strokecolor=:black,
            strokewidth=0.5,
            # zorder=10,
        )

        if plot_i == 1
            push!(legend_handles, _rrkfline)
            push!(legend_handles, _enkfline)
            push!(legend_handles, _etkfline)
            #push!(legend_handles, spectrum_legendhandle)
        end

        push!(
            covdist_axes,
            Axis(
                grid_plot[2, plot_i],
                # yscale=log10,
                xticks=round.(Int, LinRange(1, loaded_results["nval_list"][end], 5)),
                xtrimspine=true,
                ytrimspine=(true, false),
                topspinevisible = false,
                rightspinevisible = false,
                xgridvisible = false,
                ygridvisible = false,
                xticklabelsvisible=true,
                aspect=1.5,
                xlabel="low-rank dim.",
                titlesize=BASE_FONTSIZE-3,
                titlegap=0.1,
                titlealign=:left,
            )
        )
        #lines!(
        #    covdist_axes[end],
        #    cumsum(loaded_results["eval_results"][plot_i]) / sum(loaded_results["eval_results"][plot_i]),
        #    color=:grey80,
        #    linewidth=1.5,
        #)



        band!(
            covdist_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]] - 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]] + 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]],
            color=(COLORS[2], 0.4)
        )
        scatterlines!(
            covdist_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]],
            label="EnKF",
            marker=:rect, color=COLORS[2],
            )



        band!(
            covdist_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]] - 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]] + 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]],
            color=(COLORS[1], 0.4)
        )
        scatterlines!(
            covdist_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]],
            label="ETKF",
            color=COLORS[1],
            )


        scatterlines!(
            covdist_axes[end],
            loaded_results["nval_list"],
            loaded_results["eval_results"][plot_i]["rrkf"]["cov_distance"],
            label="SKF",
            marker=:diamond, color=COLORS[3], # zorder=10
        )
        CairoMakie.scatter!(
            covdist_axes[end],
            loaded_results["nval_list"][end:end],
            loaded_results["eval_results"][plot_i]["rrkf"]["cov_distance"][end:end],
            label="SKF",
            marker=:diamond,
            color=COLORS[4],
            strokecolor=:black,
            strokewidth=0.5,
            # zorder=10,
        )


    end

    Legend(grid_plot[0, :], legend_handles, [rich("RRKF (ours)", font="Times New Roman bold"), "EnKF", "ETKF"], orientation=:horizontal)

    rmse_axes[begin].ylabel = "RMSE"
    covdist_axes[begin].ylabel = rich("Frobenius", "\n", rich("distance", offset = (0.0, 1.0)))

    linkyaxes!(rmse_axes...)
    linkyaxes!(covdist_axes...)


    rowgap!(grid_plot.layout, 8.0)
    rowgap!(grid_plot.layout, 1, 8.0)
    colgap!(grid_plot.layout, 5.0)

    grid_plot
end

display(fig)
save("./Experimente/out/on-model_results_matern12.pdf", fig, pt_per_unit = 1)



