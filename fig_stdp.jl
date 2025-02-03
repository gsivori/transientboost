"""
Copyright 2023 Gaston Sivori

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

## RUN this block
using Plots
using LaTeXStrings
using JLD2
using SavitskyGolayFilters
#plotting parameters
figsize = (900,600)
default(legendfontsize = 12, guidefont = (16, :black), guide="", tickfont = (12, :gray),
    framestyle = nothing, yminorgrid = true, xminorgrid = true, size=(900,600), dpi=150)
#helper functions
include("funs.jl")

## Figure preparing

@load "../data/stdp_fig5b.jld2" means deltas delays rels
hstdp = plot(ytickfont=font(12),guidefont=font(12),yguidefontsize=12,xtickfont=font(12),
label=nothing,left_margin=2Plots.mm,minorgrid=true,grid=false,size=(450,400),
right_margin=2Plots.mm,top_margin=0Plots.mm,bottom_margin=2Plots.mm,legendfontsize=12,
xlabel="Delay from EPSP to second AP (ms)",ylabel="Relative synaptic strength (a.u.)",legend=false)
scatter!(delays,reverse(rels),markercolor=:black,markersize=1.5,markeralpha=0.15,markerstrokewidth=0.5,markershape=:circle)
#scatter!(delays,reverse(rels))
filtered_trace = savitskygolay(Float64.(reverse(means)),20,3,0)
plot!(deltas,filtered_trace,linewidth=2.0,color=:purple)
plot!(xticks=[-80,-40,0,40,80],yticks=[0.5,1,1.5,2],ylim=(0.25,2.0),minorgrid=false)
vline!([10.0],linestyle=:dash,color=:gray,linewidth=1.5)
vline!([0.0],linestyle=:solid,color=:black,linewidth=1.5)
hline!([1.0],linestyle=:solid,color=:black,linewidth=1.5)
savefig(hstdp,"Fig5b.pdf")

@load "../data/stdp_fig5a_1hz.jld2" means deltas delays rels
hstdp = plot(ytickfont=font(12),guidefont=font(12),yguidefontsize=12,xtickfont=font(12),
label=nothing,left_margin=2Plots.mm,minorgrid=false,grid=true,gridalpha=0.05,size=(450,400),
right_margin=2Plots.mm,top_margin=0Plots.mm,bottom_margin=2Plots.mm,legendfontsize=12,
xlabel="Δt (ms)",ylabel="Relative synaptic strength (a.u.)",legend=false)
scatter!(delays,reverse(rels),markercolor=:orange,markersize=1.5,markeralpha=0.15,markerstrokewidth=0.5,markershape=:circle,label=nothing)
filtered_trace = savitskygolay(Float64.(reverse(means)),20,3,0)
plot!(deltas,filtered_trace,linewidth=2.0,color=:orange,label="1 Hz.")
plot!(xticks=[-80,-40,0,40,80],yticks=[0.5,1,1.5,2],ylim=(0.25,2.0),minorgrid=false)
vline!([0.0],linestyle=:solid,color=:black,linewidth=1.5,label=nothing)
hline!([1.0],linestyle=:solid,color=:black,linewidth=1.5,label=nothing)

@load "../data_stdp/stdp_fig5a_5hz.jld2" means deltas delays rels
scatter!(delays,reverse(rels),markercolor=:purple,markersize=1.5,markeralpha=0.15,markerstrokewidth=0.5,markershape=:circle,label=nothing)
filtered_trace = savitskygolay(Float64.(reverse(means)),20,3,0)
plot!(deltas,filtered_trace,linewidth=2.0,color=:purple,label="5 Hz.")
plot!(legend=:bottomright,background_color_legend=:transparent,foreground_color_legend=:transparent)
#savefig(hstdp,"Fig5a.pdf")
