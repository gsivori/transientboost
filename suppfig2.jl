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

#the figure 
l = @layout [grid(4,1){0.15w} grid(4,1){0.85w}]
gr(minorgrid=false,grid=false,background_color_legend=:transparent,foreground_color_legend=:transparent,markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0,
legendfontsize=16,left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=0Plots.mm,guidefont=(14, :black),tickfont=(12, :black))
hsupp2 = plot(size=(1200,1200),layout=l,dpi=600)
plot!(hsupp2[1],title = "A", titleloc = :left, titlefont = font(18))
plot!(hsupp2[2],title = "B", titleloc = :left, titlefont = font(18))
plot!(hsupp2[3],title = "C", titleloc = :left, titlefont = font(18))
plot!(hsupp2[4],title = "D", titleloc = :left, titlefont = font(18))
for i in 1:4
    plot!(hsupp2[i],grid=false,showaxis=false,minorgrid=false)
end
plot!(hsupp2[8],xlabel="Time (s)")



pat = [j for (i,j) in pattern if i==2]
pat *= sim_δt
pat_time = 100.0
limits = (pat[end]-50.0,pat[end]+pat_time+50.0) ./ 1000.0
window = limits
windowticks = [18.8,18.85,18.90,18.95] #pattern-related

a,b = tosteps.(window)
tiwindow = a:b

times = spikes[findall(x -> window[1] .< x .< window[2],spikes)]
ys_plot = ones(length(times)) .* 140.0
vspans(hsupp2,5:5,timestamps,pattern,window,0.25)
plot!(hsupp2[5],t[tiwindow],8.0*ones(length(v_dend[tiwindow])),color=:green,linewidth=1.5,label=L"g_{csd}",
legend=:topleft,ylabel="Conductance (nS)",ylim=(0.0,150.0))
plot!(twinx(hsupp2[5]),t[tiwindow],v_dend[tiwindow],color=:black,alpha=0.8,sharex=true,xlim=window,
ylim=(-80.0,0.0),linewidth=1.5,label=L"v_{d}",ylabel="M.p. (mV)",legend=:best)
scatter!(hsupp2[5],times,ys_plot,color=:purple,markershape=:vline,markersize=10.0,markerstrokewidth=1.5,label=nothing,markeralpha=1.0)
plot!(hsupp2[5],t[tiwindow],ge[tiwindow],color=:navyblue,alpha=0.9,linewidth=1.5,label=L"g_{e}")
plot!(hsupp2[5],t[tiwindow],gi[tiwindow],color=:red,alpha=0.9,linewidth=1.5,label=L"g_{i}")

times = bp_spikes[findall(x -> window[1] .< x .< window[2],bp_spikes)]
ys_plot = ones(length(times)) .* 140.0
vspans(hsupp2,7:7,timestamps,pattern,window,0.25)
plot!(hsupp2[7],t[tiwindow],bp_gcs[tiwindow],color=:green,linewidth=1.5,label=L"g_{csd}",alpha=0.6,
legend=:topleft,ylabel="Conductance (nS)",ylim=(0.0,150.0))
plot!(twinx(hsupp2[7]),t[tiwindow],bp_v_dend[tiwindow],color=:black,alpha=0.8,xlim=window,ylim=(-80.0,0.0),
linewidth=1.5,label=L"v_{d}",ylabel="M.p. (mV)",legend=:best)
scatter!(hsupp2[7],times,ys_plot,color=:purple,markershape=:vline,markersize=10.0,markerstrokewidth=1.5,label=nothing,markeralpha=1.0)
plot!(hsupp2[7],t[tiwindow],bp_ge[tiwindow],color=:navyblue,alpha=0.9,linewidth=1.5,label=L"g_{e}")
plot!(hsupp2[7],t[tiwindow],bp_gi[tiwindow],color=:red,alpha=0.9,linewidth=1.5,label=L"g_{i}")

pat = [j for (i,j) in pattern if i==1]
pat *= sim_δt
pat_time = 100.0
limits = (pat[end]-50.0,pat[end]+pat_time+50.0) ./ 1000.0
window = (limits[1]-0.35,limits[2]+0.05)
windowticks = [19.3,19.4,19.5,19.6,19.7,19.8] #non-pattern related

a,b = tosteps.(window)
tiwindow = a:b

times = spikes[findall(x -> window[1] .< x .< window[2],spikes)]
ys_plot = ones(length(times)) .* 140.0
vspans(hsupp2,6:6,timestamps,pattern,window,0.25)
plot!(hsupp2[6],t[tiwindow],8.0*ones(length(v_dend[tiwindow])),color=:green,linewidth=1.5,label=L"g_{csd}",
legend=:topleft,ylabel="Conductance (nS)",ylim=(0.0,150.0))
plot!(twinx(hsupp2[6]),t[tiwindow],v_dend[tiwindow],color=:black,alpha=0.8,sharex=true,xlim=window,ylim=(-80.0,0.0),
linewidth=1.5,label=L"v_{d}",ylabel="M.p. (mV)",legend=:best)
scatter!(hsupp2[6],times,ys_plot,color=:purple,markershape=:vline,markersize=10.0,markerstrokewidth=1.5,label=nothing,markeralpha=1.0)
plot!(hsupp2[6],t[tiwindow],ge[tiwindow],color=:navyblue,alpha=0.9,linewidth=1.5,label=L"g_{e}")
plot!(hsupp2[6],t[tiwindow],gi[tiwindow],color=:red,alpha=0.9,linewidth=1.5,label=L"g_{i}")

times = bp_spikes[findall(x -> window[1] .< x .< window[2],bp_spikes)]
ys_plot = ones(length(times)) .* 140.0
vspans(hsupp2,8:8,timestamps,pattern,window,0.25)
plot!(hsupp2[8],t[tiwindow],bp_gcs[tiwindow],color=:green,linewidth=1.5,label=L"g_{csd}",alpha=0.6,
legend=:topleft,ylabel="Conductance (nS)",ylim=(0.0,150.0))
plot!(twinx(hsupp2[8]),t[tiwindow],bp_v_dend[tiwindow],color=:black,alpha=0.8,xlim=window,ylim=(-80.0,0.0),
linewidth=1.5,label=L"v_{d}",ylabel="M.p. (mV)",legend=:best)
scatter!(hsupp2[8],times,ys_plot,color=:purple,markershape=:vline,markersize=10.0,markerstrokewidth=1.5,label=nothing,markeralpha=1.0)
plot!(hsupp2[8],t[tiwindow],bp_ge[tiwindow],color=:navyblue,alpha=0.9,linewidth=1.5,label=L"g_{e}")
plot!(hsupp2[8],t[tiwindow],bp_gi[tiwindow],color=:red,alpha=0.9,linewidth=1.5,label=L"g_{i}")

savefig("suppfig2.png")


## traces for Ca2/NMDAR model: not used


times = bpn_spikes[findall(x -> window[1] .< x .< window[2],bpn_spikes)]
ys_plot = ones(length(times)) .* 150.0
hsupp2 = plot(ytickfont=font(10),dpi=300,minorgrid=false,grid=false,
guidefont=font(10),xtickfont=font(10),
yguidefontsize=10,legendfontsize=14,xlim=window,
left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=4Plots.mm,top_margin=1Plots.mm,
background_color_legend=:transparent,foreground_color_legend=:transparent,
size=(600,200),xlabel="Time (s)")
vspans(hsupp2,timestamps,pattern,window,0.25)
plot!(hsupp2,t,bpn_gcs,color=:green,linewidth=1.5,label=L"g_{csd}",alpha=0.6,
legend=:topleft,ylabel="Conductance (nS)",ylim=(0.0,150.0))
plot!(twinx(hsupp2),t,bpn_v_dend,color=:black,alpha=0.8,xlim=window,
ytickfont=font(10),guidefont=font(10),yguidefontsize=10,ylim=(-70.0,0.0),minorgrid=false,
xtickfont=font(10),linewidth=1.5,label=L"v_{d}",ylabel="M.p. (mV)",legend=:best,legendfontsize=14,
background_color_legend=:transparent,foreground_color_legend=:transparent)
scatter!(hsupp2,times,ys_plot,color=:purple,markershape=:vline,markersize=10.0,label=nothing,markeralpha=1.0)
plot!(hsupp2,t,bpn_ge,color=:navyblue,alpha=0.9,linewidth=1.5,label=L"g_{e}")
plot!(hsupp2,t,bpn_gi,color=:red,alpha=0.9,linewidth=1.5,label=L"g_{i}")
#plot!(hsupp2,t,bpn_gnmda,color=:violet,alpha=0.9,linewidth=1.5,label=L"g_{nmda}")
vline!(hsupp2,windowticks,color=:gray,alpha=0.25,linestyle=:dash,label="")

gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
times = bpn_spikes[findall(x -> window[1] .< x .< window[2],bpn_spikes)]
ys_plot = ones(length(times)) .* 150.0
hsupp2 = plot(ytickfont=font(10),dpi=300,minorgrid=false,grid=false,
guidefont=font(10),xtickfont=font(10),
yguidefontsize=10,legendfontsize=14,xlim=window,
left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=4Plots.mm,top_margin=1Plots.mm,
background_color_legend=:transparent,foreground_color_legend=:transparent,
size=(600,200),xlabel="Time (s)")
vspans(hsupp2,timestamps,pattern,window,0.25)
plot!(hsupp2,t,bpn_gcs,color=:green,linewidth=1.5,label=L"g_{csd}",alpha=0.6,
legend=:topleft,ylabel="Conductance (nS)",ylim=(0.0,150.0))
plot!(twinx(hsupp2),t,bpn_v_dend,color=:black,alpha=0.8,xlim=window,
ytickfont=font(10),guidefont=font(10),yguidefontsize=10,ylim=(-70.0,0.0),minorgrid=false,
xtickfont=font(10),linewidth=1.5,label=L"v_{d}",ylabel="M.p. (mV)",legend=:best,legendfontsize=14,
background_color_legend=:transparent,foreground_color_legend=:transparent)
scatter!(hsupp2,times,ys_plot,color=:purple,markershape=:vline,markersize=10.0,label=nothing,markeralpha=1.0)
plot!(hsupp2,t,bpn_ge,color=:navyblue,alpha=0.9,linewidth=1.5,label=L"g_{e}")
plot!(hsupp2,t,bpn_gi,color=:red,alpha=0.9,linewidth=1.5,label=L"g_{i}")
#plot!(hsupp2,t,bpn_gnmda,color=:violet,alpha=0.9,linewidth=1.5,label=L"g_{nmda}")
vline!(hsupp2,windowticks,color=:gray,alpha=0.25,linestyle=:dash,label="")
savefig(hsupp2,"data_figs/full_conds_other.png")