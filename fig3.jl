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


#the figure 3
l = @layout [[a{0.3w,0.95h} b{0.08w} c{0.08w} d{0.08w} e{0.15w}]; g{0.6w} h{0.4w}]
h3 = plot(size=(1200,750),layout=l,dpi=300,minorgrid=false,grid=false,
left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=3Plots.mm,guidefont=(12, :black))
plot!(h3[1],title = "A", titleloc = :left, titlefont = font(18))
plot!(h3[6],title = "B", titleloc = :left, titlefont = font(18))
plot!(h3[7],title = "C", titleloc = :left, titlefont = font(18))
for i in 2:5
    plot!(h3[i],yformatter=_->"",grid=false,minogrid=false);
end
plot!(h3[6],axis=false)
plot!(minorgrid=false)

## A
bpvd, bpvs, spks = bpn_v_dend,bpn_v_soma,bpn_spikes 
#single neuron response (m.p.)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
vspans(h3,1:1,timestamps,pattern,(3.,8.),0.2)
vspans(h3,2:2,timestamps,pattern,(3.0,3.4),0.2)
vspans(h3,3:3,timestamps,pattern,(3.9,4.2),0.2)
vspans(h3,4:4,timestamps,pattern,(7.8,8.1),0.2)
vspans(h3,5:5,timestamps,pattern,(17.0,19.0),0.2)
dt_dend = fit(UnitRangeTransform, bpvd, dims=1)
dt_soma = fit(UnitRangeTransform, bpvs, dims=1)
norm_v_dend = StatsBase.transform(dt_dend, bpvd)
norm_v_soma = StatsBase.transform(dt_soma, bpvs)
ys = ones(length(spks)).*0.95
scatter!(h3[1],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
#plot!(h3[1],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
plot!(h3[1],t,norm_v_dend,color="black", ylabel="Activity",label=L"V_d",linewidth=1.5,xlim=(3.,8.),tickfont=(12, :black))
plot!(h3[3],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
plot!(h3[2],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
#scatter!(h3[2],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h3[2],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(3.11,3.21),xticks=[3.11,3.21])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
#scatter!(h3[3],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h3[3],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(3.97,4.07),xticks=[3.97,4.07])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
plot!(h3[4],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
#scatter!(h3[4],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h3[4],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(7.85,7.95),xticks=[7.85,7.95])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
#plot!(h3[5],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h3[5],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h3[5],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(17.4,19.),xticks=[17.4,19.])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
for ii in 1:5
    plot!(h3[ii],xlabel="Time (s)",guidefont=(12,:black),tickfont=(12, :black))
end
plot!()

using LaTeXStrings
#C histogram
histogram!(h3[7],w_init,bins=20,alpha=0.55, color="black",label="t = 0 s.",guidefont=font(12),tickfont = (12, :black),markersize=0.0,markerstrokewidth=0.0,markeralpha=0.0)

histogram!(h3[7],w,bins=50,alpha=0.55, color="purple",ylabel="counts",xlabel="w",label="Spike trace-based",markersize=0.0,markerstrokewidth=0.0,markeralpha=0.0,
yticks=[0,250,500,750],ylim=(0,750),legend=:topright,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=10,minorgrid=false)

histogram!(h3[7],bp_w,bins=50,alpha=0.55, color="blue",ylabel="counts",xlabel="w",label="Ca-based",markersize=0.0,markerstrokewidth=0.0,markeralpha=0.0,
yticks=[0,250,500,750],ylim=(0,750),legend=:topright,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=10,minorgrid=false)

histogram!(h3[7],bpn_w,bins=50,alpha=0.55, color="green",ylabel="counts",xlabel="w",label="Ca/NMDAR-based",markersize=0.0,markerstrokewidth=0.0,markeralpha=0.0,
yticks=[0,250,500,750],ylim=(0,750),legend=:topright,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=10,minorgrid=false)

vline!(h3[7],[0.0],linewidth=2.0,linecolor=:black,markersize=0.0,markerstrokewidth=0.0,markeralpha=0.0,label="",linestyle=:dash)
plot!()
plot!(h3[6],axis=false)
savefig("fig3.png")