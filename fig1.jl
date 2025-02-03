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

#the figure 1
l = @layout [[a{0.7h}; [a{1.5w} b{0.7w} c{0.7w} d{0.7w} e{1.1w}]] [grid(1,2){0.5w,0.2h}; d{1.0w,0.15h}; e{1.0w,0.18h}] grid(4,1){1.0h}]
h1 = plot(size=(1200,900),layout=l,dpi=600,minorgrid=false,grid=false,
left_margin=3Plots.mm,right_margin=1Plots.mm,bottom_margin=2Plots.mm,guidefont=(12, :black))

plot!(h1[1],title = "A", titleloc = :left, titlefont = font(18),grid=false)
plot!(h1[2],title = "D", titleloc = :left, titlefont = font(18)) #2---5
for i in 3:6
    plot!(h1[i],yformatter=_->"",grid=false,minogrid=false);
end
plot!(h1[3],title = "", titleloc = :left, titlefont = font(18))
plot!(h1[4],title = "", titleloc = :left, titlefont = font(18))
plot!(h1[5],title = "", titleloc = :left, titlefont = font(18))
plot!(h1[6],title = "", titleloc = :left, titlefont = font(18))
plot!(h1[7],title = "B", titleloc = :left, titlefont = font(18)) #2---5
plot!(h1[8],yformatter=_->"",grid=false,minogrid=false);
plot!(h1[8],title = "", titleloc = :left, titlefont = font(18))
plot!(h1[9],title = "C", titleloc = :left, titlefont = font(18))
plot!(h1[10],title = "E", titleloc = :left, titlefont = font(18))
plot!(h1[11],title = "F", titleloc = :left, titlefont = font(18))
plot!(h1[12],title = "", titleloc = :left, titlefont = font(18))

plot!(h1[10],grid=false,minogrid=false);
#
plot!(h1[1],grid=false,showaxis=false,minorgrid=false)

#B
box_width = 100
box_len = box_width*sim_δt
Nbins = Int(round(sim_length/sim_δt/box_width,digits=2))
window = (4.8,6.2)
windowticks = [5.0,6.0]
#syns_comp = plot(size=(600,400),yminorgrid=false,xminorgrid=false,ytickfont=font(12),xtickfont=font(12),
#    guidefont=font(12),tickfont = (12, :black),yguidefontsize=12,xguidefontsize=12,legend=false,
#    left_margin=2Plots.mm,right_margin=3Plots.mm,bottom_margin=3Plots.mm,sharex=true,dpi=300,layout=l1a)
vals = []
y = []
for ci = 1:n_in
    times = view(presyns,ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0)
    push!(y,ci*ones(length(times)))
end
xs, ys, grouping = groupbypat(vals,y,pattern,sim_δt)
#input spikes pss
pss_in = zeros(Int(sim_length/sim_δt))
pss_ci = zeros(n_in,Int(sim_length/sim_δt))
for ci = 1:n_in
    pss_ci[ci,:] = bin_spikes(view(vals,ci,:)[1] .* sec_to_ms,Nbins,box_width,sim_steps,sim_δt)
end
pss_in = moving_average(sum(pss_ci,dims=1)[1,:],5000)
vspans(h1,7:8,timestamps,pattern,window,0.2)
scatter!(h1[7],xs,ys,group=grouping,markercolor=[:black :blue :red :green],ylabel="Input ID",ylim=(25,75),xlim=window,xticks=windowticks,
    xlabel="",yminorgrid=false,guidefont=font(12),tickfont = (12, :black),yticks=[25,50,75],
    markersize=5.0,markershape=:vline,legend=false, markerstrokewidth=1.0, markeralpha=1.0)
plot!(h1[7],t,pss_in,xlabel="Time (s)",xlim=window,xticks=windowticks,markersize=0.0,legend=false,
color=:brown,linewidth=2.0,guidefont=font(12),tickfont = (12, :black),minor_grid=false,grid=false)


vals = []
y = []
for ci = 1:n_in
    times = view(postsyns,ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0)
    push!(y,ci*ones(length(times)))
end
xs, ys, grouping = groupbypat(vals,y,pattern,sim_δt)
#input spikes pss
pss_in = zeros(Int(sim_length/sim_δt))
pss_ci = zeros(n_in,Int(sim_length/sim_δt))
for ci = 1:n_in
    pss_ci[ci,:] = bin_spikes(view(vals,ci,:)[1] .* sec_to_ms,Nbins,box_width,sim_steps,sim_δt)
end
pss_in = moving_average(sum(pss_ci,dims=1)[1,:],5000)
scatter!(h1[8],xs,ys,group=grouping,markercolor=[:black :blue :red :green],xlabel="",ylabel="",ylim=(25,75),xlim=window,xticks=windowticks,
        yminorgrid=false,guidefont=font(12),tickfont=(12, :black),yticks=[25,50,75],
        markersize=5.0,markershape=:vline,legend=false,markerstrokewidth=1.0,markeralpha=1.0)
plot!(h1[8],t,pss_in,xlabel="Time (s)",xlim=window,xticks=windowticks,markersize=0.0,legend=false,
color=:brown,linewidth=2.0,guidefont=font(12),tickfont=(12, :black),minor_grid=false,grid=false)
plot!()


#C 
using LaTeXStrings
syn_w = collect(Float64, 0.0:0.01:8.0)
alpha_fun = ζ.(syn_w,1e-4,.75)
plot!(h1[9],syn_w,alpha_fun,color="black",legend=false,linewidth=2.5,xlim=(-0.1,8.),xticks=[0,2,4,6,8],yticks=[0.0,0.5,1.0],guidefont=font(12),tickfont = (12, :black),yguidefontsize=16,
xlabel="magnitude(w)",ylabel=L"$\mathtt{\zeta \;\left(\| w(i)\|\right)}$",markersize=0.0, markerstrokewidth=0.0,markeralpha=0.0)

#D
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
#l = @layout [a b{0.15w} c{0.15w} d{0.15w} e{0.15w}]
#h1 = plot(minorgrid=false,grid=false,sharex=false,sharey=true,layout=l, legend=:bottomright,
#left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=3Plots.mm,size=(800,400),ylim=(0,1))
vspans(h1,2:2,timestamps,pattern,(1.2,3.3),0.2)
vspans(h1,3:3,timestamps,pattern,(2.2,2.8),0.2)
vspans(h1,4:4,timestamps,pattern,(2.2,2.8),0.2)
vspans(h1,5:5,timestamps,pattern,(2.7,3.5),0.2)
vspans(h1,6:6,timestamps,pattern,(17.0,19.0),0.2)
dt_dend = fit(UnitRangeTransform, v_dend, dims=1)
dt_soma = fit(UnitRangeTransform, v_soma, dims=1)
norm_v_dend = StatsBase.transform(dt_dend, v_dend)
norm_v_soma = StatsBase.transform(dt_soma, v_soma)
scatter!(h1[2],spikes,ones(length(spikes)).*0.95,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
#plot!(h1[1],t,norm_v_soma,color="purple",xlabel="Time (s)", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
plot!(h1[2],t,norm_v_dend,color="black", ylabel="Activity",label=L"V_d",linewidth=1.5,xlim=(1.2,3.22))
plot!(h1[3],t,norm_v_soma,color="purple",xlabel="Time (s)", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h1[3],spikes,ones(length(spikes)).*0.95,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h1[3],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(2.4,2.44),xticks=[2.40,2.44])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
plot!(h1[3],yformatter=_->"")
plot!(h1[4],t,norm_v_soma,color="purple",xlabel="Time (s)", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h1[4],spikes,ones(length(spikes)).*0.95,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h1[4],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(2.64,2.70),xticks=[2.64,2.70])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
plot!(h1[4],yformatter=_->"")
plot!(h1[5],t,norm_v_soma,color="purple",xlabel="Time (s)", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h1[5],spikes,ones(length(spikes)).*0.95,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h1[5],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(3.12,3.16),xticks=[3.12,3.16])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
plot!(h1[5],yformatter=_->"")
#plot!(h1[5],t,norm_v_soma,color="purple",xlabel="Time (s)", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h1[6],spikes,ones(length(spikes)).*0.95,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h1[6],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(17.4,19.),xticks=[17.4,18.8])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
plot!(h1[6],yformatter=_->"")
for ii in 2:6
    plot!(h1[ii],xlabel="Time (s)",guidefont=font(12),tickfont=(12, :black))
end
plot!()

#F  h1 11<-->14
#how does the model learn figure. (A and B)
vals = []
y = []
for ci = 1:n_in
    times = view(postsyns,ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times.*ms_to_sec)
    push!(y,ci*ones(length(times)))
end
xs, ys, grouping = groupbypat(vals,y,pattern,sim_δt)

id1 = 163
id2 = 621
id1_ids = findall(ys .== id1)
id1_xs = xs[id1_ids]
id1_ys = ys[id1_ids]
id2_ids = findall(ys .== id2)
id2_xs = xs[id2_ids]
id2_ys = ys[id2_ids]
tstart = 58000
tend = 65000
window = (5.8,6.5)
windowticks = [5.8,6.0,6.2,6.4]

vspans(h1,11:14, timestamps,pattern,window,0.25)
scatter!(twinx(h1[14]),id1_xs,id1_ys,group=grouping[id1_ids],markeralpha=1.0,markercolor="hotpink", xlabel="",
    markersize=10.0,markerstrokewidth=1.5,markershape=:vline,ylabel="",yticks=[],legend=false,
    yminorgrid=false,xticks=[],xlim=window,yflip=true,)
scatter!(twinx(h1[14]),id2_xs,id2_ys,group=grouping[id2_ids],markeralpha=1.0,markercolor="teal", xlabel="",
    markersize=10.0,markerstrokewidth=1.5,markershape=:vline,ylabel="",yticks=[],legend=false,
    yminorgrid=false,xticks=[],xlim=window,yflip=false)
plot!(h1[11],t[tstart:tend],v_dend[tstart:tend],color=:black,xlabel="", ylabel="M.p. (mV)",alpha=1.0,label=L"\mathtt{v_{d}}",
    yminorgrid=false,linewidth=1.5,legend=:topright,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=16,
    guidefont=font(12),tickfont = (12, :black))
plot!(h1[11],t[tstart:tend],v_soma[tstart:tend],color="purple",label=L"\mathtt{v_{s}}",alpha=1.0,linewidth=1.5,yticks=[-10,-30,-50,-70],ylim=(-75,-5),ygrid=false)
plot!(h1[12],t[tstart:tend],spk_train[tstart:tend],color="blue",legend=:top,linewidth=1.5,ylabel="Spike traces",label=L"\mathtt{Y}",minorgrid=false)
plot!(h1[12],t[tstart:tend],(spk_train .- gain_mod)[tstart:tend],color="indigo",linewidth=1.5,label=L"\mathtt{\overline{Y}}",legendfontsize=14,
    yticks=[.25,.75,1.25,1.75],legend=:topright,background_color_legend=:transparent,foreground_color_legend=:transparent,
    guidefont=font(12),tickfont = (12, :black),minorgrid=false)
plot!(h1[13],t[tstart:tend],gain_mod[tstart:tend],color="midnightblue",label="",linewidth=1.5,legend=false,ylabel=L"\mathtt{e(t)}",
    guidefont=font(16),tickfont = (12, :black),minorgrid=false)
plot!(h1[14],t[tstart:tend],P[id1,tstart:tend],color="hotpink",linewidth=1.5,label=L"\mathtt{ID\;%$id1}",
    legend=:right,ylabel="w",xlim=window)
plot!(h1[14],t[tstart:tend],P[id2,tstart:tend],color="teal",linewidth=1.5,xlabel="Time (s)",label=L"\mathtt{ID\;%$id2}",
    legend=:right,xticks=windowticks,minorgrid=false,xlim=window,
    background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14,guidefont=font(12),tickfont = (12, :black))
#to save
for i=11:13; plot!(h1[i],xformatter=_->""); end;
plot!()


#E histogram
histogram!(h1[10],w_init,bins=20,alpha=0.55, color="black",label="t = 0 s.",guidefont=font(12),tickfont = (12, :black),markersize=0.0,markerstrokewidth=0.0,markeralpha=0.0)
histogram!(h1[10],w,bins=50,alpha=0.55, color="purple",ylabel="counts",xlabel="w",label="t = 20 s.",markersize=0.0,markerstrokewidth=0.0,markeralpha=0.0,
yticks=[0,250,500,750],ylim=(0,750),legend=:topleft,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=12,minorgrid=false)
vline!(h1[10],[0.0],linewidth=2.0,linecolor=:black,markersize=0.0,markerstrokewidth=0.0,markeralpha=0.0,label="",linestyle=:dash)
#plot!(h1[10],left_margin=3Plots.mm,bottom_margin=2Plots.mm)
plot!()



savefig(h1,"Fig1.png")


tstart = 61300
tend = 64500
plot(t,cumsum(gain_mod),color="midnightblue",linewidth=1.5,legend=:topleft,grid=false,minorgrid=false,
    ylabel=L"\mathtt{cusum(t)}",label=L"\mathtt{cusum(t) = \sum_{t=0.}^{t=20.}\;e(t_i)}",
    guidefont=font(15),tickfont = (12, :black),xlabel="Time (s)",
    left_margin=4Plots.mm,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=16,ylim=(-5,400))

savefig("cusum.png")