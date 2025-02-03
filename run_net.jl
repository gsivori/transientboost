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

using Plots
using JLD2
#plotting parameters
N = 500;
excitatory_n = 400;
tfactor = 1.0;
sim_length = 10000.0 * tfactor;
sim_δt = 0.1;
sim_steps = Int(sim_length/sim_δt);
ticks = 1000.0 * tfactor;
bins = 50;
dpi = 150;
box_width = 250;
box_len = box_width*sim_δt;
Nbins = Int(round(sim_length/sim_δt/box_width,digits=2));
k = zeros(Nbins,N);
pss = zeros(Int(sim_length/sim_δt),N);
t = collect(range(0.0, stop = sim_length, length=Int(100000 * tfactor))) ./ 1000.0;
default(legendfontsize = 12, guidefont = (12, :black), guide="", tickfont = (10, :gray), framestyle = nothing, yminorgrid = true, xminorgrid = true, size=(1800,1200), dpi=150);

#helper functions
include("funs.jl");

seed = 2023;
#simulation seeds 2003-2023 were used
include("net.jl");
spikes, ns, v_dend, v_soma, gain_mod, g, κ, w, poisson_in = run_network(sim_length, N, seed)
(inputs,pattern,pats) = poisson_in;
(Pi, Pr, csid) = w;
(ge, gi, gcsd) = g;

#to save
sim_name = "newest_10sec_" #change this to appropriate name
filepathname = string("jlds/",sim_name,seed,"_pats.jld2")
#saving patter input separately
@save filepathname pats
#to load
@load filepathname spikes ns v_dend v_soma gain_mod w g κ pattern
#(inputs,pattern,pat1,pat2,pat3) = poisson_in
(Pi, Pr, csid) = w #for 500-N JLD2s
#(w_in, w_r, Pi, Pr, csid) = w # for 1000-N
(ge, gi, gcd) = g #for newest JL2s as of Mar-9, otherwise use gcd

#pre-processing
#timestamps
timestamps = [(pat_time*sim_δt/1000.0,round(pattern[index+1][2]*sim_δt,digits=1)/1000.0) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)];
push!(timestamps,(timestamps[end][2],sim_length));

#network spike times
vals, y = rasters(spikes,ns);
xs,ys, grouping = groupbynat(vals,y,sign.(sum(Matrix(Pr[1]),dims=2)),sim_δt);
#output spike bins
pss = bin_netspikes(spikes,Nbins,box_width,sim_steps,sim_δt);
#for pattern colors
patcl = palette(:default);
cln = maximum(pattern)[1];
patcl = patcl[1:cln];

gr(yminorgrid=false,ytickfont=font(12),xtickfont=font(12),guidefont=font(12),legendfontsize=12,xminorgrid=false,minorgrid=false,
yguidefontsize=12,xguidefontsize=12,left_margin=5Plots.mm,right_margin=5Plots.mm,bottom_margin=3Plots.mm,dpi=150)

#Fig7B

#3.78,3.9
window = (9.75,10.)
gr(markersize=3.,markershape=:vline,legend=false, markercolor=:black, markerstrokewidth=1.0, markeralpha=1.0, linewidth=1.0)
h0 = plot(xlabel="Time (s)", ylabel="Neuron ID",xlim=window,ylim=(0.5,N+0.5),size=(800,600))
vspans(h0, timestamps, pattern, window,0.4)
#for clusters
cls = []
for cl in palette(:Set3_3)
    push!(cls,cl)
end
cls = repeat(cls,20)
for (i,cid) in enumerate(csid)
    hspan!(h0,[cid[1],cid[end]],fillalpha=0.4,color=cls[i],linewidth=0.0,alpha=0.0,label=nothing)
end
scatter!(h0,xs,ys, group=grouping, markercolor=[:brown :black],)
plot!()
savefig("rastersv3.png")

#heatmap
l = @layout [a b]
heats = plot(layout=l,size=(1000,500))
maxvalue=5.0
heatmap!(heats[1],Matrix(Pr[1]),cbar=true,color=:RdBu,clims=(-maxvalue,maxvalue))
heatmap!(heats[2],Matrix(Pr[end]),cbar=false,color=:RdBu,clims=(-maxvalue,maxvalue))


Matrix(Pr[end][csid[7],csid[7]])[:]


hass = plot(size=(1000,800))
for cell in csid[7] 
    plot!(hass,t[17100:19500],gain_mod[cell,17100:19500])
end

Plots.histogram(nonzeros(Pr[3][csid[7],csid[7]]),bins=100)
Plots.histogram(nonzeros(Pr[end][csid[7],csid[7]]),bins=100)

# on pat


#find tuned clusters
pattimes = []
for k in 1:3
    pat_end = [(i,j) for (i,j) in reverse(pattern) if i==k][1]
    push!(pattimes,pat_end)
end
    #pat_i,pat_start = [j for (i,j) in pattern if i==k break]
    #push!(pattimes,(pat_i,pat_start))

pat = [j for (i,j) in pattern if i==1] #1=green, 2=blue, 3=red
pat *= sim_δt
pat_time = 100.0
limits=(pat[end]-50.0,pat[end]+pat_time+50.0) ./ 1000.0
#find highly-spikers in this pattern
cluster_spikes = []
cluster_cis = []
for ci in 1:N
    cispks = []
    for spk in spikes[ci,:]
        if spk >= pat[end] && spk <= pat[end]+pat_time
            push!(cispks,spk)
        end
    end
    if ~isempty(cispks)
        push!(cluster_cis,Int64.(ci))
        push!(cluster_spikes,Float64.(cispks))
    end
end
true_ids = findall(length.(cluster_spikes) .> 4)
cluster_spikes = cluster_spikes[true_ids]
cluster_cis = cluster_cis[true_ids]

cluster = deepcopy(cluster_cis)
#excitatory cells gain change
h1 = plot()
for ci in cluster[cluster .<= excitatory_n]
    plot!(h1,t,gain_mod[ci,:],linewidth=2.0,color=:black,alpha=0.3,
    markersize=0.0,markeralpha=0.0,markerstrokewidth=0.0)
end
vspans(h1, timestamps, pattern, 0.15)
plot!(xlim=limits,legend=false,tickfont = (14, :gray),left_margin = 7Plots.mm,
    guidefont=(14,:black),xlabel="Time (ms)", ylabel="Ca - μCa (a.u.)")
#inhibitory cells gain change
h2 = plot()
for ci in cluster[cluster .> excitatory_n]
    plot!(h2,t,gain_mod[ci,:],linewidth=2.0,color=:black,alpha=0.3,
    markersize=0.0,markeralpha=0.0,markerstrokewidth=0.0)
end
vspans(h2, timestamps, pattern, 0.15)
plot!(xlim=limits,legend=false,tickfont = (14, :gray),left_margin = 7Plots.mm,
    guidefont=(14,:black),xlabel="Time (ms)", ylabel="Ca - μCa (a.u.)")
#excitatory cells Ca change
h3 = plot()
for ci in cluster[cluster .<= excitatory_n]
    plot!(h3,t,Ca[ci,:],linewidth=2.0,color=:black,alpha=0.3,
    markersize=0.0,markeralpha=0.0,markerstrokewidth=0.0)
end
vspans(h3, timestamps, pattern, 0.15)
plot!(xlim=limits)
#inhibitory cells Ca change
h4 = plot()
for ci in cluster[cluster .> excitatory_n]
    plot!(h4,t,Ca[ci,:],linewidth=2.0,color=:black,alpha=0.3,
    markersize=0.0,markeralpha=0.0,markerstrokewidth=0.0)
end
vspans(h4, timestamps, pattern, 0.15)
plot!(xlim=limits)

#cluster and maximum strengths
cluster_exc = deepcopy(cluster_cis)
cluster_exc = cluster_exc[cluster_exc .<= excitatory_n]
#cluster heatmap after training max values
w_r_end = deepcopy(Pr[end])
aux_mat = Matrix(Pr[end][cluster_exc,cluster_exc])
maxvalue=maximum(aux_mat)
#now cluster before training
w_r_init = deepcopy(Pr[1])

#cluster heatmap before training
aux_mat = Matrix(w_r_init[cluster_cis,cluster_cis])
nrow,ncol = size(aux_mat)
heatmap(aux_mat,xticks=(1:5:nrow,string.(cluster_cis[1:5:ncol])),yticks=(1:5:ncol,string.(cluster_cis[1:5:ncol])),
    cbar=true,clims=(-maxvalue,maxvalue),color=:balance,fill_z=aux_mat, #) uncomment
    guidefont = (1, :black), guide="", tickfont = (4, :gray),size=(500,500)) #comment
#ann = [(i,j,text(round(aux_mat[j,i],digits=2),4,:white,:center)) # change number for hex fontsize
    #for i in 1:ncol for j in 1:nrow if i != j && aux_mat[j,i] != 0.0]
#annotate!(ann, linecolor=:white)
savefig(string(figures_string,"brown_t0.pdf"))

w_r_end = deepcopy(Pr[end])

#cluster heatmap after training
aux_mat = Matrix(w_r_end[cluster_cis,cluster_cis])
nrow,ncol = size(aux_mat)
heatmap(aux_mat,xticks=(1:5:nrow,string.(cluster_cis[1:5:ncol])),yticks=(1:5:ncol,string.(cluster_cis[1:5:ncol])),
    cbar=true,clims=(-maxvalue,maxvalue),color=:balance,fill_z=aux_mat,
    guidefont = (1, :black), guide="", tickfont = (4, :gray),size=(500,500)) # same as above
#ann = [(i,j,text(round(aux_mat[j,i],digits=2),7,:white,:center))
#    for i in 1:ncol for j in 1:nrow if i != j && aux_mat[j,i] != 0.0]
#annotate!(ann, linecolor=:white)
savefig(string(figures_string,"brown_tf.pdf"))

#cluster heatmap after training
aux_mat = Matrix(w_r_init)
nrow,ncol = size(aux_mat)
heatmap(aux_mat,cbar=true,clims=(-maxvalue,maxvalue),color=:balance,fill_z=aux_mat,
    guidefont = (10, :black), guide="", tickfont = (10, :gray)) # same as above
savefig(string(figures_string,"all_t0.pdf"))

#cluster heatmap after training
aux_mat = Matrix(w_r_end)
nrow,ncol = size(aux_mat)
heatmap(aux_mat,cbar=true,clims=(-maxvalue,maxvalue),color=:balance,fill_z=aux_mat,
    guidefont = (10, :black), guide="", tickfont = (10, :gray)) # same as above
savefig(string(figures_string,"all_tf.pdf"))


green_inh = cluster_cis[cluster_cis .> excitatory_n]
cluster = [collect(1:14); collect(114:130); collect(220:230); green_inh]
fontsize = 9 #change depending on how many cells
#cluster = collect(220:230)
@gif for ti ∈ 1:size(P)[1]
    aux_mat = Matrix(deepcopy(Pr[ti][cluster,cluster]))
    nrow,ncol = size(aux_mat)
    heatmap(aux_mat,xticks=(1:nrow,string.(cluster)),yticks=(1:ncol,string.(cluster)),
        cbar=true,clims=(-maxvalue,maxvalue),color=:balance,fill_z=aux_mat,
        guidefont = (fontsize, :black), guide="", tickfont = (fontsize, :gray)) # same as above
    ann = [(i,j,text(round(aux_mat[j,i],digits=2),fontsize,:white,:center))
        for i in 1:ncol for j in 1:nrow if i != j && aux_mat[j,i] != 0.0]
    annotate!(ann, linecolor=:white)
end every 1

#E-E
anim = @animate for ti ∈ 1:size(Pr)[1]
    aux_vec = vcat(nonzeros(Pr[ti][1:excitatory_n,1:excitatory_n]) ...)
    Plots.histogram(aux_vec,bins=100,color=:purple,legend=false,size=(1000,500),ylim=(0,15000),yticks=[],formatter=:plain, xlim=(0,10))
end
gif(anim,string(figures_string,"E-E.gif"),fps=8)

#E-I
anim = @animate for ti ∈ 1:size(Pr)[1]
    aux_vec = vcat(nonzeros(Pr[ti][1:excitatory_n,excitatory_n+1:N]) ...)
    Plots.histogram(aux_vec,bins=200,color=:blue,legend=false,size=(1000,500),ylim=(0,6000),yticks=[],formatter=:plain, xlim=(0,10))
end every 1
gif(anim,string(figures_string,"E-I.gif"),fps=8)

#I-E
anim = @animate for ti ∈ 1:size(Pr)[1]
    aux_vec = vcat(nonzeros(Pr[ti][excitatory_n+1:N,1:excitatory_n]) ...)
    Plots.histogram(aux_vec,bins=200,color=:brown,legend=false,size=(1000,500),ylim=(0,12500),yticks=[],formatter=:plain, xlim=(-10,0))
end every 1
gif(anim,string(figures_string,"I-E.gif"),fps=8)

#I-I
anim = @animate for ti ∈ 1:size(Pr)[1]
    aux_vec = vcat(nonzeros(Pr[ti][excitatory_n+1:N,excitatory_n+1:N]) ...)
    Plots.histogram(aux_vec,bins=100,color=:red,legend=false,size=(1000,500),ylim=(0,3000),yticks=[],formatter=:plain, xlim=(-10,0))
end every 1
gif(anim,string(figures_string,"I-I.gif"),fps=8)

#excitatory cells vd change
h5 = plot()
for ci in cluster[cluster .<= excitatory_n]
    plot!(h5,t,v_dend[ci,:],linewidth=2.0,color=:black,alpha=0.3,
    markersize=0.0,markeralpha=0.0,markerstrokewidth=0.0)
end
vspans(h5, timestamps, pattern, 0.15,legend=false)
#inhibitory cells vd change
h6 = plot()
for ci in cluster[cluster .> excitatory_n]
    plot!(h6,t,v_dend[ci,:],linewidth=2.0,color=:black,alpha=0.3,
    markersize=0.0,markeralpha=0.0,markerstrokewidth=0.0)
end
vspans(h6, timestamps, pattern, 0.15,legend=false)


#excitatory cells vs change
#limits = (15000.0,18050.0)
psss = sum(pss[:,cluster[cluster .< excitatory_n]],dims=2)
l = @layout [a; c{0.3h}]
h7 = plot(legend=false,left_margin = 7Plots.mm,markersize=0.0,markeralpha=0.0,
    markerstrokewidth=0.0,layout=l,sharex=true,tickfont = (14, :gray),guidefont=(14,:black))
for ci in cluster[cluster .< excitatory_n]
    plot!(h7[1],t,v_soma[ci,:],linewidth=2.0,color=:black,alpha=0.3,
    markersize=0.0,markeralpha=0.0,markerstrokewidth=0.0)
end
plot!(h7[1],xlim=limits,xlabel=nothing,ylabel="Vsoma (mV)"),
bar!(h7[2],t,psss,xlabel="Time (ms)",ylabel="Spk. count",xlim=limits,
    ylim=(0,maximum(psss)), yminorgrid=true)
vspans(h7, timestamps, pattern, 0.15)

#inhibitory cells vs change
psss = sum(pss[:,cluster[cluster .> excitatory_n]],dims=2)
l = @layout [a; c{0.3h}]
h8 = plot(legend=false,left_margin = 7Plots.mm,markersize=0.0,markeralpha=0.0,
    markerstrokewidth=0.0,layout=l,sharex=true,tickfont = (14, :gray),guidefont=(14,:black))
for ci in cluster[cluster .> excitatory_n]
    plot!(h8[1],t,v_soma[ci,:],linewidth=2.0,color=:black,alpha=0.3,
    markersize=0.0,markeralpha=0.0,markerstrokewidth=0.0)
end
plot!(h8[1],xlim=limits,xlabel=nothing,ylabel="Vsoma (mV)"),
bar!(h8[2],t,psss,xlabel="Time (ms)",ylabel="Spk. count",xlim=limits,
    ylim=(0,maximum(psss)), yminorgrid=true)
vspans(h8, timestamps, pattern, 0.15)
