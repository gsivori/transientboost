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
l = @layout [grid(2,3)]
gr(minorgrid=false,grid=false,background_color_legend=:transparent,foreground_color_legend=:transparent,markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0,
legendfontsize=16,left_margin=7Plots.mm,right_margin=3Plots.mm,bottom_margin=6Plots.mm,guidefont=(14, :black),tickfont=(14, :black))
hsupp1 = plot(size=(1500,800),layout=l,dpi=600)
#plot!(hsupp2[1],title = "A", titleloc = :left, titlefont = font(18))
#plot!(hsupp2[2],title = "B", titleloc = :left, titlefont = font(18))
#plot!(hsupp2[3],title = "C", titleloc = :left, titlefont = font(18))
#plot!(hsupp1[4],title = "D", titleloc = :left, titlefont = font(18))
for ii in 4:6
    plot!(hsupp1[ii],xlabel="w")
end
for ii in [1,4]
    plot!(hsupp1[ii],ylabel="Density")
end


green_ins = any(pats[3],dims=2)[:]
blue_ins = any(pats[1],dims=2)[:]
red_ins = any(pats[2],dims=2)[:]

green_palette = ["#575761","#7BB78A","#67A267","#466D22"]
red_palette = ["#575761","#FBCCC6","#E94C49","#A30015"]
blue_palette = ["#575761","#CDCECA","#53BCDF","#255ED0"]

using StatsPlots,KernelDensity
#1,1
dens = kde(P[green_ins,200000][P[green_ins,200000].>0],boundary=(-0.,10.),npoints=100)
plot!(hsupp1[1],dens,color=green_palette[4],label="t = 20 s.",linewidth=2.5)
dens = kde(P[green_ins,120000][P[green_ins,120000].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[1],dens,color=green_palette[3],label="t = 12 s.",linewidth=2.5)
dens = kde(P[green_ins,60000][P[green_ins,60000].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[1],dens,color=green_palette[2],label="t = 6 s.",linewidth=2.5)
dens = kde(P[green_ins,1][P[green_ins,1].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[1],dens,color=green_palette[1],label="t = 0 s.",linewidth=2.5)
plot!(hsupp1[1],xlim=(-.1,8.5))
#1,2
dens = kde(P[red_ins,200000][P[red_ins,200000].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[2],dens,color=red_palette[4],label="t = 20 s.",linewidth=2.5)
dens = kde(P[red_ins,120000][P[red_ins,120000].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[2],dens,color=red_palette[3],label="t = 12 s.",linewidth=2.5)
dens = kde(P[red_ins,60000][P[red_ins,60000].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[2],dens,color=red_palette[2],label="t = 6 s.",linewidth=2.5)
dens = kde(P[red_ins,1][P[red_ins,1].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[2],dens,color=red_palette[1],label="t = 0 s.",linewidth=2.5)
plot!(hsupp1[2],xlim=(-.1,8.5))
#1,3
dens = kde(P[blue_ins,200000][P[blue_ins,200000].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[3],dens,color=blue_palette[4],label="t = 20 s.",linewidth=2.5)
dens = kde(P[blue_ins,120000][P[blue_ins,120000].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[3],dens,color=blue_palette[3],label="t = 12 s.",linewidth=2.5)
dens = kde(P[blue_ins,60000][P[blue_ins,60000].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[3],dens,color=blue_palette[2],label="t = 6 s.",linewidth=2.5)
dens = kde(P[blue_ins,1][P[blue_ins,1].>0],boundary=(0.,10.),npoints=100)
plot!(hsupp1[3],dens,color=blue_palette[1],label="t = 0 s.",linewidth=2.5)
plot!(hsupp1[3],xlim=(-.1,8.5))
#2,1
dens = kde(P[green_ins,200000][P[green_ins,200000].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[4],dens,color=green_palette[4],label="t = 20 s.",linewidth=2.5)
dens = kde(P[green_ins,120000][P[green_ins,120000].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[4],dens,color=green_palette[3],label="t = 12 s.",linewidth=2.5)
dens = kde(P[green_ins,60000][P[green_ins,60000].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[4],dens,color=green_palette[2],label="t = 6 s.",linewidth=2.5)
dens = kde(P[green_ins,1][P[green_ins,1].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[4],dens,color=green_palette[1],label="t = 0 s.",linewidth=2.5)
plot!(hsupp1[4],xlim=(-8.5,0.1),xlabel="w")
#2,2
dens = kde(P[red_ins,200000][P[red_ins,200000].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[5],dens,color=red_palette[4],label="t = 20 s.",linewidth=2.5)
dens = kde(P[red_ins,120000][P[red_ins,120000].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[5],dens,color=red_palette[3],label="t = 12 s.",linewidth=2.5)
dens = kde(P[red_ins,60000][P[red_ins,60000].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[5],dens,color=red_palette[2],label="t = 6 s.",linewidth=2.5)
dens = kde(P[red_ins,1][P[red_ins,1].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[5],dens,color=red_palette[1],label="t = 0 s.",linewidth=2.5)
plot!(hsupp1[5],xlim=(-8.5,0.1),xlabel="w")
#2,3
dens = kde(P[blue_ins,200000][P[blue_ins,200000].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[6],dens,color=blue_palette[4],label="t = 20 s.",linewidth=2.)
dens = kde(P[blue_ins,120000][P[blue_ins,120000].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[6],dens,color=blue_palette[3],label="t = 12 s.",linewidth=2.)
dens = kde(P[blue_ins,60000][P[blue_ins,60000].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[6],dens,color=blue_palette[2],label="t = 6 s.",linewidth=2.5)
dens = kde(P[blue_ins,1][P[blue_ins,1].<0],boundary=(-10.,0.),npoints=100)
plot!(hsupp1[6],dens,color=blue_palette[1],label="t = 0 s.",linewidth=2.5)
plot!(hsupp1[6],xlim=(-8.5,0.1))

savefig("suppfig1.png")