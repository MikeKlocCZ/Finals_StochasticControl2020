

#cd("/home/mike/Documents/Projects/JuliaControl/ToShare")
cd("/home/michal/Documents/Projects/JuliaControl/ToShare/QB-simpleNN")

# make your scripts automatically re-activate your project
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()


using Flux, DiffEqFlux, DiffEqSensitivity
using DifferentialEquations
using Plots
using Printf, BSON
using QuantumOptics
using Statistics
using Zygote
using StaticArrays
using LinearAlgebra
using JLD  #saving models


#################################################
#read parameters from external file
include("parameters_qb_final_150.jl")
using Main.parameters

struct Parameters
	dim::Int64
	w::Float32
	force_mag::Float32
	n_steps::Int64
	dt::Float32
	n_substeps::Int64
	gamma::Float32
end

MyParameters=Parameters(
	parameters.parameters_para["N"],
	parameters.parameters_para["w"],
	parameters.parameters_para["force_mag"],
	parameters.parameters_para["max_episode_steps"],
	parameters.parameters_para["dt"],
	parameters.parameters_para["n_substeps"],
	parameters.parameters_para["gamma"])

n_par=256


########################
using Random
Random.seed!(2)

################################
# Hamiltonian and evolution
basis=SpinBasis(1//2) #max n=dim-1
σ_z=sigmaz(basis)
σ_x=sigmax(basis)
σ_p=sigmap(basis)
σ_m=sigmam(basis)
σ_pm=σ_p*σ_m

H_0pre=MyParameters.w/2.0f0*σ_z #drift Hamiltonian
H_1pre=σ_x #control field

#to the dense regime, all real, otherwise Flux.train! complains
H_0=real(Array{Float32}(H_0pre.data))
H_1=real(Array{Float32}(H_1pre.data))

Arrayσ_p=real(Array{Float32}(σ_p.data))
Arrayσ_m=real(Array{Float32}(σ_m.data))
Arrayσ_pm=real(Array{Float32}(σ_pm.data))

I_matrix=Matrix(1.0f0I, MyParameters.dim, MyParameters.dim)
###############################################
"""
Goal: target |up>
"""
#PREFERRED
function qb_dynamics_dt!(du,u, α, t)  #du/u have a standard dimension
    ψRe = u[1:MyParameters.dim]   #(dim,)
    ψIm = u[MyParameters.dim+1:end]

    HRe=H_0.+α[1]*H_1 #size (dim,dim) ,alpha is just a single number

	Reρ=ψRe.*transpose(ψRe)+ψIm.*transpose(ψIm)
	ex_x=real(sum(diag((Arrayσ_p+Arrayσ_m)*Reρ)))

	HIm=-Arrayσ_pm/2 +ex_x*Arrayσ_m
	du[1:MyParameters.dim]=dψRe= HIm*ψRe+HRe*ψIm; #size dim)
	du[MyParameters.dim+1:end]=dψIm= HIm*ψIm-HRe*ψRe;
end

function qb_dynamics_dW!(du,u, α, t)  #last action IS NOT STORED IN u
	ψRe = u[1:MyParameters.dim]   #(dim)
	ψIm = u[MyParameters.dim+1:end]
#
	HRe=Arrayσ_m

	du[1:MyParameters.dim,:]=dψRe= HRe*ψRe; #size (dim)
	du[MyParameters.dim+1:end,:]=dψIm= HRe*ψIm;
end

################################################
#model
@info("Constructing simpler model...") #input: couple of last steps, say 4
#state-aware
state_1 = Dense(2*MyParameters.dim, 256,relu ,initb = Flux.glorot_uniform)
state_2 = Dense(256, 128,relu, initb = Flux.glorot_uniform)
state_3 = Dense(128, 64,relu, initb = Flux.glorot_uniform)
state_4 = Dense(64, 1,softsign,initb = Flux.glorot_uniform)


model=Chain(state_1, state_2, state_3,state_4)

using BSON: @load
@load string("model_state-in_parallel_END.bson")  p1 re opt
#p1, re = Flux.destructure(model)

###############################################
# initial state anywhere on the Bloch sphere
u0 = Array{Float32,2}(undef, 2*MyParameters.dim,n_par)
fill!(u0,0.0f0)
u0[2,:].=1.0f0 #down state

function prepare_initial!(u0) #random position on the Bloch sphere
	fill!(u0,0.0f0)
	theta=acos.(2*rand(n_par).-[1])  #uniform sampling for cos(theta) between -1 and 1
	phi=rand(n_par)*2*pi
    #real parts
	u0[1,:]+=cos.(theta/2)
	u0[2,:]+=sin.(theta/2).*cos.(phi)
   #imag parts
	#u0[3,:].+=0
	u0[4,:]+=sin.(theta/2).*sin.(phi)
	#normalize initial state -already normalized by definition
	norm_factor=sqrt.(sum(u0[1:2*MyParameters.dim,:].^2,dims=1))
	u0[1:2*MyParameters.dim,:]=u0[1:2*MyParameters.dim,:]./norm_factor
	return u0
end

# target state
# ψtar = |up>
ut_complex = Array{Float32,1}(undef, MyParameters.dim)
fill!(ut_complex,0.0f0)
ut_complex[1,:].=1.0f0
Re_ut=real(Array{Float32}(ut_complex))
Im_ut=imag(Array{Float32}(ut_complex))
#To visualize
Fock_target=SVector{MyParameters.dim}(Re_ut.^2+Im_ut.^2)

############################################
#Mutate vector outside gradients!
# Necessary for collectiong results during the training
mut(A,index,x) = (A[index] = x)
mut_row(A,index,x)= (A[index,:]=x)
mut_vec(A,x)= (A.=x)

##################################
# time range for the solver
t_interval=round(MyParameters.n_substeps*MyParameters.dt,digits=5)
tspan = (0.0f0,t_interval)

#########################################
#Static Arrays to collect results
mean_fid_store=zeros(MVector{MyParameters.n_steps+1,Float32})
std_fid_store=zeros(MVector{MyParameters.n_steps+1,Float32})
#single_traj_fid_store=zeros(MMatrix{MyParameters.n_steps+1,n_par,Float32})
single_traj_fid_store=zeros(MMatrix{MyParameters.n_steps+1,1,Float32}) #only the first one
mean_action_store=zeros(MVector{MyParameters.n_steps,Float32})
std_action_store=zeros(MVector{MyParameters.n_steps,Float32})
#single_traj_action_store=zeros(MMatrix{MyParameters.n_steps,n_par,Float32})
single_traj_action_store=zeros(MMatrix{MyParameters.n_steps,1,Float32})
Fock_end_example=zeros(MVector{MyParameters.dim,Float32})

#########################################
# compute loss
###########################
using  DiffEqSensitivity
using DiffEqNoiseProcess

condition(u,t,integrator) = true
function affect!(integrator)
	 integrator.u=integrator.u/norm(integrator.u)
end
cb = DiscreteCallback(condition,affect!,save_positions=(false,false))

CreateGrid(W1) =  (NoiseGrid(Array{Float32}((0.0:MyParameters.dt:(t_interval+MyParameters.dt))),W1))
# set scalar random process
W = sqrt(MyParameters.dt)*randn(Float32,MyParameters.n_substeps+1) #for 1 trajectory
W1 = cumsum([zero(MyParameters.dt); W[1:end-1]], dims=1)
NG = CreateGrid(W1)
# define SDE problem
prepare_initial!(u0)
α=zeros(Float32,1,n_par)
prob = SDEProblem{true}(qb_dynamics_dt!,qb_dynamics_dW!, u0[:,1], tspan,α[1,1], noise=NG )


function propagate_along_trajectory(p1)

	u=u0 #initial uncertainty set to zero, last in u is uncertainty
	α=zeros(Float32,1,n_par)

	function prob_func(prob, i, repeat)
		#prepare tge vector of Wiener Process
		W = sqrt(MyParameters.dt)*randn(Float32,MyParameters.n_substeps+1) #for 1 trajectory
		W1 = cumsum([zero(MyParameters.dt); W[1:end-1]], dims=1)
		NG = CreateGrid(W1) # EM and RKMil
		remake(prob,p=α[i:i],u0=u[:,i],noise=NG)
	end

	#initial state
	Re_uj= @view u[1:MyParameters.dim,:]
	Im_uj= @view u[MyParameters.dim+1:end,:]
	fid=zeros(Float32,1,n_par)
	fid+=abs2.(sum(Re_ut.*Re_uj,dims=1)) #average over n_par
	fid+=abs2.(sum(Im_ut.*Im_uj,dims=1))
	fid+=abs2.(sum(Im_ut.*Re_uj,dims=1))
	fid+=abs2.(sum(Re_ut.*Im_uj,dims=1))
	fid+=2*(sum(Re_ut.*Re_uj,dims=1)).*sum(Im_ut.*Im_uj,dims=1)
	fid-=2*(sum(Re_ut.*Im_uj,dims=1)).*sum(Im_ut.*Re_uj,dims=1)
	mut_row(single_traj_fid_store,1,fid[:,1])
	mut(mean_fid_store,1,mean(fid))
	mut(std_fid_store,1,std(fid))

	for j in 1:MyParameters.n_steps
		#update action from NN
		α = re(p1)(u).*MyParameters.force_mag #dim (1,n_par)
		#define ensemble problem
		ensembleprob = EnsembleProblem(prob,prob_func = prob_func)

		u = Array(solve(ensembleprob, RKMil(),
		ensemblealg=EnsembleThreads(), # EnsembleCPUArray(), EnsembleDistributed()
		sensealg = ForwardDiffSensitivity(),
		dt=MyParameters.dt,# save_everystep=false,
		saveat=[t_interval],callback=cb, adaptive=false, trajectories=n_par, batch_size=32,
		save_noise=false,
		timeseries_errors=false,weak_timeseries_errors=false,weak_dense_errors=false
		))[:,1,:]

		Re_uj= @view u[1:MyParameters.dim,:]
		Im_uj= @view u[MyParameters.dim+1:end,:]
		fid=zeros(Float32,1,n_par)
		fid+=abs2.(sum(Re_ut.*Re_uj,dims=1)) #average over n_par
	#	fid+=abs2.(sum(Im_ut.*Im_uj,dims=1))
	#	fid+=abs2.(sum(Im_ut.*Re_uj,dims=1))
		fid+=abs2.(sum(Re_ut.*Im_uj,dims=1))
	#	fid+=2*(sum(Re_ut.*Re_uj,dims=1)).*sum(Im_ut.*Im_uj,dims=1)
	#	fid-=2*(sum(Re_ut.*Im_uj,dims=1)).*sum(Im_ut.*Re_uj,dims=1)
		mut(mean_fid_store,j+1,mean(fid))
		mut(std_fid_store,j+1,std(fid))
		mut_row(single_traj_fid_store,j+1,fid[:,1])
		mut(mean_action_store,j,mean(2*α)) ### HERE RESCALING TO OMEGA!
		mut(std_action_store,j,std(2*α))
		mut_row(single_traj_action_store,j,2*α[:,1])
		#punish large actions--note, that for j we are pointing to the j-1st action!
   		#emphasize the main interval
   		if j>(MyParameters.n_steps-50)
   			mut_vec(Fock_end_example,Re_uj[:,1].^2+Im_uj[:,1].^2);
   		end
	end
end

#test
#prepare_initial!(u0)
#loss_along_trajectory(p1)

using DelimitedFiles  #save txt files





function test_run!(test_func, p1, u0)
	prepare_initial!(u0)  #different initial states!
	u0[:,1].=0.0f0
	u0[2,1]=1.0f0
	ini_fid=abs2.(sum(Re_ut.*u0[1:2],dims=1))+abs2.(sum(Re_ut.*u0[3:4],dims=1))
	@show mean(ini_fid)
	test_func(p1)
	fig1 = plot( [1:MyParameters.n_steps+1,1:MyParameters.n_steps+1],
	  [mean_fid_store mean_fid_store],
	    fillrange=[mean_fid_store.-std_fid_store mean_fid_store.+std_fid_store], fillalpha=0.3, c=:blue,
		xlabel = "steps", ylim=(0,1),xlim=(0,MyParameters.n_steps), lw = 1.5, title="Fidelity" ,legend=false)
	fig1=plot!(1:MyParameters.n_steps+1,single_traj_fid_store[:,1], c=:black  )
	fig2 = plot( [1:MyParameters.n_steps,1:MyParameters.n_steps],
	  [mean_action_store mean_action_store],
	    fillrange=[mean_action_store.-std_action_store mean_action_store.+std_action_store], fillalpha=0.3, c=:orange,
		xlabel = "steps", ylim=(-2*MyParameters.force_mag,2*MyParameters.force_mag), xlim=(0,MyParameters.n_steps), lw = 1.5, title="Action" ,legend=false)
	fig2=plot!(1:MyParameters.n_steps,single_traj_action_store[:,1], c=:black  )
	display(plot(fig1, fig2, layout = (1, 2), legend = false,size=(800,300)));
	png("Figure_test") #for saving figs
	epoch_output=hcat(1:MyParameters.n_steps,mean_fid_store[1:end-1],single_traj_fid_store[1:end-1,1],mean_action_store,single_traj_action_store[:,1])
	#epoch_output=hcat(1:MyParameters.n_steps,mean_fid_store[1:end-1],mean_action_store) #save results for epoch
#	writedlm(string("Test.txt"), epoch_output)
end

#training
Random.seed!(180)
 test_run!(propagate_along_trajectory, p1, u0)
#writedlm("TestRun_seed180.txt", hcat(1:MyParameters.n_steps,mean_fid_store[1:end-1],single_traj_fid_store[1:end-1,1],mean_action_store,single_traj_action_store[:,1]))

# loss hyperparameters
C1 = 0.8f0 # evolution state fid
C2 = 0.001f0 # action amplitudes
C3 = 1.8f0*MyParameters.n_steps/50
loss_norm=C1*MyParameters.n_steps+C3*50

#plotting of the loss function yaxis=:log
plot(training, xlabel = "epochs", ylabel="loss", lw = 1.5, title = "Training")
plot(training/loss_norm, xlabel = L"$i$", ylabel=L"$\mathcal{L}$",  lw = 1.5, yaxis=:log)
savefig("Training_state-in_parallel_simpleNN.pdf")
png(string("Training_state-in_parallel_simpleNN.pdf"))
#save txt
#writedlm("Training_state-in_parallel_simpleNN.txt", training)

training=readdlm("Training_state-in_parallel_simpleNN.txt")

using LaTeXStrings
fig0=plot(training/loss_norm, xlabel = "epoch", # title=L"$\mathcal{L}$",
 lw = 1.5, #yaxis=:log,
  ylim=(0,1),
 xtickfont = font(8), ytickfont = font(8), guidefont=font(10), c=:navyblue,legend=false,grid=false)


fig1 = plot( [1:MyParameters.n_steps+1,1:MyParameters.n_steps+1],
  [mean_fid_store mean_fid_store],
	fillrange=[mean_fid_store.-std_fid_store mean_fid_store.+std_fid_store], fillalpha=0.5, c=:navyblue,
	xlabel = L"$i$", ylim=(0,1),xlim=(0,MyParameters.n_steps), lw = 1.5,# title=L"$\overline{F}(t_i)$" ,
	xtickfont = font(8), ytickfont = font(8), guidefont=font(12), legend=false, grid=false)
fig1=plot!(1:MyParameters.n_steps+1,single_traj_fid_store[:,1], c=:black ,lw = 1.5, )
fig2 = plot( [1:MyParameters.n_steps,1:MyParameters.n_steps],
  [mean_action_store mean_action_store],
	fillrange=[mean_action_store.-std_action_store mean_action_store.+std_action_store], fillalpha=0.5, c=:orangered4,
	xtickfont = font(8), ytickfont = font(8), guidefont=font(12),  grid=false,
	xlabel = L"$i$", ylim=(-2*MyParameters.force_mag,2*MyParameters.force_mag), xlim=(0,MyParameters.n_steps), lw = 1.5, #title=L"$\overline{\Omega}(t_i)$"
	legend=false)
fig2=plot!(1:MyParameters.n_steps,single_traj_action_store[:,1], c=:black,lw = 1.5,)


figCom = plot(fig0, fig1, fig2, margin=3Plots.mm, layout = (1, 3), legend = false, size=(900,300))

#pl1 = plot(1:MyParameters.n_steps+1, mean_fid_store,
#      ribbon = std_fid_store,grid=false,
#      ylim = (0,1), xlim = (0,MyParameters.n_steps),
#      c=:navyblue, lw = 1.5,  xlabel = L"i",
	  # title=L"F(t_i)",
#	    legend=false)
#plt1=plot!(1:MyParameters.n_steps+1,single_traj_fid_store[:,1], c=:black  )
#  pl2 = plot(1:MyParameters.n_steps, mean_action_store,grid=false,
#      ribbon = std_action_store,
#      ylim=(-2*MyParameters.force_mag,2*MyParameters.force_mag), xlim = (0,MyParameters.n_steps),
#      c=:orangered4, lw = 1.5, xlabel = L"i",
#	 # title=L"\Omega(t_i)",
#	  legend=false)
#pl2=plot!(1:MyParameters.n_steps,single_traj_action_store[:,1], c=:black,grid=false)

#  pl = plot(pl1, pl2, layout = (1, 2), legend = false, size=(800,360))

 # plot training loss
# pl3 = plot(training/loss_norm, lw = 1.5, yticks = 0:0.2:1,ylim = (0,1),
# xlabel = LaTeXString("epoch"),
 # title=L"\mathfrak{L}",
#  legend=false,grid=false, c=:navyblue)

#pl = plot(pl3, pl1, pl2, margin=3Plots.mm, layout = (1, 3), legend = false, size=(900,300))

 savefig("StateBack_Periodic_NoTitles.pdf")

#### QUANTITATIVE: add up mean fidelity
overall_fid=mean(mean_fid_store) #0.89330983f0
overall_std_fid=mean(std_fid_store) #0.099090025f0
