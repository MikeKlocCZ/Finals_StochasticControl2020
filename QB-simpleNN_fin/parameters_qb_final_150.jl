# In this module, all parameters for the cat state preparation are defined:

module parameters
export parameters_qb

parameters_para=Dict(
    "N" =>2,  #16,
    "w" => 20.0f0, # 2*pi*3.9,
    "force_mag" => 5.0f0,#2.0f0, #1.8f0, # 2*pi*0.3,
    "max_episode_steps" => 150,    #250,
    "dt" => 0.001f0,
    "n_substeps" => 20,
    "gamma" => 1.0f0)
end
