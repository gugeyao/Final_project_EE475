import numpy as np
import sys
from couzinswarm import Swarm

# Torus (Milling) Behavior Simulation
box_length = 1000
# Define the number of time steps
N_t = 4000
num_trials = 1
P = np.zeros((num_trials))
m = np.zeros((num_trials))
speed=np.float64(sys.argv[1])
turning_rate_degree=np.float64(sys.argv[2])
repulsion_radius = np.float64(sys.argv[3])
orientation_width = np.float64(sys.argv[4])
total_length = 18
early_time = 500
attraction_width = total_length - repulsion_radius - orientation_width
for i in range(num_trials):

    # Run the simulation and store positions and directions
    swarm = Swarm(
        number_of_fish=50,
        speed=speed,
        noise_sigma=0.05,
        turning_rate=turning_rate_degree/180*np.pi,
        repulsion_radius=repulsion_radius,
        orientation_width=orientation_width, # 2 is parallel, 0 is swarm
        attraction_width=attraction_width,
        angle_of_perception=270/360*np.pi,
        box_lengths=[box_length, box_length, box_length],
        reflect_at_boundary=[False,False,False],  # Use periodic boundaries to avoid edge effects
        verbose=False,
        initialization_radii = 15,
        using_verletlist = True,
        n_min_group_size = 25,
    )
    r, v,P,m  = swarm.simulate(N_t)
    position_early = r[:,early_time,:]
    direction_early = v[:,early_time,:]
    earlytime_snapshot = np.concatenate((position_early, direction_early), axis=1)
    # Print the results
    print("Trial: "+str(i+1)+" is finished!")
    np.savetxt("P_"+str(i+1)+".txt",P)
    np.savetxt("m_"+str(i+1)+".txt",m)
    np.savetxt("early_snapshot_"+str(i+1)+".txt",earlytime_snapshot)
    
