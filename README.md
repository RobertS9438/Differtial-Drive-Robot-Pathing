# Differtial-Drive-Robot-Pathing
This is the repository for my CS 5335 project regarding pathing algorithms for a differential drive robot.

Brief Overview:
  This is the repository for my CS 5335 project regarding pathing algorithms for a differential drive robot. In addition to the dunctions for creatinging the RRG graphs and using them to extract wheel commands to control a differential drive robot to path to a desired location, there is also code for testing and simulation purposes.
  
  
  Files:
  pathing.py:
    This is the primary pathing algorithm code. It has all the code for creating the RRG graph including different inter-node pathing algorithms, the A* search algorithm for determining a path between nodes of the RRG graph, and code for extracting a sequence of wheel commands from the the node path.
    
    
  create_occ_grid.py
    This file contains all code handling the occupancy grid. This includes code for generating a set of cylinders with random sizes and positions according to the functions input parameters. Another function uses this randomly generated obstacle set to create an occupancy grid and saves both it and the obtsacle information to two separate csv files. There is code for loading both the csv files into objects usable by python as well. Finally, the code for modifying the occupancy grid further to account for things like the robot radius is located here too.
     
  simulation_setup.py
  
    This file is used for setting up and running the simulation. It includes functions for using the obstacle file from create_occ_grid.py to create a matching environment and load the urdf file for the rover, driving the simulation using input wheel commands, and code for using the RRG graph and an input node path to traverse the environment. The traversal algorithm even includes a course correction component, but one unresolved error with it can cause the course correction to get stuck in a loop despite converging to the desired pose. This error is not considered a serious issue since the robot still reaches the desired postition.
    
    Note: The path traversal algorithm accomodates the implementation of artificial noise in the pose information sent from the simulation as well as a particle filter to try and address the errors this would cause, however, the parameters of the particle filter need to be determined experimentally and it still does not account for collisions so it is currently incomplete.
  
  pathing_test.py:
      File that uses the create_occ_grid.py and pathing.py files to create a RRG. Requires a valid name input for the occupancy grid file already generated by create_occ_grid previously. File must be modified to use desired parameters such as robot wheel radius, robot width, and a specified inter-node pathing function from pathing.py.
      Note 1: particle_num and sigma_2 should remain as 1 and 0 respectively. They are remnants from a stretch goal utilizing a particle filter to determine edge costs that was not completed.
      Note 2: Code is currently set up to scale to a 100mx100m environment, for a differently scaled, but still square, occupancy grid, the robot width and wheel radius must be adjusted accordingly.
      
   sim_test.py:
      File for running a simulation of pathing using the RRG generated by pathing_test.py. It requires most of the parameters match those used for pathing.test. Also require additional paramters to determine how new nodes are added to the RRG which can differ from thoses used by paathing_test.py. The occupancy grid name and RRG pickle file name must be manually changed in the file.
      
   test_script.py:
      Plots the edges and node of an RRG graph generated by pathing_test. This can take a while depending on the size of the RRG and the method used for determining wheel commands associated with each edge.
      
      
    test_occ files:
        Some example occupancy grids.
        
    test_graph files:
        Some example RRG pickle files. The use robot width of 2 and wheel radius of 1. The files with 'few' in the name uses test_occ and the ones with 'many' use test_occ_3. The pathing method and node number are also in the names.
   
