# import libraries
import numpy as np
from collections import namedtuple
from numpy import random

class environment:
    
    '''
    Create an envirionment with an agent and a given number of ghosts that move with the given probability.
    Valid number of ghosts is [1,2,3] and valid value for the probability of ghosts moving is between 0 and 1.
    '''
  
    def __init__(self,n_ghosts=1,p=0.5,random_env=True,random_learning=True):

        # random or not
        self.random_env = random_env
        self.random_learning = random_learning
        
        # Convenient data structure to hold information about actions
        Action = namedtuple('Action', 'name index delta_i delta_j')

        up = Action('up', 0, -1, 0)    
        down = Action('down', 1, 1, 0)    
        left = Action('left', 2, 0, -1)    
        right = Action('right', 3, 0, 1)    
        index_to_actions = {}
        for action in [up, down, left, right]:
            index_to_actions[action.index] = action
        
        # create an environment as a numpy array
        self.env  =  np.array(  [[1,1,1,1,1,1,1,1,1,1,1],
                                 [1,0,0,0,0,1,0,0,0,0,1],
                                 [1,0,1,1,0,1,0,1,1,0,1],
                                 [1,0,1,0,0,0,0,0,1,0,1],
                                 [1,0,1,0,1,1,1,0,1,0,1],
                                 [1,0,0,0,0,0,0,0,0,0,1],
                                 [1,0,1,0,1,1,1,0,1,0,1],
                                 [1,0,1,0,0,0,0,0,1,0,1],
                                 [1,0,1,1,0,1,0,1,1,0,1],
                                 [1,0,0,0,0,1,0,0,0,0,1],
                                 [1,1,1,1,1,1,1,1,1,1,1]])
        
        
        # define sizes of the environment and number of possible actions
        self.size = 11
        N = self.size
        self.action_n = 4
        
        # define number of ghosts and the probability of their moves
        self.n_ghosts = n_ghosts
        self.p = [p,1-p]

        # initial position of the agent will be decided by resetting the environment.
        self.position_agent = None
        
        # run time
        self.time_elapsed = 0
        self.time_limit = N**2
        
        # display help
        self.dict_map_display={ 0:'.',
                                1:'#',
                                2:'G',
                                3:'P',
                                4:'A',
                                5:' '}      
        
        # time of power
        self.power_time = None
        
        # previous positions of the ghosts
        self.pre_pos1 = [0,0]
        self.pre_pos2 = [0,0]
        self.pre_pos3 = [0,0]
        
        # dead or alive
        self.alive_ghost1 = True
        self.alive_ghost2 = True
        self.alive_ghost3 = True
        
        
        # Additional calculations to get the transition and reward matrix
        self.reward_matrix = -1 * np.ones((N*N, 4)) # All movements give you by default a reward of -1
        self.transition_matrix = np.zeros((N*N, 4, N*N))
        
        # In order to explicitely show that the way you represent states doesn't matter, 
        # we will assign a random index for each coordinate of the grid        
        index_states = np.arange(0, N*N)
        np.random.shuffle(index_states)
        self.coord_to_index_state = index_states.reshape(N,N)
        
        # fill the matrix with appropriate values
        for i in range(1, N-1):
            for j in range(1, N-1):
                
                current_state = self.coord_to_index_state[i,j]
                current_cell = self.env[i,j]
                
                for action in [up, left, down, right]:
                    
                    destination_cell = self.env[i +action.delta_i , j + action.delta_j]
                    next_state = self.coord_to_index_state[i + action.delta_i, j + action.delta_j]

                    # Check if you bump
                    if destination_cell in [0, 2, 3]:    
                        self.transition_matrix[current_state, action.index, next_state] = 1
                    else:
                        self.transition_matrix[current_state, action.index, current_state] = 1
                        destination_cell = current_cell
                        next_state = current_state
                        self.reward_matrix[current_state, action.index] += -5 

                    # Check where the agent ends up
                    if destination_cell == 2:
                        self.reward_matrix[current_state, action.index] += -20
                    elif destination_cell == 3:
                        self.reward_matrix[current_state, action.index] += N**2
                        
    
    def get_empty_cells(self, n_cells):
        
        #pick up empty cells and randamly choose one
        empty_cells_coord = np.where( (self.env == 0) | (self.env == 5) )
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates
        
    
    def step(self, action):
        
        # define actions by number for convenience 
        if action == 0:
            action = 'up'
        if action == 1:
            action = 'down'        
        if action == 2:
            action = 'left'        
        if action == 3:
            action = 'right'       
        
        # Ghost1 moves
        if self.alive_ghost1:
            move = np.random.choice([True,False], 1, p=self.p)
            if move:
                directions1 = [[self.position_ghost1[0]-1,self.position_ghost1[1]],
                                       [self.position_ghost1[0]+1,self.position_ghost1[1]],
                                       [self.position_ghost1[0],self.position_ghost1[1]-1],
                                       [self.position_ghost1[0],self.position_ghost1[1]+1]]
                possible_directions1 = []
                for d in directions1:
                    if not self.env[d[0],d[1]] == 1:
                        if (d[0] == self.pre_pos1[0]) and (d[1] == self.pre_pos1[1]):
                            continue
                        else:
                            possible_directions1.append(d)
                random_direction1 = np.random.randint(0,len(possible_directions1))
                direction1 = possible_directions1[random_direction1]
                self.pre_pos1 = self.position_ghost1
                self.position_ghost1 = direction1
        
        # Ghost2 moves
        if self.n_ghosts >= 2:
            if self.alive_ghost2:
                move = np.random.choice([True,False], 1, p=self.p)
                if move:
                    directions2 = [[self.position_ghost2[0]-1,self.position_ghost2[1]],
                                           [self.position_ghost2[0]+1,self.position_ghost2[1]],
                                           [self.position_ghost2[0],self.position_ghost2[1]-1],
                                           [self.position_ghost2[0],self.position_ghost2[1]+1]]
                    possible_directions2 = []
                    for d in directions2:
                        if not self.env[d[0],d[1]] == 1:
                            if (d[0] == self.pre_pos2[0]) and (d[1] == self.pre_pos2[1]):
                                continue
                            else:
                                possible_directions2.append(d)
                    random_direction2 = np.random.randint(0,len(possible_directions2))
                    direction2 = possible_directions2[random_direction2]
                    self.pre_pos2 = self.position_ghost2
                    self.position_ghost2 = direction2
        
        # Ghost3 moves
        if self.n_ghosts >= 3:
            if self.alive_ghost3:
                move = np.random.choice([True,False], 1, p=self.p)
                if move:
                    directions3 = [[self.position_ghost3[0]-1,self.position_ghost3[1]],
                                           [self.position_ghost3[0]+1,self.position_ghost3[1]],
                                           [self.position_ghost3[0],self.position_ghost3[1]-1],
                                           [self.position_ghost3[0],self.position_ghost3[1]+1]]
                    possible_directions3 = []
                    for d in directions3:
                        if not self.env[d[0],d[1]] == 1:
                            if (d[0] == self.pre_pos3[0]) and (d[1] == self.pre_pos3[1]):
                                continue
                            else:
                                possible_directions3.append(d)
                    random_direction3 = np.random.randint(0,len(possible_directions3))
                    direction3 = possible_directions3[random_direction3]
                    self.pre_pos3 = self.position_ghost3
                    self.position_ghost3 = direction3

        
        # check which action is going to be taken by the agent
        if action == 'up':
            next_position = np.array( (self.position_agent[0] - 1, self.position_agent[1] ) )
        if action == 'down':
            next_position = np.array( (self.position_agent[0] + 1, self.position_agent[1] ) )
        if action == 'left':
            next_position = np.array( (self.position_agent[0] , self.position_agent[1] - 1 ) )
        if action == 'right':
            next_position = np.array( (self.position_agent[0] , self.position_agent[1] + 1) )
        
        # If the agent bumps into a wall, it does not move
        if self.env[next_position[0], next_position[1]] == 1:
            bump = True
        else:
            self.position_agent = next_position
            
        # current cell type for convenience
        current_cell_type = self.env[self.position_agent[0], self.position_agent[1]]
        
        # -1 reward at at every timestep
        reward = -1
        bump = False
        power = False
        
        # Check if the agent gets power
        if current_cell_type == 3:
            power = True
            self.power_time = 5
            reward += 20
            # The agent eats the power pellet
            self.env[self.position_agent[0], self.position_agent[1]] = 5
        
        # check if the agent eats a dot
        if current_cell_type == 0:
            reward += 5
            # The agent eats the dot/point
            self.env[self.position_agent[0], self.position_agent[1]] = 5
            
        # check if the agent eats a ghost
        if current_cell_type == 2:
            if power == True:
                reward += 50
                if (self.position_ghost1 == self.position_agent).all():
                    self.alive_ghost1 = False
                    self.position_ghost1 = None
                if self.n_ghosts >= 2:
                    if (self.position_ghost2 == self.position_agent).all():
                        self.alive_ghost2 = False
                        self.position_ghost2 = None
                if self.n_ghosts >= 3:
                    if (self.position_ghost3 == self.position_agent).all():
                        self.alive_ghost3 = False
                        self.position_ghost3 = None
        
        # update time
        self.time_elapsed += 1
        
        # verify termination condition
        done = False
        
        # timeout
        if self.time_elapsed == self.time_limit:
            done = True
            
        # killed by ghosts
        if not power:
            if (self.position_agent == self.position_ghost1).all():
                done = True
            elif self.n_ghosts >= 2:
                if  (self.position_agent == self.position_ghost2).all():
                    done = True
            elif self.n_ghosts >= 3:
                if  (self.position_agent == self.position_ghost3).all():
                    done = True
                    
        # ate all the dots
        if not np.any(np.isin(self.env,0)):
            done = True   
        state = self.coord_to_index_state[ self.position_agent[0], self.position_agent[1]]
        
        return state, reward, done
    

    def display(self):
        
        # display the environment with the agent and the ghosts
        envir_with_agent = self.env.copy()
        envir_with_agent[self.position_agent[0], self.position_agent[1]] = 4
        envir_with_agent[self.position_ghost1[0],self.position_ghost1[1]] = 2
        if self.n_ghosts >= 2:
            envir_with_agent[self.position_ghost2[0],self.position_ghost2[1]] = 2
        if self.n_ghosts >= 3:
            envir_with_agent[self.position_ghost3[0],self.position_ghost3[1]] = 2        
        full_repr = ""
        for r in range(self.size):
            line = ""
            for c in range(self.size):
                string_repr = self.dict_map_display[ envir_with_agent[r,c] ]
                line += "{0:2}".format(string_repr)
            full_repr += line + "\n"
        print(full_repr)

        
    def reset(self,t=0):

        # reset time and the environment
        self.time_elapsed = 0
        self.env[self.env==5] = 0
        
        # random or not
        if self.random_env == False:
            np.random.seed(0)
        if self.random_learning == False:
            np.random.seed(t)
            
        # position of the power pellet
        self.position_power = self.get_empty_cells(1)
        self.env[self.position_power[0], self.position_power[1]] = 3
        
        # set position of the agent that is a numpy array
        self.position_agent = np.asarray(self.get_empty_cells(1))
        
        # set position of the ghosts that are numpy arrays
        self.position_ghost1 = np.asarray(self.get_empty_cells(1))
        if self.n_ghosts >= 2:
            self.position_ghost2 = np.asarray(self.get_empty_cells(1))
        if self.n_ghosts >= 3:
            self.position_ghost3 = np.asarray(self.get_empty_cells(1))

        # set state
        state = self.coord_to_index_state[ self.position_agent[0], self.position_agent[1]]
        
        return state