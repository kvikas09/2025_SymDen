import numpy as np

class XYModelMetropolisSimulation:
    '''H_matrix is valid only for 2D model'''

    def __init__(self,lattice_shape,beta,J=1,K=1,random_state=None):
        self.beta = beta
        self.rs = np.random.RandomState(seed=random_state)
        self.L = self.rs.rand(*lattice_shape)
        self.lattice_shape = lattice_shape
        self.initial_L = self.L.copy()
        self.t = 0
        self.J = J
        self.K = K
        self.modified_in_last_step = False

        self.H_matrix = np.zeros(self.L.shape)        
        self._calculate_H_matrix()
        
        
        self.H = np.sum(self.H_matrix)
        self.H_vals = [self.H]
        self.accept = 0
        self.reject = 0
        

    def _calculate_H_matrix(self):
        for i in range(self.L.shape[0]):
            for j in range(self.L.shape[1]):
                self.H_matrix[i,j]  = 0
                self.H_matrix[i,j] -= 0.50*self.J*np.cos(2 * np.pi * (self.L[i,j] - self.L[i,(j+1) % self.L.shape[1]]))
                self.H_matrix[i,j] -= 0.50*self.J*np.cos(2 * np.pi * (self.L[i,j] - self.L[i,(j-1) % self.L.shape[1]]))
                self.H_matrix[i,j] -= 0.50*self.J*np.cos(2 * np.pi * (self.L[i,j] - self.L[(i+1) % self.L.shape[0],j]))
                self.H_matrix[i,j] -= 0.50*self.J*np.cos(2 * np.pi * (self.L[i,j] - self.L[(i-1) % self.L.shape[0],j]))
                self.H_matrix[i,j] -= 0.25*self.K*np.cos(2 * np.pi * (self.L[i,j] + self.L[(i-1) % self.L.shape[0],(j-1) %                                      self.L.shape[1]] - self.L[i,(j-1) % self.L.shape[1]] - self.L[(i-1) % self.L.shape[0],j]))
                self.H_matrix[i,j] -= 0.25*self.K*np.cos(2 * np.pi * (self.L[i,j] + self.L[(i-1) % self.L.shape[0],(j+1) %                                       self.L.shape[1]] - self.L[i,(j+1) % self.L.shape[1]] - self.L[(i-1) % self.L.shape[0],j]))
                self.H_matrix[i,j] -= 0.25*self.K*np.cos(2 * np.pi * (self.L[i,j] + self.L[(i+1) % self.L.shape[0],(j-1) %                                       self.L.shape[1]] - self.L[i,(j-1) % self.L.shape[1]] - self.L[(i+1) % self.L.shape[0],j]))
                self.H_matrix[i,j] -= 0.25*self.K*np.cos(2 * np.pi * (self.L[i,j] + self.L[(i+1) % self.L.shape[0],(j+1) %                                       self.L.shape[1]] - self.L[i,(j+1) % self.L.shape[1]] - self.L[(i+1) % self.L.shape[0],j]))
        
    def _get_delta_H(self, pos, new_val):
        ans = 0
        old_val = self.L[pos]
        pos_list = list(pos)
        i = pos_list[0]
        j = pos_list[1]
        i_1 = (i-1)%self.L.shape[0]
        i1  = (i+1)%self.L.shape[0]
        j_1 = (j-1)%self.L.shape[1]
        j1  = (j+1)%self.L.shape[1]
        
        ans += self.J*(np.cos(2*np.pi*(self.L[i,j_1] - new_val)) - np.cos(2*np.pi*(self.L[i,j_1] - old_val)))
        ans += self.J*(np.cos(2*np.pi*(self.L[i,j1] - new_val)) - np.cos(2*np.pi*(self.L[i,j1] - old_val)))
        ans += self.J*(np.cos(2*np.pi*(self.L[i_1,j] - new_val)) - np.cos(2*np.pi*(self.L[i_1,j] -old_val)))
        ans += self.J*(np.cos(2*np.pi*(self.L[i1,j] - new_val)) - np.cos(2*np.pi*(self.L[i1,j] - old_val)))
        
        
        ans += self.K*(np.cos(2*np.pi*(self.L[i_1,j] + self.L[i,j_1] - self.L[i_1,j_1] - new_val))  
                     - np.cos(2*np.pi*(self.L[i_1,j] + self.L[i,j_1] - self.L[i_1,j_1] - old_val)))
        ans += self.K*(np.cos(2*np.pi*(self.L[i_1,j] + self.L[i,j1] - self.L[i_1,j1] - new_val)) 
                     - np.cos(2*np.pi*(self.L[i_1,j] + self.L[i,j1] - self.L[i_1,j1] - old_val)))
        ans += self.K*(np.cos(2*np.pi*(self.L[i,j_1] + self.L[i1,j] - self.L[i1,j_1] - new_val)) 
                     - np.cos(2*np.pi*(self.L[i,j_1] + self.L[i1,j] - self.L[i1,j_1] - old_val)))
        ans += self.K*(np.cos(2*np.pi*(self.L[i1,j] + self.L[i,j1] - self.L[i1,j1] - new_val))  
                     - np.cos(2*np.pi*(self.L[i1,j] + self.L[i,j1] - self.L[i1,j1] - old_val)))
        
        return -ans
    
    def _renew_H_matrix(self, pos, new_val):
        old_val = self.L[pos]
        pos_list = list(pos)
        i = pos_list[0]
        j = pos_list[1]
        i_1 = (i-1)%self.L.shape[0]
        i1  = (i+1)%self.L.shape[0]
        j_1 = (j-1)%self.L.shape[1]
        j1  = (j+1)%self.L.shape[1]
        link_delta_H1 = 0.5*self.J*(np.cos(2*np.pi*(self.L[i,j_1] - new_val)) - np.cos(2*np.pi*(self.L[i,j_1] - old_val)))
        link_delta_H2 = 0.5*self.J*(np.cos(2*np.pi*(self.L[i,j1] - new_val)) - np.cos(2*np.pi*(self.L[i,j1] - old_val)))
        link_delta_H3 = 0.5*self.J*(np.cos(2*np.pi*(self.L[i_1,j] - new_val)) - np.cos(2*np.pi*(self.L[i_1,j] -old_val)))
        link_delta_H4 = 0.5*self.J*(np.cos(2*np.pi*(self.L[i1,j] - new_val)) - np.cos(2*np.pi*(self.L[i1,j] - old_val)))
        
        
        ring_delta_R1 = 0.25*self.K*(np.cos(2*np.pi*(self.L[i_1,j] + self.L[i,j_1] - self.L[i_1,j_1] - new_val)) 
                                   - np.cos(2*np.pi*(self.L[i_1,j] + self.L[i,j_1] - self.L[i_1,j_1] - old_val)))
        ring_delta_R2 = 0.25*self.K*(np.cos(2*np.pi*(self.L[i_1,j] + self.L[i,j1] - self.L[i_1,j1] - new_val)) 
                                   - np.cos(2*np.pi*(self.L[i_1,j] + self.L[i,j1] - self.L[i_1,j1] - old_val)))
        ring_delta_R3 = 0.25*self.K*(np.cos(2*np.pi*(self.L[i,j_1] + self.L[i1,j] - self.L[i1,j_1] - new_val)) 
                                   - np.cos(2*np.pi*(self.L[i,j_1] + self.L[i1,j] - self.L[i1,j_1] - old_val)))
        ring_delta_R4 = 0.25*self.K*(np.cos(2*np.pi*(self.L[i1,j] + self.L[i,j1] - self.L[i1,j1] - new_val))  
                                   - np.cos(2*np.pi*(self.L[i1,j] + self.L[i,j1] - self.L[i1,j1] - old_val)))
        self.H_matrix[i,j_1]  -= link_delta_H1 - ring_delta_R1 - ring_delta_R3
        self.H_matrix[i,j1]   -= link_delta_H2 - ring_delta_R2 - ring_delta_R4
        self.H_matrix[i_1,j]  -= link_delta_H3 - ring_delta_R1 - ring_delta_R2
        self.H_matrix[i1,j]   -= link_delta_H4 - ring_delta_R3 - ring_delta_R4
        self.H_matrix[i_1,j_1]-= ring_delta_R1
        self.H_matrix[i-1,j1] -= ring_delta_R2
        self.H_matrix[i1,j_1] -= ring_delta_R3
        self.H_matrix[i1,j1]  -= ring_delta_R4
        self.H_matrix[i,j]    -= link_delta_H1 - link_delta_H2 - link_delta_H3 - link_delta_H4 
        self.H_matrix[i,j]    -= ring_delta_R1 - ring_delta_R2 - ring_delta_R3 - ring_delta_R4
        
        
            
    def make_step(self):
        change_pos = tuple([self.rs.randint(_) for _ in self.lattice_shape])
        new_val = self.rs.rand()
        delta_H = self._get_delta_H(change_pos, new_val)
        if (delta_H > 0):
            if (self.rs.rand() < np.exp(-self.beta * delta_H)):
                self._renew_H_matrix(change_pos, new_val)
                self.L[change_pos] = new_val
                self.H += delta_H
                self.modified_in_last_step = True
                self.accept +=1
            else:
                self.modified_in_last_step = False
                self.reject +=1
        else:
            self._renew_H_matrix(change_pos, new_val)
            self.L[change_pos] = new_val
            self.H += delta_H
            self.modified_in_last_step = True
            self.accept +=1
        self.t += 1


    def simulate(self, steps, iters_per_step):
        for i in range(steps):
            for j in range(iters_per_step):
                self.make_step()
            self.H_vals.append(self.H)
            
            
            
   

    