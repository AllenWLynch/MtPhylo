from abc import ABC, abstractmethod
from scipy.optimize import linprog
import numpy as np
from .node import log_safe_matmul


class EvolutionModel(ABC):
    
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def add_constant_site(self, observation: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def logp_obs_given_state(self, observation: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def transition_matrix(self, branch_length: float) -> np.ndarray:
        pass

    @abstractmethod
    def ddt_transition_matrix(self, branch_length: float) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def site_weight_vector(self):
        pass

    @property
    @abstractmethod
    def steady_state(self):
        pass

    @property
    @abstractmethod
    def n_states(self) -> int:
        pass

    @property
    @abstractmethod
    def n_obs_states(self,) -> int:
        pass

    @property
    def site_dim(self) -> int:
        return len(self.site_weight_vector)



class JCNucModel(EvolutionModel):

    _n_obs_states=4
    _n_states=4

    def __init__(self,n_variable_sites,n_constant_A=0, n_constant_C=0, n_constant_G=0, n_constant_T=0):
        self._n_variable_sites = n_variable_sites
        self.site_weight_vector_ = np.array([1]*n_variable_sites + [n_constant_A, n_constant_C, n_constant_G, n_constant_T])
    
    @staticmethod
    def logp_obs_given_state(seq):
        states=['A','C','G','T']
        onehot_matrix = np.full((4,len(seq)), -np.inf)
        for i, base in enumerate(seq):
            onehot_matrix[states.index(base),i] = 0
        
        return onehot_matrix
    
    @property
    def site_weight_vector(self):
        return self.site_weight_vector_

    @staticmethod
    def transition_matrix(v):
        return (-1/4*np.ones((4,4)) + np.eye(4))*np.exp(-v) + 1/4

    @staticmethod
    def ddt_transition_matrix(v):
        return -np.exp(-v)*(-1/4*np.ones((4,4)) + np.eye(4))
    
    @staticmethod
    def dddt_transition_matrix(v):
        return np.exp(-v)*(-1/4*np.ones((4,4)) + np.eye(4))
    
    @property
    def steady_state(self):
        return np.array([1/4,1/4,1/4,1/4])
    
    @staticmethod
    def add_constant_site(seq):
        return seq + 'ACGT'
    
    @property
    def n_states(self):
        return self._n_states
    
    @property
    def n_obs_states(self):
        return self._n_obs_states
    

class VirtualPopulationModel(EvolutionModel):
    
    def __init__(self,*, virtual_popsize, full_popsize, mutation_rate,
                 n_variable_sites, n_constant_sites):
        self._n_obs_states = full_popsize+1
        self._n_states = virtual_popsize+1
        self._n_variable_sites = n_variable_sites
        self._n_constant_sites = n_constant_sites
        self.site_weight_vector_ = np.array([1]*n_variable_sites + [n_constant_sites])

        self.Q_ = self._make_rate_matrix(mutation_rate*full_popsize/virtual_popsize, virtual_popsize)
        self.steady_state_= self._get_stationary_distribution(self.Q_)

        self.transition_model_, self.ddt_transition_model_, self.dddt_transition_model_ \
            = self._get_diffusion_functions(self.Q_)
        
        # The projection matrix goes from state space to obs space, and has rows which sum to one
        # The aggregation matrix goes from obs space to state space, and has columns which sum to one
        self.aggregation_matrix_, self.projection_matrix_ = \
            self._make_aggregation_matrix(self._n_obs_states, self._n_states)

    
    def logp_obs_given_state(self, minor_allele_counts):
        eps=1e-10
        onehot_matrix = np.full((self.n_obs_states,len(minor_allele_counts)), np.log(eps))
        for i, count in enumerate(minor_allele_counts):
            onehot_matrix[count,i] = np.log(1-eps*(self.n_obs_states-1))
        
        return log_safe_matmul(self.projection_matrix_, onehot_matrix)
    

    def transition_matrix(self, v):
        return self.transition_model_(v)

    def ddt_transition_matrix(self, v):
        return self.ddt_transition_model_(v)
    
    def dddt_transition_matrix(self, v):
        return self.dddt_transition_model_(v)
    
    @property
    def steady_state(self):
        return self.steady_state_
    
    @staticmethod
    def add_constant_site(minor_allele_counts):
        return list(minor_allele_counts) + [0]
    
    @property
    def n_states(self):
        return self._n_states
    
    @property
    def n_obs_states(self):
        return self._n_obs_states
    
    @property
    def site_weight_vector(self):
        return self.site_weight_vector_    
    
    @staticmethod
    def _make_aggregation_matrix(n_obs_states, n_states):
        
        aggregation_matrix=np.full((n_obs_states, n_states), 0)
        aggregation_matrix[0,0]=1
        aggregation_matrix[-1,-1]=1

        n_intermediates = n_states-2
        
        substates_per_obs = (n_obs_states - 1) // n_intermediates
        remainder = (n_obs_states - 1) % n_intermediates
        assert remainder == 0, remainder
        
        for i in range(0, n_states-2):
            startrow=1+i*substates_per_obs; endrow= min(1+(i+1)*substates_per_obs, n_obs_states)
            aggregation_matrix[startrow : endrow, i+1] = 1

        projection_matrix = (aggregation_matrix/aggregation_matrix.sum(0, keepdims=True)).T

        return aggregation_matrix, projection_matrix
    
        
    @staticmethod
    def _is_valid_Q(Q):
        diag_mask = ~np.eye(Q.shape[0], dtype=bool)
        assert (np.diag(Q) < 0).all() and (Q[diag_mask] >= 0).all()

    @staticmethod
    def _make_rate_matrix(lamba_, N):
        Q=np.zeros((N+1,N+1))
        #N+=1
        for i in range(N+1):
            #print(i)
            if i==0:
                Q[i,i+1]=lamba_*N
            elif i==N:
                Q[i,i-1]=lamba_*N
            else:
                Q[i,i-1]= (N-i)*(i)/N
                Q[i,i+1]= (N-i)*(i)/N

        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q

    @staticmethod
    def _coarsen_rate_matrix(*,Q,pi,A):
        Dn = np.diag(pi)
        DN = A.T @ Dn @ A
        p_eq = pi[:, None] @ np.ones((1,pi.size))
        P_eq = (pi @ A)[:, None] @ np.ones((1,A.shape[1]))
        
        return P_eq - DN @ np.linalg.inv(A.T @ np.linalg.inv(p_eq - Q) @ Dn @ A)


    @staticmethod
    def _get_stationary_distribution(Q):
        dim=Q.shape[0]
        res = linprog(np.ones(dim), 
                A_eq=np.hstack([Q, np.ones((dim,1))]).T,
                b_eq=[0]*dim + [1],
                bounds=[(0, 1)]*dim
                )
        
        assert res.success

        return res.x


    @staticmethod
    def _get_diffusion_functions(Q):
        
        _, vecs = np.linalg.eig(Q)
        D = np.linalg.inv(vecs) @ Q @ vecs
        assert np.allclose(D * (1-np.eye(D.shape[0])), 0)
        inv_vecs=np.linalg.inv(vecs)
        D=np.diag(D)[np.newaxis,:]

        def diffuse(t):
            return ( (vecs * np.exp(D*t)) @ inv_vecs ).astype(float)
        
        def ddt_diffuse(t):
            return (
                (vecs * D * np.exp(D*t)) @ inv_vecs
            ).astype(float)
        
        def dddt_diffuse(t):
            return (
                (vecs * D**2 * np.exp(D*t)) @ inv_vecs
            ).astype(float)
        
        return diffuse, ddt_diffuse, dddt_diffuse