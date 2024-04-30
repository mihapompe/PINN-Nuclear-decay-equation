"""
Code that defines PINN classes and its associated functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from CRAMsolve import CRAM16
from SineCosineLayer import *
from scipy.sparse import issparse
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

# Access the GPU
device = "mps" if getattr(torch,'has_mps',False) \
    else "cuda" if torch.cuda.is_available() else "cpu"
if device == "mps": device = "cpu"  # Remove this line if you want to run on mps instead of cpu.
DEVICE = torch.device(device)
print("Running on", device)
   

class PINN():
    """
    Parent class for the PINN containing common attributes and the loss function
    for decay and burnup problems.

    Attributes:
    -----------
    size : int
        Size of the network and the output vector
    initial : torch tensor
        Initial conditions
    lamda_matrix : torch tensor
        Burnup or decay matrix A.
    sparse_lamda_matrix : torch sparse tensor
        Sparse representation of matrix A.
    seed : int
        Seed for all random number generators.
    sparse : bool
        If True all the computations with matrix A will be done in 
        the sparse representation.

    Methods:
    --------
    lossFunction(t, weights, batch_size = None)
        Computes the loss function.
    """

    def __init__(self, lamda_matrix, initial, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed
        self.size = lamda_matrix.shape[0]
        self.initial = torch.as_tensor(initial, dtype=DTYPE, device=DEVICE)
        assert lamda_matrix.shape[0]==lamda_matrix.shape[1], "Lambda_matrix is not a square matrix"
        self.lamda_matrix = torch.as_tensor(lamda_matrix, dtype=DTYPE, device=DEVICE)
        assert self.initial.shape[0] == self.size, "Lambda_matrix and initial do not match in size"
        self.sparse_lamda_matrix = self.lamda_matrix.to_sparse()
        self.sparse = False


    def lossFunction(self, t, weights=[1,1], batch_size=None):
        """"
        Computes the loss function.

        t : torch.tensor (shape (num_t_steps, 1))
            Vector of times used for calculating the loss.
        weights : list (default [1,1])
            The first element if the IC weight and the second on the ODE weight.
        batch_size : int (default None)
            Number of time steps used for calculating the loss. If bath_size 
            is given, a random sample of training point is used for each epoch.
        """
        # Initial time in correct shape
        t0 = torch.tensor([0], dtype=DTYPE, device=DEVICE).reshape(-1,1)
        
        if batch_size != None:
            n = t.shape[0]
            t = t[np.random.choice(n,int(batch_size*n),replace=False)]
        
        # Exact and predicted initial values
        N_exact_0 = self.initial
        N_0 = self(t0)
        # Predict values of the network at training times
        N_t = self(t) 

        # Initial conditions loss 
        IC_loss = (N_exact_0 - N_0).square().mean()
        
        # ODE loss
        if self.sparse:
            rhs = torch.sparse.mm(self.sparse_lamda_matrix, torch.transpose(N_t, 0, 1))
        else:
            rhs = torch.matmul(self.lamda_matrix, torch.transpose(N_t, 0, 1))

        ODE_loss = 0
        for i in range(len(self.initial)):
            Ni_t = N_t[:,i]
            Ni_t_dot = torch.autograd.grad(Ni_t, t, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(Ni_t))[0].transpose(0,1)
            ODE_loss += (Ni_t_dot - rhs[i,:]).square().mean()

        IC_loss = IC_loss.real
        ODE_loss = ODE_loss.real

        weights = np.array(weights, np.float64)
        weights /= 10**(int(np.log10(weights[0]))-20)
        return (weights[0]*IC_loss + weights[1]*ODE_loss)/sum(weights), IC_loss, ODE_loss


class DecayPINN(nn.Module, PINN):
    """
    PINN class for solving the decay problem.

    Attributes:
    -----------
    lamda : torch tensor
        Array of eigenvalues.
    type : str (default = "decay")
        Identifies the matrix type, decay or burnup.

    Methods:
    --------
    initialize_parameters(params_file=None)
        Sets the initial parameters of the network.
    forward(t)
        Computes the forward pass of the network.
    """
    def __init__(self, lamda_matrix, lamda, initial, seed=0):
        super(DecayPINN,self).__init__()
        assert not np.iscomplex(lamda).any(), "Lambda matrix is not a decay matrix."
        assert np.allclose(lamda_matrix, np.tril(lamda_matrix), atol=10**(-20)), "Lambda matrix is not lower triangular."
        PINN.__init__(self, lamda_matrix, initial, seed)

        self.lamda = torch.abs(torch.diagonal(self.lamda_matrix))   # Get the diagonal elements of the lamda matrix
        self.type = "decay"

        # Build NN model
        self.input_layer = nn.Linear(1, self.size, False, device=DEVICE)
        self.output_layer = nn.Linear(self.size, self.size, False, device=DEVICE)

        # Manually implemented linear layer with ability to use float64 datatypes.
        # self.input_layer = LinearLayer(1, self.size, DTYPE)
        # self.output_layer = LinearLayer(self.size, self.size, DTYPE)

        self.parameter_number = np.sum([np.prod([i for i in el.shape]) for el in self.parameters()])    #total number of parameters
        self.to(DEVICE)

    def initialize_parameters(self, params_file=None):
        """
        Sets the initial parameters of the network. Set the first layer weights
        to the eigenvalues of matrix A. If params_file is passed the weights
        of the second layer are imported from the file. Otherwise the weights
        are set to be triangular.

        Parameters:
        -----------
        params_file : str (default = None)
            File path to the file with network weights.
        """
        # self.input_layer.weight = self.input_layer.weight.double()
        # self.output_layer.weight = self.output_layer.weight.double()
        for i, param in enumerate(self.parameters()):
            # Set weights of input layer to lamda_i
            if i == 0:
                l = self.lamda.reshape(self.size, 1)
                if self.lamda[0] > 0: l = -l
                param.data = l.clone().detach().requires_grad_(False)
            if i == 1:
                if params_file is None:
                    # Set the initial output layer weights to lower triangular
                    param.data *= torch.as_tensor(np.tri(self.size, self.size), device=DEVICE)
                else:
                    # Load old parameters
                    param.data = torch.as_tensor(np.load(params_file))
                    print("Loaded old parameters")
        # Fix the weights of the input layer
        self.input_layer.weight.requires_grad = False

    def forward(self, t):
        """
        Compute the forward pass of the neural network.

        Parameters:
        -----------
        t : torch array
            Array of times.
        """
        t = torch.exp(self.input_layer(t))
        t = self.output_layer(t)
        return t
    
    
class BurnupPINN(nn.Module, PINN):
    """
    PINN class for solving the burnup problem.

    Attributes:
    -----------
    lamda : torch tensor
        Array of eigenvalues.
    type : str (default = "decay")
        Identifies the matrix type, decay or burnup.
    lamda_real : torch tensor
        Array of real parts of eigenvalues.
    lamda_imag : torch tensor
        Array of imaginary parts of eigenvalues.

    Methods:
    --------
    initialize_parameters(params_file=None)
        Sets the initial parameters of the network.
    forward(t)
        Computes the forward pass of the network.
    """
    def __init__(self, lamda_matrix, lamda, initial, seed=0):
        super(BurnupPINN,self).__init__()
        PINN.__init__(self, lamda_matrix, initial, seed)
        
        self.lamda = torch.as_tensor(lamda, dtype=torch.complex64, device=DEVICE)
        self.type = "burnup"

        # Convert lamda matrix to [real, imag]
        self.lamda_real = self.lamda.real
        self.lamda_imag = self.lamda.imag
        self.lamda = torch.cat([self.lamda.real, self.lamda.imag])

        # Build NN model
        self.input_layer1 = CosineLayer(1, self.size, self.lamda).to(device)
        self.input_layer2 = SineLayer(1, self.size, self.lamda).to(device)
        self.output_layer1 = nn.Linear(self.size, self.size, False, device=DEVICE)
        self.output_layer2 = nn.Linear(self.size, self.size, False, device=DEVICE)

        self.parameter_number = np.sum([np.prod([i for i in el.shape]) for el in self.parameters()])    #total number of parameters
        self.to(DEVICE)

    def initialize_parameters(self, params_file=None):
        """
        Sets the initial parameters of the network. Set the row and the column
        of elements with real eigenvalues to zero, aka remove these nuclides
        from the complex neural network.
        """
        # Set weights of input layer to lamda_i
        for i, param in enumerate(self.parameters()):
            if i == 3:  # complex output layer
                for j, eig in enumerate(self.lamda_imag):
                    if eig == 0:
                        param.data[j] *= 0
                        param.data[:,j] *= 0
        # Fix the weights of the input layer 
        self.input_layer1.weight.requires_grad = False
        self.input_layer2.weight.requires_grad = False

    def forward(self, t):
        """
        Compute the forward pass of the neural network.

        Parameters:
        -----------
        t : torch array
            Array of times.
        """
        t1 = self.input_layer1(t)
        t2 = self.input_layer2(t)
        t1 = self.output_layer1(t1)
        t2 = self.output_layer2(t2)
        t = t1 + t2
        return t
        

def train(network, t_max, t_steps=100, epochs=int(1e5), weights=[1,1], auto_stop=True, abstol=1e-6, IC_tol=1e-2, ODE_tol=1e-2, patience=5, max_epoch=50, batch_size=None, verbose=False):
    """
    Function that trains the network.

    Parameters:
    -----------
    network : BurnupPINN or DecayPINN class
        Instantiation of the BurnupPINN or DecayPINN class.
    t_max : float
        Final time of time evolution.
    t_steps : int (default = 100)
        Number of time steps used for trining.
    epochs : int
        Number of training epochs.
    weights : list
        List of weights, the first element is IC weight, the second is ODE weight.
    auto_stop : bool (default = True)
        If True stop the training when criteria are met. If False it will finish
        training after max_epoch.
    abstol : float
        Stopping criteria, describing the min change of loss.
    IC_tol : float
        IC_loss has to be below IC_tol
    ODE_tol : float
        ODE_loss has to be below ODE_tol
    patience : int
        After the model satisfies all the criteria, run the training for
        "patience" epochs more.
    max_epoch : int
        Maximal number of training epochs.
    batch_size : int (default None)
        Number of time steps used for calculating the loss. If bath_size 
        is given, a random sample of training point is used for each epoch. 
    verbose : bool (default False)
        Print the metrics and performance at each epoch.

    Returns:
    --------
    loss_history : list
        A list of losses at each epoch.
    epoch : int
        Number of epochs that it took for training.
    IC_loss_history : list
        A list of IC losses at each epoch.
    ODE_loss_history : list
        A list of ODE losses at each epoch.
    """

    optimizer = optim.LBFGS(network.parameters(), lr=0.1, max_iter=50, line_search_fn='strong_wolfe') # instead os SGD

    t = torch.linspace(0, t_max, t_steps, requires_grad=True, dtype=torch.float, device=DEVICE).reshape(-1,1)    #reshape it to a tensor [t_steps, 1]
   
    loss_history = []   # Container to store the loss
    IC_loss_history = []
    ODE_loss_history = []
    previous_loss = 0
    static_epoch = 0    # Number of epochs the loss does not improve

    for epoch in range(1, epochs+1):
        t_epoch = time.time()
        network.train() # Set the nn ready to be trained

        # Gradually increase t_max
        top_epoch = 10  # Typical number of epochs before convergence
        t_ = t[:min(len(t), int(len(t)/top_epoch*epoch))]
        loss, IC_loss, ODE_loss = network.lossFunction(t_, weights, batch_size)

        loss_history.append(loss.item())
        IC_loss_history.append(IC_loss.item())
        ODE_loss_history.append(ODE_loss.item())

        # Stopping criteria
        if auto_stop and abs(previous_loss-loss)<abstol and IC_loss < IC_tol and ODE_loss < ODE_tol:
                static_epoch += 1
                if static_epoch == patience:
                    return loss_history, epoch, IC_loss_history, ODE_loss_history
        else:
            static_epoch = 0    # Reset the counter

        optimizer.zero_grad()    # Reset the gradients before restating the backward propagation

        def closure():
            """Define this new closure to use the adaptive learning in LBFGS"""
            optimizer.zero_grad()
            loss, _, _ = network.lossFunction(t_, weights, batch_size)
            loss.backward(retain_graph=True)

            # Set gradient of upper triangle to zero, to fix the values to zero
            for i, param in enumerate(network.parameters()):
                if network.type == "decay":
                    if i == 1:
                        param.grad *= torch.as_tensor(np.tri(network.size, network.size), dtype=torch.float, device=DEVICE)
                elif network.type == "burnup":
                    if i == 3:
                        for j, eig in enumerate(network.lamda_imag):
                            if eig == 0:
                                param.grad[j] *= 0
                                param.grad[:,j] *= 0                    
            return loss
        
        optimizer.step(closure)

        epoch_duration = time.time() - t_epoch

        previous_loss = loss.item()

        # Save model
        save_model(network, "burnup_model.pt")

        # If verbose=True print epoch loss and epoch_duration
        if verbose: print(f'Epoch {epoch} \t|\tloss: {loss:.4}\t|\tIC loss {IC_loss.item():.4}\t|\tODE loss {ODE_loss.item():.4}\t|\tduration {epoch_duration:.2f} s')

        # In case of auto_stop=True check to not go over the number of max_epoch
        if auto_stop: 
            if epoch > max_epoch:
                return loss_history, epoch, IC_loss_history, ODE_loss_history

    return loss_history, epoch, IC_loss_history, ODE_loss_history


def save_model(network, filename):
    """
    Function that save PINN model. File extension should be .pt.

    Parameters:
    -----------
    network : BurnupPINN or DecayPINN class
        Instantiation of the BurnupPINN or DecayPINN class.
    filename : str
        Filepath to the storage location.
    """
    torch.save(network, filename)
    return    


def load_model(filename):
    """
    Function that loads PINN model. File extension should be .pt.

    Parameters:
    -----------
    filename : str
        Filepath to the storage location.
    """
    network = torch.load(filename, map_location=DEVICE)
    return network


def get_AnalyticalSolution(lamda, N_0, t):
    """"
    Compute the analytical solution of the Bateman equation. Only works for
    2 or 3 isotopes.

    Parameters:
    -----------
    lamda : numpy array
        Array of eigenvalues. 
    N_0 : list
        Initial conditions, only the first element should be non-zero.
    t : numpy array
        Array of times.

    Returns:
    --------
    List of solutions with shape (number of isotopes, number of time steps).
    """
    size = len(N_0)
    lamda = np.abs(lamda)
    N1_t = N_0[0] * np.exp(-lamda[0]*t)
    N2_t = N_0[0] * lamda[0]/(lamda[1]-lamda[0]) * (np.exp(-lamda[0]*t) - np.exp(-lamda[1]*t) )
    if size == 3:
        N3_t = N_0[0] - N1_t - N2_t
        return [N1_t, N2_t, N3_t]
    return [N1_t, N2_t]


def get_AnalyticalSolution_CRAM(lamda_matrix, N_0, ts):
    """"
    Compute the analytical solution of the Bateman equation. Only works for
    2 or 3 isotopes.

    Parameters:
    -----------
    lamda_matrix : numpy array
        Burnup or decay matrix
    N_0 : list
        Initial conditions.
    ts : numpy array
        Array of times.

    Returns:
    --------
    Array of solutions with shape (number of isotopes, number of time steps).
    """
    nstep = len(ts)
    n = lamda_matrix.shape[0]
    NsCRAM = np.zeros([nstep, n])
    for i in range(nstep):
        NsCRAM[i,:] = CRAM16(lamda_matrix,N_0,ts[i])
    return NsCRAM


def get_stiffness(lamda):
    """
    Computes the stiffness of decay or burnup matrix from an array of eigenvalues.

    Parameters:
    -----------
    lamda : numpy array
        Array of eigenvalues.
    """
    assert type(lamda) == list or type(lamda) == np.ndarray, "Pass in get_stiffness lamda as a list or np.ndarray"
    return max(abs(lamda))/min(abs(lamda))


def get_lamda_matrix(lamda, n_isotopes=3):
    """
    Generates decay matrix with given eigenvalues. Works for 2 or 3
    isotopes.

    Parameters:
    -----------
    lamda : numpy array
        Array of eigenvalues.
    n_isotopes : int
        Number of isotopes, either 2 or 3.
    """
    assert len(lamda) == 2 or len(lamda)==3, "Too many lamda are passed, this code works for 2 or 3 isotopes"
    if n_isotopes == 3:
        lamda_matrix = np.array([[-lamda[0],0,0],[lamda[0],-lamda[1],0],[0,lamda[1],0]], dtype=np.float64)
    else:
        lamda_matrix = np.array([[-lamda[0],0],[lamda[0],-lamda[1]]], dtype=np.float64)
    return lamda_matrix


def get_3x3_burnup_matrix(lamda):
    """
    Generates burnup matrix with given eigenvalues. Works for 3 isotopes.

    Parameters:
    -----------
    lamda : numpy array
        Array of eigenvalues.
    """
    assert len(lamda) == 3, "This code only works for matrices of size 3x3."
    assert np.sqrt(lamda[0])-np.sqrt(lamda[1]) < np.sqrt(lamda[2]) and np.sqrt(lamda[0])+np.sqrt(lamda[1]) > np.sqrt(lamda[2]), "Lamda values don't follow the criteria."
    lamda = np.abs(lamda)
    return np.array([[-lamda[0], 0, lamda[2]], [lamda[0], -lamda[1], 0], [0, lamda[1], -lamda[2]]])


def reduce_stiffness(A, stiff):
    '''
    Returns matrix A, where the stiffness has been reduced. This is achieved 
    by scaling the columns. Overall it is ensured that the stiffness ratio 
    |λ_max| / |λ_min| = stiff. Furthermore, stable nuclides are removed from 
    the matrix (i.e. nuclides with lambda = 0).
    
    Parameters:
    -----------
    A : matrix or sparse matrix
        It is assumed that the eigenvalues of A are its diagonal!
        (i.e. A is assumed to be triangular or permutable to triangular)
    stiff : float
        Desired stiffness
    '''
    A = A.toarray() if issparse(A) else A
    A = remove_zeros(A)
    lams = np.abs(A.diagonal())
    lmax = lams.max()
    lmin = lams.min()
    lmax_new = lmin * stiff
    lams_scaled = (lams - lmin) / (lmax - lmin) * (lmax_new - lmin) + lmin
    scale_matrix = np.diag(lams_scaled / lams)
    return np.matmul(A,scale_matrix)


def remove_zeros(A):
    '''
    Removes columns and rows of A where there is a 0 in the diagonal.
    If A is diagonal or triangular, this is equivalent to removing columns
    that correspond to an eigenvalue 0. If additionally all the remaining 
    diagonal entries are distinct, this operation is equivalent to 
    rendering A non-singular.

    Parameters:
    -----------
    A : numpy array
        Decay or burnup matrix.
    '''
    diag = A.diagonal()
    if not (0.0 in diag):
        return A
    return A[:, diag != 0][diag != 0, :]


def permute_matrix(A, atol=1e-20):
    """
    Permutes the matrix A into lower triangular, if that is possible.

    Parameters:
    -----------
    A : numpy array
        Decay matrix.
    atol : float (default = 1e-20)
        The tolerance between output matrix and triangular matrix.
    """
    print("Permuting matrix")
    max_cnt = 10*A.shape[0]
    cnt = 0
    while np.allclose(A, np.tril(A), atol=atol) == False:
        assert cnt < max_cnt, f"Could not permute matrix within max_cnt={max_cnt}"
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i < j and A[i,j] != 0:
                    A[[j,i]] = A[[i,j]]     # swap rows
                    A[:,[j,i]] = A[:,[i,j]] # swap columns
        cnt += 1
    return A


def fluctuate_equal_eigenvalues(A, sigma = 1e-3):
    """
    Slightly change the value of the eigenvalues that repeat.

    Parameters:
    -----------
    A : numpy array
        Lower triangular matrix.
    sigma : float (default = 1e-3)
        Permute the eigenvalue by 1 + gaussian with variance sigma.
    """
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i != j and A[i,i] == A[j,j]:
                A[j,j] = A[j,j]*(1+np.random.normal(0, sigma, 1)[0])
                if A[j,j] > 0:
                    raise ValueError("Not all eigenvalues are negative. Check the input matrix and the sigma value.")
    return A
