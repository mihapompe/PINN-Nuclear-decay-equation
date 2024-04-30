"""
Code for running the algorithm on NxN matrix.
"""

from Bateman_PINN import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-muted')
plt.style.use('seaborn-v0_8-ticks')
import time
import pandas as pd
import os
from RKsolve import RK45solve
from scipy.sparse import load_npz


def plot_concentrations_and_errors(t_eval, N_t, N_real):
    """
    Function that produces performance plots of the model.

    Parameters:
    -----------
    t_eval : torch tensor
        Array of times.
    N_t : torch tensor
        Array of solutions computed by the model.
    N_real : numpy array
        Array of solutions computed analytically or numerically, this serves
        as the ground truth.
    """
    w, h = 3,2
    t = t_eval.cpu().detach().numpy()
    N_t = N_t.cpu().detach().numpy()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    plt.figure(figsize=(12,18))
    
    # Get the indices of the three elements with the largest change. 
    max_change = [np.max(N_real[:,i])-np.min(N_real[:,i]) for i in range(N_real.shape[1])]
    indices = []
    for i in range(3):
        max_index = max(range(len(max_change)), key=max_change.__getitem__)
        indices.append(max_index)
        max_change[max_index] = float('-inf')

    for cnt, i in enumerate(indices):
        plt.subplot(w,h,1)
        plt.title("Concentration of isotopes over time")
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        skip = 500
        plt.plot(t, N_real[:, i], c=colors[cnt], label=f"CRAM {i+1}")
        plt.plot(t[::skip], N_t[::skip, i], "o", c=colors[cnt], label=f"PINN {i+1}")
        plt.legend()

        plt.subplot(w,h,2)
        plt.title("Error of isotopes over time")
        plt.xlabel("Time")
        plt.yscale("log")
        plt.ylabel("Absolute error")
        plt.plot(t, np.abs(N_t[:, i]-N_real[:, i]), c=colors[cnt], label=f"Isotope {i+1}")
        plt.legend()    

    plt.subplot(w,h,3)
    plt.title("Final concentrations")
    plt.plot(N_t[-1], "<", c=colors[0], label="PINN")
    plt.plot(N_real[-1], ">", c=colors[1],label="CRAM")
    plt.xlabel("Isotope")
    plt.yscale("log")
    plt.ylabel("Final concentration")
    plt.legend()

    plt.subplot(w,h,4)
    plt.title("Final absolute error")
    err = np.abs(N_t[-1]-N_real[-1])
    plt.plot(err,".")
    plt.plot(err.mean()*np.ones(err.shape), label=f"Mean {err.mean():.2}", c=colors[3])
    plt.legend()

    plt.xlabel("Isotope")
    plt.yscale("log")
    plt.ylabel("Final absolute error")

    error = np.abs(N_real - N_t)
    error_per_isotope = np.sum(error, axis = 0)
    error_per_time_step = np.sum(error, axis = 1)

    plt.subplot(w,h,5)
    plt.plot(error_per_isotope)
    plt.xlabel("Isotope")
    plt.ylabel("Sum of errors for all times")
    plt.title(f"Error per isotope")
    
    plt.subplot(w,h,6)
    plt.plot(error_per_time_step)
    plt.xlabel("Time")
    plt.ylabel("Sum of errors for all isotopes")
    plt.title(f"Error per time step")

    plt.savefig("results/conc_and_error_over_time10.png")
    return

def get_numerical_solution(lamda_matrix, t_max, t_steps, initial, solver = "CRAM"):
    """
    Function that computes the numerical solution of the problem.

    Parameters:
    -----------
    lamda_matrix : numpy array
        Burnup or decay matrix
    t_max : float
        Final time of time evolution.
    t_steps : int
        Number of time steps used for evaluation.
    initial : numpy array
        Initial conditions.
    solver : str (default = "CRAM")
        Choose the solver, either "CRAM" or "RK45"
    """
    print("Calculating analytical solution")
    ts = np.linspace(0, t_max, t_steps)
    time_1 = time.time()
    if solver == "CRAM":
        Ns = get_AnalyticalSolution_CRAM(lamda_matrix, initial, ts)
        time_cram = time.time()-time_1
        print(f"Time CRAM: {time_cram:.2f}")
    elif solver == "RK45":
        Ns = []
        for t_i in ts:
            Ns.append(RK45solve(lamda_matrix, initial, t_i, atol = 1e-8, rtol = 1e-5))
        Ns = np.array(Ns)
        time_cram = time.time()-time_1
        print(f"Time RK45: {time_cram:.2f}")
    else:
        raise NameError("The solver you selected doesn't exist, choose either 'CRAM' or 'RK45'.")
    return Ns, time_cram


def calculate_error(N_cal, N_real):
    """
    Function for calculating total and final absolute and relative errors.

    Parameters:
    -----------
    N_cal : numpy array
        Array of solutions computed by the model.
    N_real : numpy array
        Array of solutions computed analytically or numerically, this serves
        as the ground truth.
    """
    total_error = 0
    for i in range(N_cal.shape[1]):
        total_error += np.sum(np.abs(N_cal[:, i] - N_real[:, i]))
    final_error = np.sum(np.abs(N_cal[-1] - N_real[-1]))
    relative_final_error = final_error/np.sum(N_real[-1])
    return total_error, final_error, relative_final_error


def find_solution(lamda_matrix, lamda, prob_type, initial, weights, t_max, t_steps, seed, abstol, IC_tol, ODE_tol, patience, max_epoch, filename = "6", load_filename=None, save_filename=None):
    """
    Function that calculates the numerical solution, trains the model and 
    evaluates the solution.

    Parameters:
    -----------
    lamda_matrix : numpy array
        Burnup or decay matrix
    lamda : numpy array
        Array of eigenvalues.
    prob_type : str
        Problem type, "burnup" or "decay".
    initial : numpy array
        Initial conditions.
    weights : list
        List of weights, the first element is IC weight, the second is ODE weight.
    t_max : float
        Final time of time evolution.
    t_steps : int (default = 100)
        Number of time steps used for trining.
    seed : int
        Seed for all random number generators.
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
    filename : str
        Filename of this solution.
    load_filename : str
        Filename where the model is stored.
    save_filename : str
        Filename where we want the model to be stored.
    """

    # Calculate the numerical solution
    run_numerical = True
    if run_numerical:
        # Check if the numerical solution already exists for this problem.
        if not (os.path.exists(f"data/real_solutions/{filename}.npy") and os.path.exists(f"data/real_solutions/time_{filename}.npy")):
            N_real, time_cram = get_numerical_solution(lamda_matrix, t_max, t_steps, initial, solver="CRAM")
            np.save(f"data/real_solutions/{filename}.npy", N_real)
            np.save(f"data/real_solutions/time_{filename}.npy", time_cram)
        else:
            print("Load CRAM solution")
            N_real = np.load(f"data/real_solutions/{filename}.npy")
            time_cram = np.load(f"data/real_solutions/time_{filename}.npy")
    else:
        N_real = None

    # Train PINN
    time_pinn = time.time()
    if load_filename is not None:
        print("Load PINN")
        PINN = load_model(load_filename)
    else:
        if prob_type == "burnup":
            PINN = BurnupPINN(lamda_matrix, lamda, initial, seed=seed) # Create the PINN
        elif prob_type == "decay":
            PINN = DecayPINN(lamda_matrix, lamda, initial, seed=seed) # Create the PINN
        else:
            assert False, "Wrong value, prob_type can only be 'burnup' or 'decay'."
        PINN.initialize_parameters(params_file=None)
    print("Training PINN")
    loss_history, epochs, IC_loss_history, ODE_loss_history = train(PINN, t_max, t_steps, weights=weights, auto_stop=True, abstol=abstol, IC_tol=IC_tol, ODE_tol=ODE_tol, patience=patience, max_epoch=max_epoch, verbose=True)

    # Save model
    if save_filename is not None: save_model(PINN, save_filename)

    # Evaluate the population with the PINN
    PINN.eval() # Prepare the PINN to evaluate
    t_eval = torch.linspace(0, t_max, t_steps, requires_grad=True, device=DEVICE).reshape(-1,1)    #reshape it to a tensor [1, t_steps]
    N_t = PINN(t_eval)
    pinn_time = time.time()-time_pinn

    # Compare results
    if N_real is None:
        total_error, final_error, relative_final_error = 0,0,0
    else:
        total_error, final_error, relative_final_error = calculate_error(N_t.cpu().detach().numpy(), N_real)
    relative_error = total_error/np.sum(np.sum(N_real))
    plot_concentrations_and_errors(t_eval, N_t, N_real)
    
    print(f"""=================== CONFIG RESULTS ===================
        Eigenvalues: {lamda}
        Time CRAM: {time_cram:.2f} s
        Time PINN: {pinn_time:.2f} s
        IC/ODE weights: {weights}
        Stiffness: {get_stiffness(lamda):.2e}
        Number of epochs: {epochs}
        Loss: {loss_history[-1]:.2e}
        IC loss: {IC_loss_history[-1]:.2e}
        ODE loss: {ODE_loss_history[-1]:.2e}
        Absolute error: {total_error:.2f}
        Absolute final error: {final_error:.2e}""")
    if N_real is None:
        print(f'        Relative error: None')
        print(f'        Relative final error: None')
    else:
        print(f'        Relative error: {relative_error*100:.4} %')
        print(f'        Relative final error: {relative_final_error*100:.4} %')
    print("======================================================")

    # for i, param in enumerate(PINN.parameters()):
    #     print(param)    # Prints neural network parameters

    loss_history = np.pad(np.array(loss_history), (0, max_epoch+1-len(loss_history)))
    IC_loss_history = np.pad(np.array(IC_loss_history), (0, max_epoch+1-len(IC_loss_history)))
    ODE_loss_history = np.pad(np.array(ODE_loss_history), (0, max_epoch+1-len(ODE_loss_history)))
    out = [pinn_time, time_cram, epochs, total_error, final_error, relative_error, relative_final_error, *loss_history, *IC_loss_history, *ODE_loss_history]
    return out
    
def load_and_preprocess_matrix(prob_type, stiffness = None, n_max = None):
    """
    Load and preprocesses the burnup or decay matrix.

    Parameters:
    -----------
    prob_type : str
        Problem type, "burnup" or "decay".
    stiffness : float (default = None)
        Desired stiffness of the final matrix.
    n_max : int (default = None)
        Desired size of the final matrix.
    """
    PATH = "/Users/mihapompe/Documents/Projects/PINNs/miha-semester/data/A_matrices/"
    five = "Pu241_decay_n=5.npz"
    twelve = "Th232_decay_n=12.npz"
    seventeen = "Rn222_decay_n=17.npz"
    big = ["endfb68.npz",
        "endfb7.npz",
        "endfb71.npz",
        "endfb80.npz",
        "jeff31.npz",
        "jeff33.npz",
        "jeff311.npz",
        "burnup_matrix_endfb71.npz",
        "burnup250.npy",
        "burnup50.npy",
        "burnup713.npy"]

    if prob_type == "decay":
        # Load matrix from a file.
        lamda_matrix = load_npz(PATH+big[0]).toarray()
        lamda_matrix = remove_zeros(lamda_matrix)
        lamda_matrix = permute_matrix(lamda_matrix)
        lamda_matrix = fluctuate_equal_eigenvalues(lamda_matrix)
        if n_max is not None: lamda_matrix = lamda_matrix[:n_max,:n_max]

        # For 2x2 and 3x3 matrix
        # lamda = [10**10, 1]
        # lamda_matrix = get_lamda_matrix(lamda, n_isotopes=3)
    elif prob_type == "burnup":
        # Load matrix from a file.
        # lamda_matrix = np.load(PATH+"complex_eigens.npy")
        # lamda_matrix = np.load(PATH+big[-2])
        # lamda_matrix = load_npz(PATH+"medium_burnup_matrix/reduced_burnup.npz").toarray()
        # n_max = 47
        # n_min = 4
        # if n_max is not None: lamda_matrix = lamda_matrix[:n_max,:n_max]

        # For 3x3 complex matrices
        lamda = [1, 10**(-4), 1.0000001]
        lamda_matrix = get_3x3_burnup_matrix(lamda)

    print("==================== LAMBDA MATRIX ===================")
    print(f"Shape: {lamda_matrix.shape}")
    lamda = np.linalg.eig(lamda_matrix)[0]
    print(f"Initial stiffness: {get_stiffness(lamda):.2e}")
    if stiffness is not None: lamda_matrix = reduce_stiffness(lamda_matrix, stiffness)
    lamda = np.linalg.eig(lamda_matrix)[0]
    print(f"Reduced stiffness: {get_stiffness(lamda):.2e}")
    print("======================================================")
    return lamda_matrix, lamda


def initial_conditions(size):
    """
    Function that sets the initial conditions. Uncomment the code for
    a particular case.

    Parameters:
    -----------
    size : int
        Number of isotopes.
    """

    # Set all isotopes to 0, except the first on to 100
    # initial = np.zeros(size)
    # initial[0] = 100.

    # Set all isotopes to 1.
    initial = np.ones(size)

    # Load initial condition
    # PATH = "/Users/mihapompe/Documents/Projects/PINNs/miha-semester/data/A_matrices/"
    # with open(PATH+"medium_burnup_matrix/reduced_n0.txt", "r") as file:
    #     # Read all lines and convert them to floats
    #     lines = file.readlines()
    #     numbers = [float(line.strip()) for line in lines]
    # # Convert the list to a NumPy array
    # initial = np.array(numbers)
    # initial[initial < 1e15] = 0
    # initial /= np.max(initial)
    # assert len(initial) == size
    return initial


def run_model(weights, st, n_max):
    """
    Run the whole model with a given stiffness, size and weight ratio.

    Parameters:
    -----------
    weights : list
        The first element if the IC weight and the second on the ODE weight.
    st : float
        Desired stiffness of the final matrix.
    n_max : int
        Desired size of the final matrix.
    """
    t_max = 2.5
    t_steps = 10000 # Must be bigger than N (NxN lambda matrix)
    seed = 1
    prob_type = "decay"
    abstol =    1e-3
    IC_tol =    1e-2
    ODE_tol =   1e-2
    patience =  15
    stiffness = 10**st
    max_epoch = 50
    filename = f"CRAM_IC_{st}_{n_max}"
    
    if os.path.exists(f"data/run_analysis/result_{filename}_{weights[0]}_{weights[1]}.npy"):
        return None
        
    lamda_matrix, lamda = load_and_preprocess_matrix(prob_type, stiffness, n_max)
    initial = initial_conditions(len(lamda))
    out_ = find_solution(lamda_matrix, lamda, prob_type, initial, weights, t_max, t_steps, seed, abstol, IC_tol, ODE_tol, patience, max_epoch, filename)
    out = [weights[0], weights[1], weights[0]/weights[1], st, stiffness, n_max, *out_]
    np.save(f"data/run_analysis/result_{filename}_{weights[0]}_{weights[1]}.npy", np.array(out))
    return out

if __name__ == "__main__":
    t_max = 2.5
    t_steps = 10000 # Must be bigger than N (NxN lambda matrix)
    seed = 1
    weights = [1,1]
    prob_type = "decay"
    abstol =    1e-3
    IC_tol =    1e-2
    ODE_tol =   1e-2
    patience =  15
    st = 16
    stiffness = 10**st
    n_max = 50
    max_epoch = 50
    filename = f"CRAM_decay_{st}_{n_max}"
    
    lamda_matrix, lamda = load_and_preprocess_matrix(prob_type, stiffness, n_max)
    initial = initial_conditions(len(lamda))
    out_ = find_solution(lamda_matrix, lamda, prob_type, initial, weights, t_max, t_steps, seed, abstol, IC_tol, ODE_tol, patience, max_epoch, filename, load_filename=None, save_filename=None)
    out = [weights[0], weights[1], weights[0]/weights[1], st, stiffness, n_max, *out_]
