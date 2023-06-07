import itertools

import pickle
import tqdm
import os

import hurwitz
import te_inference

# Function to generate time-series data for random matrices and save them as pickles. 
def gen_rand_timeseries(save_folder = "pickles/data/runs/", data_points = 100000, sample_steps = [0.1], Ns = [10], edge_probs = [0.2], mats_num = 100, mats_reps = 1, allow_sym = True, allow_underdet = False, comments = ""):
    
    for N, mat_num, ep, step_size in tqdm.tqdm(list(itertools.product(Ns, range(mats_num), edge_probs, sample_steps))):
        file_name = save_folder+"N{}_MatNum_{}_ss{:.2f}_ep{:.2f}{}{}.pkl".format(N, mat_num, step_size, ep, "_as"*allow_sym, "_au"*allow_underdet)
        
        if allow_sym:
            A = hurwitz.random_hurwitz(N, ep, allow_underdet=allow_underdet)
        else:    
            A = hurwitz.random_hurwitz_nonsym(N, ep)

        if os.path.isfile(file_name):
            pkl_obj = pickle.load(open(file_name, "rb"))
        else:
            pkl_obj = {
                "A" : A,
                "Sims" : [],
                "Comments" : comments
            }

        while len(pkl_obj["Sims"]) < mats_reps:
            # Generate simulated data for analysis. 
            data_np = hurwitz.run_process_jl(A, data_points*step_size, step_size)

            pkl_obj["Sims"].append(data_np)
        
        pkl_file = open(file_name, "wb")
        pickle.dump(pkl_obj, pkl_file)
        pkl_file.close()

def run_E_inference(save_folder = "pickles/results/inferences/", run_folder = "pickles/data/runs/", data_points = 100000, alphas = [0.05], sample_steps = [0.1], Ns = [10], edge_probs = [0.2], mats_num = 100, mats_reps = 1, allow_sym = True, allow_underdet = False):
    
    for N, mat_num, ep, step_size in tqdm.tqdm(list(itertools.product(Ns, range(mats_num), edge_probs, sample_steps))):
        save_file_name = save_folder+"N{}_ss{:.2f}_ep{:.2f}{}{}.pkl".format(N, step_size, ep, "_as"*allow_sym, "_au"*allow_underdet)
        if os.path.isfile(save_file_name):
            pkl_obj = pickle.load(open(save_file_name,"rb"))
        else:
            pkl_obj = dict()

        run_file_name = run_folder+"N{}_MatNum_{}_ss{:.2f}_ep{:.2f}{}{}.pkl".format(N, mat_num, step_size, ep, "_as"*allow_sym, "_au"*allow_underdet)
        if os.path.isfile(run_file_name):
            pkl_dt_obj = pickle.load(open(run_file_name,"rb"))
        if not os.path.isfile(run_file_name) or len(pkl_dt_obj["Sims"])<mats_reps:
            print("Generating run. Saving to {}".format(run_file_name))
            gen_rand_timeseries(run_folder, data_points=data_points, sample_steps=sample_steps, Ns=Ns, edge_probs=edge_probs, mats_num=mats_num, mats_reps=mats_reps, allow_sym=allow_sym, allow_underdet=allow_underdet)
            pkl_dt_obj = pickle.load(open(run_file_name,"rb"))
        
        pkl_obj[mat_num] = {
            "A" : pkl_dt_obj["A"],
            "Run Data" : run_file_name
        }

        for alpha in alphas: pkl_obj[mat_num][alpha] = []
        for rep in range(mats_reps):
            # Generate simulated data for analysis. 
            for alpha in alphas:
                E_inf = te_inference.perform_inference(pkl_dt_obj["Sims"][rep][0:data_points], gpu = True, alpha = alpha, report_edges=False)
                pkl_obj[mat_num][alpha].append(E_inf)
        
        pkl_save_file = open(save_file_name, "wb")
        pickle.dump(pkl_obj, pkl_save_file)
        pkl_save_file.close()