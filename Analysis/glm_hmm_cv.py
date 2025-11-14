import argparse
import numpy as np
import numpy.random as npr
import ssm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, train_test_split
import os
import random
import pandas as pd 

def generate_synthetic_data(session_idx, glmhmm, num_trials, session_inputs):
    true_z, true_y = glmhmm.sample(num_trials, input=session_inputs)
    return true_z, true_y

def cross_val_at_seed(seed, n_states, splits, N_iters, inpts, true_choices, obs_dim, input_dim, num_categories):
    np.random.seed(seed)
    val_scores = []

    for train_indices, val_indices in splits:
        x_train_fold = [inpts[i] for i in train_indices]
        y_train_fold = [true_choices[i] for i in train_indices]
        x_val_fold = [inpts[i] for i in val_indices]
        y_val_fold = [true_choices[i] for i in val_indices]

        hmm_glm = ssm.HMM(n_states, obs_dim, input_dim, observations="input_driven_obs",
                          observation_kwargs=dict(C=num_categories), transitions="standard")
        
        _ = hmm_glm.fit(y_train_fold, inputs=x_train_fold, method="em", num_iters=N_iters, tolerance=10**-4)
        val_scores.append(hmm_glm.log_likelihood(y_val_fold, inputs=x_val_fold))
    
    cv_score = np.mean(val_scores)
    
    return seed, cv_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM-HMM Cross-validation on HPC")
    parser.add_argument('--n_states', type=int, required=True, help='Number of states')
    parser.add_argument('--nb_rand_initilizations', type=int, required=True, help='Number of random initializations')
    args = parser.parse_args()

    # Arguments
    n_states = args.n_states
    nb_rand_initilizations = args.nb_rand_initilizations

    # Seed for reproducibility
    npr.seed(42)

    # Parameters
    num_sess = 100
    num_trials_per_sess = 400
    input_dim = 2
    obs_dim = 1
    num_categories = 2
    N_iters = 200
    n_true_states = 7

    # Synthetic Data Generation (simplified for brevity, include your own generation logic)
    true_glmhmm = ssm.HMM(n_true_states, obs_dim, input_dim, observations="input_driven_obs", 
                        observation_kwargs=dict(C=num_categories), transitions="standard")
    
    gen_weights = np.array([
    [[6, 1]],
    [[2, -3]],
    [[2, 3]],
    [[-1, 4]],
    [[5, -2]],
    [[-3, 5]],
    [[4, 0]]
])

trans_mat = np.array([[
    [0.9, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
    [0.01, 0.9, 0.01, 0.02, 0.02, 0.02, 0.02],
    [0.01, 0.01, 0.9, 0.02, 0.02, 0.02, 0.02],
    [0.02, 0.02, 0.02, 0.9, 0.01, 0.01, 0.02],
    [0.02, 0.02, 0.02, 0.01, 0.9, 0.01, 0.02],
    [0.02, 0.02, 0.02, 0.01, 0.01, 0.9, 0.02],
    [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88]
]])
gen_log_trans_mat = np.log(trans_mat)

inpts = np.ones((num_sess, num_trials_per_sess, input_dim))
stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
inpts[:,:,0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess))
inpts = list(inpts)
results = Parallel(n_jobs=-1)(delayed(generate_synthetic_data)(sess, true_glmhmm, num_trials_per_sess, inpts[sess]) for sess in range(num_sess))
true_latents, true_choices = zip(*results)

# save the true latents and choices in a dictionary
synth_data = {
    "latents": true_latents,
    "choices": true_choices,
    "inputs": inpts
}

# save the synthetic data in a pickle file (path ../Data/synth_data.pkl)
pd.to_pickle(synth_data, "../Data/synth_data.pkl")

# Cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
splits = list(kf.split(inpts))
random_seeds = random.sample(range(1, 10000), nb_rand_initilizations)
results = Parallel(n_jobs=-1)(delayed(cross_val_at_seed)(seed, n_states, splits, N_iters, inpts, true_choices, obs_dim, input_dim, num_categories) for seed in random_seeds)

# Compile and print results
cv_scores_per_seed = {seed: cv_score for seed, cv_score in results}
cv_scores_per_seed_and_state = []
for seed, cv_score in cv_scores_per_seed.items():
    cv_scores_per_seed_and_state.append({'N_States': 2,'Seed': seed, 'CV_Score': cv_score})

pd.to_pickle(cv_scores_per_seed_and_state, "../Data/cv_scores_per_seed_and_state.pkl")

# %%

