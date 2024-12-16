import numpy as np

# Transition probabilities (a_ij)
a_ij = {
    'START': {'NN': 0.5, 'VB': 0.25, 'JJ': 0.25, 'RB': 0},
    'NN': {'STOP': 0.25, 'NN': 0.25, 'VB': 0.5, 'JJ': 0, 'RB': 0},
    'VB': {'STOP': 0.25, 'NN': 0.25, 'VB': 0, 'JJ': 0.25, 'RB': 0.25},
    'JJ': {'STOP': 0, 'NN': 0.75, 'VB': 0, 'JJ': 0.25, 'RB': 0},
    'RB': {'STOP': 0.5, 'NN': 0.25, 'VB': 0, 'JJ': 0.25, 'RB': 0},
}

# Emission probabilities (b_ik)
b_ik = {
    'NN': {'time': 0.1, 'flies': 0.01, 'fast': 0.01},
    'VB': {'time': 0.01, 'flies': 0.1, 'fast': 0.01},
    'JJ': {'time': 0, 'flies': 0, 'fast': 0.1},
    'RB': {'time': 0, 'flies': 0, 'fast': 0.1},
}

# POS Tags and Words
states = ['NN', 'VB', 'JJ', 'RB']
sentence = ['time', 'flies', 'fast']

# Viterbi Algorithm Implementation
def viterbi(sentence, states, start_prob, transition_prob, emission_prob):
    T = len(sentence)
    N = len(states)

    # Initialization
    V = np.zeros((T, N))  # Viterbi matrix
    backpointer = np.zeros((T, N), dtype=int)  # Backpointer to reconstruct path

    # Initialize with START probabilities
    for i, state in enumerate(states):
        V[0, i] = start_prob[state] * emission_prob[state].get(sentence[0], 0)

    # Recursion step
    for t in range(1, T):
        for j, state_j in enumerate(states):
            max_prob, max_state = 0, 0
            for i, state_i in enumerate(states):
                prob = V[t-1, i] * transition_prob[state_i].get(state_j, 0) * emission_prob[state_j].get(sentence[t], 0)
                if prob > max_prob:
                    max_prob, max_state = prob, i
            V[t, j] = max_prob
            backpointer[t, j] = max_state

    # Termination: Transition to STOP
    final_probs = [V[T-1, i] * transition_prob[states[i]].get('STOP', 0) for i in range(N)]
    best_final_state = np.argmax(final_probs)

    # Backtracking
    best_path = [best_final_state]
    for t in range(T-1, 0, -1):
        best_path.insert(0, backpointer[t, best_path[0]])

    # Convert state indices to state names
    best_path_states = [states[state] for state in best_path]
    return best_path_states, max(final_probs)

# Start probabilities from START
def get_start_prob(states):
    return {state: a_ij['START'].get(state, 0) for state in states}

# Run Viterbi
start_prob = get_start_prob(states)
most_likely_tags, max_prob = viterbi(sentence, states, start_prob, a_ij, b_ik)

# Output Results
print("Sentence:", sentence)
print("Most Likely POS Tags:", most_likely_tags)
print("Probability of Best Path:", max_prob)
