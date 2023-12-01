import argparse

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Process state weights and run optimized Viterbi algorithm.")
    parser.add_argument('--state_weights_file', default='state_weights.txt', help='Path to the state weights file')
    parser.add_argument('--state_action_state_weights_file', default='state_action_state_weights.txt', help='Path to the state-action-state weights file')
    parser.add_argument('--state_observation_weights_file', default='state_observation_weights.txt', help='Path to the state-observation weights file')
    parser.add_argument('--observation_actions_file', default='observation_actions.txt', help='Path to the observation-actions file')
    parser.add_argument('--output_file', default='states.txt', help='Path to the output file')
    return parser

def viterbi(observations_seq, action_seq, states, start_prob, trans_prob, emit_prob):
    alpha = [{}]
    path = {}

    for state in states:
        # initialization 
        alpha[0][state] = start_prob.get(state, 0) * emit_prob.get(state, {}).get(observations_seq[0], 0) # 1st observation used pi * theta
        path[state] = [state]

    for t in range(1, len(observations_seq)):
        alpha.append({})
        new_path = {}
        for curr_state in states:
            # alpha max(i)(curr_state) = max((alpha(i-1)(prev_state)*P(curr_state| action, prev_state)*theta(curr_state, observation) for all  prev_state combinations) 
            # get prev_state which gives this max probability for the current state
            # save path in the dictionary variable path, such that key is the curr state and the value is the array of states traveresed. 
            max_prob, best_prev_state = max(
                (alpha[t-1][prev_state] * trans_prob.get(action_seq[t-1], {}).get(prev_state, {}).get(curr_state, 0) * emit_prob.get(curr_state, {}).get(observations_seq[t], 0), prev_state)
                for prev_state in states
            )
            print(max_prob, best_prev_state)
            if max_prob > 0:
                alpha[t][curr_state] = max_prob
                new_path[curr_state] = path[best_prev_state] + [curr_state]
                print(new_path)

        if not new_path:
            break
        path = new_path

    n = len(observations_seq) - 1
    print(alpha)
    if alpha[n]:
        most_probable_state = max(alpha[n], key=alpha[n].get)
        return path[most_probable_state]
    else:
        return None
    

# Parsing the arguments
args = setup_arg_parser().parse_args()
state_weights_file_path = args.state_weights_file
state_action_state_weights_file_path = args.state_action_state_weights_file
state_observation_weights_file_path = args.state_observation_weights_file
observation_actions_file_path = args.observation_actions_file
output_file_path = args.output_file


### READ INPUTS ###
def read_state_weights(file_path):
    state_weights_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Extracting the default weight from the second line
        initials = lines[1].split()
        default_weight = initials[1] if len(initials) > 1 else 0
        default_weight = int(default_weight)
        for line in lines[1:]:
            if line.strip() and line[0] == '"':
                parts = line.strip().split()
                state = parts[0].strip('"')
                weight = int(parts[1]) if len(parts) > 1 and int(parts[1]) else default_weight
                weight = int(weight)
                state_weights_data.append((state, weight))

    return state_weights_data

state_weights_data = read_state_weights(state_weights_file_path)

def read_state_action_state_weights(file_path):
    state_action_state_weights_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        _, _, _, default_weight = lines[1].split()
        default_weight = int(default_weight)
        for line in lines[2:]:
            if line.strip() and line[0] == '"':
                parts = line.strip().split()
                state, action, next_state = parts[0].strip('"'), parts[1].strip('"'), parts[2].strip('"')
                weight = int(parts[3]) if len(parts) > 3 else default_weight
                state_action_state_weights_data.append((state, action, next_state, weight))
    return state_action_state_weights_data, default_weight

state_action_state_weights_data, default_state_action_state_weight = read_state_action_state_weights(state_action_state_weights_file_path)

def read_state_observation_weights(file_path):
    state_observation_weights_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        _, _, _, default_weight = lines[1].split()
        default_weight = int(default_weight)
        for line in lines[2:]:
            if line.strip() and line[0] == '"':
                parts = line.strip().split()
                state, observation = parts[0].strip('"'), parts[1].strip('"')
                weight = int(parts[2]) if len(parts) > 2 else default_weight
                state_observation_weights_data.append((state, observation, weight))
    return state_observation_weights_data, default_weight

state_observation_weights_data, default_state_observation_weight = read_state_observation_weights(state_observation_weights_file_path)


def read_observation_actions(file_path):
    observation_actions_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            if line.strip() and line[0] == '"':
                parts = line.strip().split('" "')
                if len(parts) == 2:
                    observation, action = parts[0].strip('"'), parts[1].strip('"')
                else:
                    observation = parts[0].strip('"')
                    action = None

                observation_actions_data.append((observation, action))

    return observation_actions_data

observation_actions_data = read_observation_actions(observation_actions_file_path)


### NORMALIZE DATA ###

# Extracting the states, observations and actions from the data
states = set([s for s, _ in state_weights_data])
observations = set([o for _, o, _ in state_observation_weights_data])
actions = set([a for _, a, _, _ in state_action_state_weights_data if a])



def normalize_weights(weights, default_weight= None, total_states= None):
    if default_weight is None:
        default_weight = 0
    total_weight = sum(weights.values())
    return {state: weight / total_weight for state, weight in weights.items()}

# Convert list of tuple state_weights_data to dict
state_weights = {state: weight for state, weight in state_weights_data}
start_prob = normalize_weights(state_weights)

def normalize_observation_weights(weights, default_state_observation_weight, total_states, total_observations):
    normalized_weights = {state: {} for state in states}
    
    #update values as per the given state_observation_weight
    for (state, observation), weight in weights.items():
        normalized_weights[state][observation] = weight
        
    #set default weight to the ones missing in state_observation_weight and normalize against each state
    for state in normalized_weights:
        total_weight = sum(normalized_weights[state].values())
        missing_obs_count = total_observations - len(normalized_weights[state])
        total_weight += missing_obs_count * default_state_observation_weight

        for observation in normalized_weights[state]:
            normalized_weights[state][observation] /= total_weight

        for missing_observation in observations - set(normalized_weights[state].keys()):
            normalized_weights[state][missing_observation] = default_state_observation_weight / total_weight

    return normalized_weights

# Convert list of tuples state_observation_weights_data to dict 
state_observation_weights = {(state, obs): weight for state, obs, weight in state_observation_weights_data}
emit_prob = normalize_observation_weights(state_observation_weights, default_state_observation_weight, len(states), len(observations))

def normalize_transition_weights(weights, default_state_action_state_weight, total_states):
    normalized_weights = {action: {state: {} for state in states} for action in actions}
    #update values as per the given state_action_state_weight 
    for (state, action, next_state), weight in weights.items():
        normalized_weights[action][state][next_state] = weight
        
    #set default value to the rest that is not available in state_action_state_weight
    for action in actions:
        for state in states:
            for next_state in states:
                if next_state not in normalized_weights[action][state]:
                    normalized_weights[action][state][next_state] = default_state_action_state_weight

    #normalize the weight against one action and one state
    for action in actions:
        for state in normalized_weights[action]:
            total_weight = sum(normalized_weights[action][state].values())
            for next_state in normalized_weights[action][state]:
                normalized_weights[action][state][next_state] /= total_weight
    return normalized_weights

# Convert list of tuples state_action_state_weights_data to dict 
state_action_state_weights = {(s, action, next_s): weight for s, action, next_s, weight in state_action_state_weights_data}
trans_prob = normalize_transition_weights(state_action_state_weights, default_state_action_state_weight, len(states))

observations_sequence = [obs for obs, _ in observation_actions_data]
action_sequence = [action for _, action in observation_actions_data]

most_likely_states = viterbi(observations_sequence, action_sequence[:-1], list(states), start_prob, trans_prob, emit_prob)
# -1 because after last observation there is no action. For the tuple (observation, action) we have only observation and action is None.
# We will use observation 1 for alpha 0(initialization), rest action k-1, observation k will be used for calculating other alpha values in iteration.

def write_states_to_file(states, file_path):
    with open(file_path, 'w') as file:
        file.write("states\n")
        file.write(f"{len(states)}\n")
        for state in states:
            file.write(f'"{state}"\n')


write_states_to_file(most_likely_states, output_file_path)