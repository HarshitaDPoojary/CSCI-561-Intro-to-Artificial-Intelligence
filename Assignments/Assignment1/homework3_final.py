import argparse
import os
import random
import math
import numpy as np
import time

dataset = []
matrix = []

class Location:
    '''
    Every city it creates a location instance with x,y,z, parameters of city and id for every city
    '''
    def __init__(self, id, x, y, z):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.id = int(id)

class Chromosome:
    '''
    For every path the class creates circulatar path, distance, fitness value 
    '''
    def __init__(self, node_list, matrix):
        path_list = list(node_list)+[list(node_list)[0]] #Add initial location to the path to get a full cycle cost
        self.chromosome = path_list
        chr_representation = []
        for i in range(0, len(path_list)):
            chr_representation.append(self.chromosome[i].id)
        self.chr_representation = chr_representation

        distance = 0
        for j in range(0, len(self.chr_representation)-1):  # get distances from the matrix
            distance += matrix[self.chr_representation[j]][self.chr_representation[j + 1]]
        self.cost = distance
        if (distance > 0):
            self.fitness_value = 1 / self.cost
        else:
            self.fitness_value = 0
        self.list = node_list
        self.item_list = self.chr_representation[:-1]

    def sort_priority(self):
        return self.fitness_value

def read_input_file (input):
    '''
    Check if file exists
    Get the first line for the number of cities.
    and Create an instance fo node class for all the cities mentioned in the input file.
    '''
    if not (os.path.exists(input)):
        return
    with open(input, 'r') as file:
        data = file.readlines()
        N = int(data[0])  
        for i in range(1, N+1):
            line = data[i].strip().split()
            x, y, z = line[0], line[1], line[2] 
            dataset.append(Location(id=i-1, x=x, y=y, z=z))
    return dataset, N

def create_distance_matrix(node_list, num_of_cities):
    '''
    Get the distance between two cities
    And store i in matrix for ease of access
    '''
    matrix = np.zeros((num_of_cities, num_of_cities))
    for i in range(num_of_cities):
        for j in range(i + 1, num_of_cities):
            dx = node_list[i].x - node_list[j].x
            dy = node_list[i].y - node_list[j].y
            dz = node_list[i].z - node_list[j].z
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            # since path A to B is same as path B to A 
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix


def tournament_selection(chromosomes, max_tnm):  
    '''
    get random candidates from population
    return the fittest chromosome as the parent
    '''
    selection_ix = np.random.randint(len(chromosomes)) 
    for _ in np.random.randint(0, len(chromosomes), max_tnm - 1):
        if chromosomes[_].fitness_value > chromosomes[selection_ix].fitness_value:
            selection_ix = _
    return chromosomes[selection_ix]



def rw_selection(chromosomes): 
    '''
    Create a wheel with higher probabilities for chromosome with hoioigher fitness value
    Pick a random value from the wheel

    '''
    total_fitness = sum(chromosome.fitness_value for chromosome in chromosomes)
    probabilities = [chromosome.fitness_value / total_fitness for chromosome in chromosomes]
    selected_index_arr = np.random.choice(len(chromosomes), size=2, p=probabilities)
    return chromosomes[selected_index_arr[0]],  chromosomes[selected_index_arr[1]]

def generate_initial_population_heuristic(node_list, population_size, matrix):
    '''
    Creates a combination of cities based on the distance to the adjacent city
    '''
    num_cities = len(node_list)
    initial_population = []

    for start_city in node_list:
        current_city = start_city
        path = [current_city]
        visited = [False] * num_cities
        visited[current_city.id] = True

        while len(path) < num_cities:
            nearest_city = None
            min_distance = float('inf')

            for next_city in node_list:
                if not visited[next_city.id] and matrix[current_city.id][next_city.id] < min_distance:
                    nearest_city = next_city
                    min_distance = matrix[current_city.id][next_city.id]

            if nearest_city is not None:
                path.append(nearest_city)
                visited[nearest_city.id] = True
                current_city = nearest_city

        initial_population.append(Chromosome(path, matrix))

        if len(initial_population) >= population_size:
            break
    return initial_population

def find_best_and_worst(generation):
    '''
    Sort the chromosomes based on the fitness value in descending order
    '''
    elite_percentage = 0.02
    ranked_population = [x for x in sorted(generation, key=lambda x: x.fitness_value, reverse=True)]
    num_elites = int(elite_percentage * len(generation)) + 1
    next_generation = ranked_population[:num_elites]
    worst_index = -1
    return ranked_population, next_generation, ranked_population[worst_index]


def crossover_mix(parent1, parent2):
    '''
    Pick random indexes and create child chromosome using the indices for the substring from the parent
    '''
    randomp1, randomp2 = random.sample(range(len(parent1.list)), 2)
    start = min(randomp1, randomp2)
    end = max(randomp1, randomp2)

    child_1_part1 = parent1.list[:start]
    child_1_part2 = parent1.list[end:]
    child_1 = child_1_part1 + child_1_part2
    child_2 = parent2.list[start:end+1]

    child_1_remain = [item for item in parent2.list if item not in child_1]
    child_2_remain = [item for item in parent1.list if item not in child_2]

    child_1 = child_1_part1 + child_1_remain + child_1_part2
    child_2 += child_2_remain

    return child_1, child_2 

def mutation(chromosome_list): 
    '''
    swap two nodes of the chromosome
    '''
    mut_idx1, mut_idx_2 = random.sample(range(len(chromosome_list)), 2)
    chromosome_list[mut_idx1], chromosome_list[mut_idx_2] = chromosome_list[mut_idx_2], chromosome_list[mut_idx1]
    return chromosome_list

def mutation_check(chromsome, matrix):
    '''
    check if mutation improved the child chromossome
    '''
    mutated_child = mutation(chromsome.list)
    mutated_chromosome = Chromosome(mutated_child, matrix)
    if mutated_chromosome.fitness_value > chromsome.fitness_value:
        return mutated_chromosome
    else:
        return chromsome

def local_search(chromosome, matrix):
    '''
    Apply 2 opt swap and check if it improves the child chromosome
    '''
    path = chromosome.list
    num_cities = len(path)
    improved_path = path[:]

    improvement = True
    while improvement:
        improvement = False

        for i in range(1, num_cities - 2):
            for j in range(i + 1, num_cities):
                if j - i == 1:
                    continue 

                current_distance = (
                    matrix[improved_path[i - 1].id][improved_path[i].id]
                    + matrix[improved_path[j - 1].id][improved_path[j].id]
                )

                new_distance = (
                    matrix[improved_path[i - 1].id][improved_path[j - 1].id]
                    + matrix[improved_path[i].id][improved_path[j].id]
                )

                if new_distance < current_distance:
                    improved_path[i:j] = reversed(improved_path[i:j])
                    improvement = True
        improved_child = Chromosome(improved_path, matrix)
        if(improved_child.fitness_value < chromosome.fitness_value):
            improved_child = chromosome

    return improved_child

def create_new_generation_with_heuristics(previous_generation, matrix, mutation_rate, combinations, total_combinations, count, prev_worst, time_limit=4):
    '''
    Add the elite chromosome to maintain good population
    For multiple iterations do the following:
        Select parents using roulette wheel selection
        Crossover and mutate to create child chromosome
        Add the child to the population only if its better than the worst population of previous generation
        Check if the iterations exceed the time limit
    If the population size is less than intended, add a few top chromosomes to the new generation
    '''
    current_iteration = combinations
    ranked_pop_prev, elites, worst = find_best_and_worst(previous_generation)    
    if prev_worst is not None and hasattr(prev_worst,"fitness_value"):
        if prev_worst.fitness_value > worst.fitness_value:
            worst = prev_worst  
    new_generation = set()
    elite_modify = local_search(elites[0],  matrix)
    
    new_generation.add(elite_modify)
    
    start_time = time.time()
        
    sa_count = 0
    mut_rate = mutation_rate
    
    while True :
        #parent_1 , parent_2= rw_selection(previous_generation)
        parent_1 = tournament_selection(previous_generation, len(previous_generation)+10)
        parent_2 = tournament_selection(previous_generation, len(previous_generation)+10)
        child_1, child_2 = crossover_mix(parent_1, parent_2)
        child_1 = Chromosome(node_list=child_1, matrix=matrix)
        child_2 = Chromosome(node_list=child_2, matrix=matrix)

        if count > 15:
            sa_count += 1
        if sa_count> 5:
            sa_count = 0
            mut_rate = 0.9

        if random.random() < mut_rate:
            mut_rate = mutation_rate
            child_1 = mutation_check(child_1, matrix)
            child_2 = mutation_check(child_2, matrix)

        child_1 = variable_adj_search(child_1, matrix, time_limit=4, max_iterations=30, no_improvement_threshold=10)  
        child_2 = variable_adj_search(child_2, matrix, time_limit=4, max_iterations=30, no_improvement_threshold=10) 
    
        
        if child_1.fitness_value < elites[0].fitness_value:
            child_1 = local_search(child_1, matrix)

        if child_2.fitness_value < elites[0].fitness_value:
            child_2 = local_search(child_2, matrix)

        
        if (child_1 not in previous_generation)  :
            if child_1.fitness_value > worst.fitness_value:
            #or left_count_2 > 5 or count > 4:
                new_generation.add(child_1)
                current_iteration += 1

        if (child_2 not in previous_generation) :
            if child_2.fitness_value > worst.fitness_value:
            #or left_count_2 > 5 or count > 4:
                new_generation.add(child_2)
                current_iteration += 1
            
        elapsed_time = time.time() - start_time
        if  current_iteration > total_combinations or len(new_generation) > len(previous_generation)-1 or elapsed_time > time_limit:
            break
    for element in ranked_pop_prev[1:]:
        if(len(new_generation)<len(previous_generation)):
            new_generation.add(element)
    return list(new_generation), elites[0], current_iteration, worst

def swap_adj(chromosome_list, matrix):
    '''
    Swap two random cities and create a new chromosome
    '''
    new_chromosome_list = chromosome_list[:]
    pos1, pos2 = random.sample(range(len(chromosome_list)), 2)
    new_chromosome_list[pos1], new_chromosome_list[pos2] = new_chromosome_list[pos2], new_chromosome_list[pos1]
    new_chromosome = Chromosome(new_chromosome_list, matrix)
    return new_chromosome

def reverse_adj(chromosome_list, matrix):
    '''
    Select a sub path from the path in the chromosome and reverse the path
    '''
    new_chromosome_list = chromosome_list[:]
    pos1, pos2 = random.sample(range(len(chromosome_list)), 2)
    start, end = min(pos1, pos2), max(pos1, pos2)
    new_chromosome_list[start:end + 1] = reversed(new_chromosome_list[start:end + 1])
    new_chromosome = Chromosome(new_chromosome_list, matrix)
    return new_chromosome


def relocate_adj(chromosome_list, matrix):
    '''
    Remove a city from the path and insert it to a new position
    '''
    new_chromosome_list = chromosome_list[:]
    city_to_relocate = random.choice(new_chromosome_list)
    current_position = new_chromosome_list.index(city_to_relocate)
    new_chromosome_list.pop(current_position)
    new_position = random.randint(0, len(new_chromosome_list))
    new_chromosome_list.insert(new_position, city_to_relocate)
    new_chromosome = Chromosome(new_chromosome_list, matrix)  # Adjust this based on your Chromosome class constructor
    return new_chromosome


def variable_adj_search(initial_solution, matrix, time_limit=60, max_iterations=1000, no_improvement_threshold=10):
    '''
    Call various mutations
    If the time limit exceeds or if there is no improvement in the chromosomes, then return the best chromosome generated 
    '''
    recent_sol = initial_solution
    current_trail = swap_adj
    no_imp_cnt = 0
    best_solution = recent_sol
    best_solution_cost = recent_sol.cost
    max_iterations_per_structure = 20
    start_time = time.time()

    for iteration in range(max_iterations):
        if iteration % max_iterations_per_structure == 0:
            if current_trail == swap_adj:
                current_trail = reverse_adj
            elif current_trail == reverse_adj:
                current_trail = relocate_adj
            else:
                break  # You can define a stopping condition here

        updated_sol = current_trail(recent_sol.list, matrix)

        if updated_sol.cost < recent_sol.cost:
            recent_sol = updated_sol
            no_imp_cnt = 0
            if updated_sol.cost < best_solution_cost:
                best_solution = updated_sol
                best_solution_cost = updated_sol.cost
        else:
            no_imp_cnt += 1

        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            break

        if no_imp_cnt >= no_improvement_threshold:
            break

    return best_solution

def write_chromosome_to_file(chromosome, filename="output.txt"):
    '''
    Add the path cost of best solution
    Add the coordinates of the city as per the route that generates minimum path cost
    '''
    try:
        with open(filename, "w") as file:
            if(chromosome is None):
                file.write('%.3f\n'%(float(0)))
                return
            file.write('%.3f\n'%(chromosome.cost))
            for city in (chromosome.chromosome):
                file.write(f"{city.x} {city.y} {city.z}\n")
    except Exception as e:
        print(f"Error writing to file: {str(e)}")

def main(input: str, output: str):
    '''
    Read the input file
    Implement error first mechanism for less cities
    Generate the parameters for population size, generations, timelimit as per the number of cities
    Get the initial population with heuristics
    Get new generations and the best path over the number of iterations or until the time limit
    Write the optimal path in the output file
    '''
    start_time = time.time()
    data, N = read_input_file(input)
    if N < 1:
        write_chromosome_to_file(None, output)
        return
    matrix = create_distance_matrix(data, N)
    if N == 1:
        ch = Chromosome(data, matrix)
        write_chromosome_to_file(ch, output)
        return    
    numbers_of_generations = max(100 * (N//100), 150)
    
    population_multiplier = 20
    pop_size = N * population_multiplier
    total_combinations = math.factorial(len(dataset))
    population_size = min(pop_size,total_combinations, 20)
    if population_size % 2 != 0:
        population_size += 1
    #print(population_size," ", numbers_of_generations)
    new_gen = generate_initial_population_heuristic(node_list=dataset, population_size=population_size, matrix=matrix)

    combinations = len(new_gen)
    previous_gen_best_cost = float('inf')
    count = 0
    mutation_rate = 0.9
    worst_solution = None
    max_count = min(max(numbers_of_generations//6, 50), 50)
    max_limit = 198
    if(N<=50):
        max_limit = 58
    elif (N<=100):
        max_limit = 73
    elif (N<=200):
        max_limit = 118

    for iteration in range(numbers_of_generations):
        t1 = time.time()
        new_gen, best_solution, combinations, worst_solution = create_new_generation_with_heuristics(new_gen, matrix, mutation_rate, combinations, total_combinations, count, worst_solution, 1.60)
        if time.time() - start_time >= max_limit:
            break
        if best_solution.cost == previous_gen_best_cost:
            count+=1
        elif best_solution.cost < previous_gen_best_cost:
            previous_gen_best_cost = best_solution.cost
            count = 0
            mutation_rate = 0.9
        if count > max_count * 0.89:
            mutation_rate = 0.4
        elif count > max_count * 0.7:
            mutation_rate = 0.8
             
        #print(f'{iteration}. gen cost= {best_solution.cost}, time= {time.time() - t1}')
             # + "path --> " + str(best_solution.item_list)
            
        elapsed_time = time.time() - start_time
        if combinations == total_combinations  or count > max_count or elapsed_time >= max_limit:
            break
    _, elites, _ = find_best_and_worst(new_gen)

    write_chromosome_to_file(elites[0], output)
    return best_solution

if __name__ == "__main__":
    #start_time = time.time()
    parser = argparse.ArgumentParser(
        description='Provide input and output path')
    parser.add_argument('--input', type=str, default='input.txt')
    parser.add_argument('--output', type=str, default='output.txt')
    args = vars(parser.parse_args())
    main(**args)
    #print("--- %s seconds ---" % (time.time() - start_time))