"""
The knapsack problem arises in resource allocation where the decision makers
have to choose from a set of non-divisible items, tasks or projects under 
scertain constraints, such as space, budget, time and etc. 
    
The optimization goal of this example is to maximize total values of items 
packaged into X containers, where each container has a limitation on weight, 
volume and radiation.
    
We compare solution time versus problem size between classical computing [1] 
and quantum [2] methods.    
    
Reference:
   [1] https://developers.google.com/optimization/bin/multiple_knapsack
   [2] https://github.com/dwave-examples/knapsack

"""
import os
import time
from numpy import random

# Generate a set of random data for experiment
def generate_random_dataset(
            bag_weight_limit=50, 
            bag_volume_limit=50, 
            bag_rad_limit=5):
    # generate random items
    values = random.uniform(10, 50, number_items)
    weights = random.uniform(9, 35, number_items)
    volume = random.uniform(5, 30, number_items)
    radiation = random.uniform(0, 3, number_items)

    data = {
        "values": values,
        "weights": weights,
        "volume": volume,
        "rad": radiation,
        "bag_weights": [bag_weight_limit] * number_bags,
        "bag_volume": [bag_volume_limit] * number_bags,
        "bag_radiations": [bag_rad_limit] * number_bags,
    }
    return data


def dataset_content(data):
    for i in range(number_items):
        print(
            "Item {} value {:.0f} weight {:.0f} volume {:.0f} radidation {:.0f}"\
            .format(
                i + 1,
                data["values"][i],
                data["weights"][i],
                data["volume"][i],
                data["rad"][i],
            )
        )


# Use Google OR-tools as classical LP solver
# Addapted from implementation in [1]
import ortools
from ortools.linear_solver import pywraplp


def display_alloc_c(x):
    for j in range(number_bags):
        bag_value, bag_weight, bag_volume, bag_rad = 0, 0, 0, 0
        items_in_bag = []
        for i in range(number_items):
            if x[i, j].solution_value() > 0:
                items_in_bag.append(i + 1)
                bag_value += data["values"][i]
                bag_volume += data["volume"][i]
                bag_weight += data["weights"][i]
                bag_rad += data["rad"][i]
        print(
            "Container ({}) packed items{}".format(int(j) + 1, items_in_bag),
            "values({:.0f}) weights({:.0f})".format(bag_value, bag_weight),
            "volume({:.0f}) radiation({:.0f})".format(bag_volume, bag_rad),
        )


def ortools_lp():
    t1 = time.time()

    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.set_time_limit(ortool_solver_max_milliseconds)

    # declare variables
    x = {}
    for i in range(number_items):
        for j in range(number_bags):
            x[(i, j)] = solver.IntVar(0, 1, "x_{}_{}".format(i, j))

        # one item being placed in max 1 bag
        solver.Add(sum(x[i, j] for j in range(number_bags)) <= 1)

    # constrain bag capacity
    for j in range(number_bags):
        # item numbers
        solver.Add(
            sum(x[i, j] * data["weights"][i] for i in range(number_items))
            <= data["bag_weights"][j]
        )
        # items' volume
        solver.Add(
            sum(x[i, j] * data["volume"][i] for i in range(number_items))
            <= data["bag_volume"][j]
        )
        # radiations
        solver.Add(
            sum(x[i, j] * data["rad"][i] for i in range(number_items))
            <= data["bag_radiations"][j]
        )

    #  objective function
    objective = solver.Objective()
    for i in range(number_items):
        for j in range(number_bags):
            objective.SetCoefficient(x[(i, j)], data["values"][i])
    objective.SetMaximization()

    t_prob_build = time.time() - t1
    print(f"LP problem built in {round(t_prob_build,0)} seconds")

    # solve problem
    solv = solver.Solve()
    t2 = time.time()

    if not os.path.isfile(logfile):
        with open(logfile, "w") as f:
            f.write(
                "method,sol_total_second,prob_build_second,packed_value,number_bags,number_items\n"
            )

    if solv != pywraplp.Solver.OPTIMAL:
        print("There is no optimal solution")
        with open(logfile, "a") as f:
            f.write(
                "orlp,{:.0f},{:.0f},{},{},None\n".format(
                    t2 - t1, t_prob_build, number_bags, number_items
                )
            )
    else:
        print(
            "Total packed values {:.0f}  ({} bags << {} items)".format(
                objective.Value(), number_bags, number_items
            ),
            "Solved end-to-end in {:.0f} seconds".format(t2 - t1),
        )
        if display_verbose:
            display_alloc_c(x)
        with open(logfile, "a") as f:
            f.write(
                "orlp,{:.0f},{:.0f},{},{},{:.0f}\n".format(
                    t2 - t1, t_prob_build, number_bags, 
                    number_items, objective.Value()
                )
            )


# Use D-wave hybrid quantum solver on Leap cloud
import pandas as pd
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel
from dimod import BinaryQuadraticModel, QuadraticModel
from dimod import Binary


def build_knapsack_cqm(data):

    cqm = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype="BINARY")
    constraint = QuadraticModel()

    # declare variables
    x = {
        (i, j): Binary("x_{}_{}".format(i, j))
        for i in range(number_items)
        for j in range(number_bags)
    }

    for i in range(number_items):
        for j in range(number_bags):
            obj.add_variable("x_{}_{}".format(i, j))
            obj.set_linear("x_{}_{}".format(i, j), -data["values"][i])

        # one item place in max. 1 bag
        cqm.add_constraint(
            sum(x[i,j] for j in range(number_bags))<=1.0,label="c-1-{}".format(i)
        )

    for j in range(number_bags):
        # items' weight
        cqm.add_constraint(
            sum(x[i, j] * data["weights"][i] for i in range(number_items))
            <= data["bag_weights"][j],
            label="c-2-{}".format(j),
        )
        # items' volume
        cqm.add_constraint(
            sum(x[i, j] * data["volume"][i] for i in range(number_items))
            <= data["bag_volume"][j],
            label="c-3-{}".format(j),
        )
        # items' radiations
        cqm.add_constraint(
            sum(x[i, j] * data["rad"][i] for i in range(number_items))
            <= data["bag_radiations"][j],
            label="c-4-{}".format(j),
        )

    cqm.set_objective(obj)
    return cqm


def parse_solution(sampleset, data):
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    if not len(feasible_sampleset):
        raise ValueError("No feasible solution found")

    best = feasible_sampleset.first

    packed_values = 0
    bag_items = {
        j: {"item": [], "value": 0, "weight": 0, "volume": 0, "rad": 0}
        for j in range(number_bags)
    }
    for k, v in best.sample.items():
        if v == 1 and "x" in str(k):
            (_, item, bag) = k.split("_")
            item, bag = int(item), int(bag)
            bag_items[bag]["item"].append(item + 1)
            bag_items[bag]["value"] += data["values"][item]
            bag_items[bag]["weight"] += data["weights"][item]
            bag_items[bag]["volume"] += data["volume"][item]
            bag_items[bag]["rad"] += data["rad"][item]
            packed_values += data["values"][item]

    return best.energy, packed_values, bag_items


def display_alloc_q(bag_items):
    for j, v in bag_items.items():
        print(
            "Container ({}) packed items{}".format(int(j) + 1, v["item"]),
            "values({:.0f}) weights({:.0f})".format(v["value"], v["weight"]),
            "volume({:.0f}) radiation({:.0f})".format(v["volume"], v["rad"]),
        )


def dwave_cqm():
    t1 = time.time()

    # build a CQM problem at local and submit to Leap for solving
    sampler = LeapHybridCQMSampler()
    cqm = build_knapsack_cqm(data)  # initialization
    t_prob_build = time.time() - t1
    print(
        f"QCM problem built in {round(t_prob_build,0)} seconds.",
        "Submit to hybrid solver on cloud ..."
    )

    # submit to cloud for solving
    sampleset = sampler.sample_cqm(cqm, label="Multiple Knapsacks")

    # fetch result at local
    q_energy, packed_values, bag_items = parse_solution(sampleset, data)
    t2 = time.time()

    print(
        "Total packed values {:.0f}  ({} containers << {} items)".format(
            packed_values, number_bags, number_items
        ),
        "Solved end-to-end in {:.0f} seconds".format(t2 - t1),
    )
    if display_verbose:
        display_alloc_q(bag_items)

    if not os.path.isfile(logfile):
        with open(logfile, "w") as f:
            f.write(
            "method,sol_total_second,prob_build_second,packed_value,number_bags,number_items\n"
            )
    with open(logfile, "a") as f:
        f.write(
            "qcm,{:.0f},{:.0f},{},{},{:.0f}\n".format(
                t2 - t1, t_prob_build, number_bags, number_items, packed_values
            )
        )


###
# Solve the multiple knapsack problems
###
ortool_solver_max_milliseconds = (
    1 * 60 * 60 * 1000
)  # wait maximum 1 hour for classical solver
logfile = "output/knapsack-multiple-bags.log"  # write result to file
display_verbose = False  # True display solution details


# First set PROBLEM SCALE
number_bags = int(input("Enter number of containers : "))
number_items = int(input("Enter number of items to choose from : "))
print(
    "({} choices of items for {} containers)".format(\
                            number_items, number_bags).upper()
)


# generate test date
data = generate_random_dataset()

print("\n")
print("Quantum Hybrid Solver".upper())
dwave_cqm()

print("\n")
print("Classical Solver".upper())
ortools_lp()
