import pickle
import numpy as np
import matplotlib.pyplot as plt
from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN
from utils import *
from cmaes import CMA


if __name__ == "__main__":

    # ----------------------------------------------------------
    # 1) Initialize the Mujoco plant
    # ----------------------------------------------------------
    reacher = SequentialReacher(plant_xml_file="arm_model.xml")
    # reacher = SequentialReacher(plant_xml_file="one_joint_arm.xml")

    # ----------------------------------------------------------
    # 2) Initialize task
    # ----------------------------------------------------------
    env = SequentialReachingEnv(
        plant=reacher,
        target_duration={"mean": 3, "min": 1, "max": 6},
        num_targets=10,
        num_interneurons=20,
        loss_weights={
            "euclidean": 1,
            "manhattan": 0,
            "energy": 0,
            "ridge": 0,
            "lasso": 0
        }
    )

    # ----------------------------------------------------------
    # 3) Define RNN policy
    # ----------------------------------------------------------
    rnn = RNN(
        input_size=3 + reacher.num_sensors,
        hidden_size=25,
        output_size=env.num_interneurons,
        activation=tanh,
        alpha=reacher.model.opt.timestep / 0.01
    )

    # ----------------------------------------------------------
    # 4) Evolutionary optimization (CMA-ES)
    # ----------------------------------------------------------
    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)
    num_generations = 3 #10000
    fitnesses = []

    for gen in range(num_generations):
        solutions = []

        for i in range(optimizer.population_size):
            x = optimizer.ask()
            fitness = -env.evaluate(rnn.from_params(x), seed=gen)
            solutions.append((x, fitness))
            fitnesses.append((gen, i, fitness))
            print(f"#{gen}.{i}  Fitness: {fitness:.4f}")

        optimizer.tell(solutions)

        best_rnn = rnn.from_params(optimizer.mean)

        if gen % 10 == 0:
            env.evaluate(best_rnn, seed=0, render=False, log=True)
            env.plot()

        if gen % 1000 == 0:
            file = f"../../models/optimizer_gen_{gg}_cmaesv2.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(optimizer, f)

    # ----------------------------------------------------------
    # 5) Plot fitness over generations
    # ----------------------------------------------------------
    fitnesses = np.array(fitnesses)
    generations = np.unique(fitnesses[:, 0])
    avg_fitness = []
    std_fitness = []

    for gen in generations:
        gen_fitness = fitnesses[fitnesses[:, 0] == gen][:, 2]
        avg_fitness.append(np.mean(gen_fitness))
        std_fitness.append(np.std(gen_fitness))

    avg_fitness = np.array(avg_fitness)
    std_fitness = np.array(std_fitness)

    plt.figure()
    plt.plot(generations, avg_fitness, label="Average Fitness")
    plt.fill_between(
        generations,
        avg_fitness - std_fitness,
        avg_fitness + std_fitness,
        color="blue",
        alpha=0.2,
        label="Standard Deviation"
    )
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Objective Function Value (Loss)")
    plt.title("Fitness During CMA-ES Optimization")
    plt.show()
