# ****************************************************************************************
# Runs a simulation and regularly creates save points.
# After creating a save point the number of replicators for each certain color is checked.
# If it drops below a certain value a previous savepoint is used. If this remains the case
# after a certain number of retries, older save points will be used.
# ****************************************************************************************

import subprocess
import csv
import shutil
from datetime import datetime

# paths
ALIEN_PATH = 'd:/dev/alien/build/Release/'  # path to alien.exe
SIM_PATH = ALIEN_PATH                       # path to input.sim
                                            # output.sim and temporary files will also be created there

# algorithm parameters
TIMESTEPS_PER_ITERATION = 100000
TOTAL_ITERATIONS = 400
REPLICATORS_LOWER_BOUND = 200
MAX_SAVEPOINTS = 10                         # max. number of temporary files which are used in cyclic order
MAX_RETRIES = 5                             # max. retries before resorting to previous savepoint


def get_base_filename(iteration, offset):
    return "savepoint" + str((iteration + offset + MAX_SAVEPOINTS) % MAX_SAVEPOINTS)


def run_cli(iteration, savepoint):
    for i in range(0, savepoint + 1):
        input_filename = get_base_filename(iteration, -savepoint + i) + ".sim"
        output_filename = get_base_filename(iteration, -savepoint + i + 1) + ".sim"
        print(f"Execute {input_filename} -> {output_filename}")
        command = [ALIEN_PATH + "cli.exe", "-i",
                   SIM_PATH + input_filename, "-o",
                   SIM_PATH + output_filename, "-t",
                   str(TIMESTEPS_PER_ITERATION)]
        subprocess.run(command)


def read_num_replicators(iteration, offset):
    filename = get_base_filename(iteration, offset) + ".statistics.csv"
    with open(SIM_PATH + filename, newline='') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=',', skipinitialspace=True))
        values = [float(rows[-1][i]) for i in range(9, 16)]
        return values


def copy_sim(input, output):
    shutil.copy(input + '.sim', output + '.sim')
    shutil.copy(input + '.settings.json', output + '.settings.json')
    shutil.copy(input + '.statistics.csv', output + '.statistics.csv')


def main():
    iteration = 0
    savepoint = 0
    retry = 0

    copy_sim(SIM_PATH + 'input', SIM_PATH +  get_base_filename(0, 0))

    num_replicators = read_num_replicators(0, 0)
    colors = [i for i in range(7) if num_replicators[i] > 1]
    print(f"The following colors will be considered: {colors}")

    while True:
        print("*******************************************")
        print(f"Iteration {iteration}, Savepoint: {savepoint}, Retry: {retry}")
        print(datetime.now().strftime("%H:%M:%S"))
        print("*******************************************")

        run_cli(iteration, savepoint)
        num_replicators = read_num_replicators(iteration, 1)
        print(f"Num replicators: {num_replicators}")

        if all(num_replicators[i] >= REPLICATORS_LOWER_BOUND for i in colors):
            iteration = iteration + 1
            retry = 0
            savepoint = 0
        else:
            print("Repeat")
            if retry < MAX_RETRIES:
                retry = retry + 1
            else:
                retry = 0
                if savepoint < MAX_SAVEPOINTS - 1:
                    savepoint = savepoint + 1
        if iteration >= TOTAL_ITERATIONS:
            copy_sim(SIM_PATH + get_base_filename(iteration, 0), SIM_PATH + 'output')
            print("Script successfully executed")
            print("output.sim written")
            break


if __name__ == "__main__":
    main()
