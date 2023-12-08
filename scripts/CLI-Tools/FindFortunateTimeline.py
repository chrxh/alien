import subprocess
import csv
import shutil
from datetime import datetime

#constants
ALIEN_PATH = 'd:/dev/alien/build/Release/'

TIMESTEPS_PER_ITERATION = 100000
TOTAL_ITERATIONS = 400
REPLICATORS_LOWER_BOUND = 200
MAX_TEMP_FILES = 10
MAX_RETRIES = 5

def runCli(iteration, fallBackLevel):
    for i in range(0, fallBackLevel + 1):
        inputFile = "cycle" + str((iteration - fallBackLevel + i + MAX_TEMP_FILES) % MAX_TEMP_FILES) + ".sim"
        outputFile = "cycle" + str((iteration - fallBackLevel + i + MAX_TEMP_FILES + 1) % MAX_TEMP_FILES) + ".sim"

        print(f"Execute {inputFile} -> {outputFile}")

        command = [ALIEN_PATH + "cli.exe", "-i", ALIEN_PATH + inputFile, "-o", ALIEN_PATH + outputFile, "-s", ALIEN_PATH + "statistics.csv", "-t", str(TIMESTEPS_PER_ITERATION)]
        subprocess.run(command)

def readStatistics():
    with open(ALIEN_PATH + 'statistics.csv', newline='') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=',', skipinitialspace=True))
        values = [float(rows[-1][i]) for i in range(8, 11)]
        return values

def main():
    iteration = 0
    fallBackLevel = 0
    retry = 0
    shutil.copy(ALIEN_PATH + 'initial.sim', ALIEN_PATH + 'cycle0.sim')
    shutil.copy(ALIEN_PATH + 'initial.settings.json', ALIEN_PATH + 'cycle0.settings.json')
    shutil.copy(ALIEN_PATH + 'initial.statistics.csv', ALIEN_PATH + 'cycle0.statistics.csv')

    while True:
        print("*****************************************")
        print(f"Iteration {iteration}, fallback level: {fallBackLevel}, retry: {retry}")
        print(datetime.now().strftime("%H:%M:%S"))
        print("*****************************************")

        runCli(iteration, fallBackLevel)
        numReplicators = readStatistics()
        print(f"Num replicators: {numReplicators}")

        if all(numReplicators[i] >= REPLICATORS_LOWER_BOUND for i in range(3)):
            iteration = iteration + 1
            retry = 0
            fallBackLevel = 0
        else:
            print("Retry")
            if retry < MAX_RETRIES:
                retry = retry + 1
            else:
                retry = 0
                if fallBackLevel < MAX_TEMP_FILES - 1:
                    fallBackLevel = fallBackLevel + 1
        if iteration >= TOTAL_ITERATIONS:
            break
    print("Script successfully executed")

if __name__ == "__main__":
    main()