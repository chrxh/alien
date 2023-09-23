#include <algorithm>
#include <iostream>

#include "CLI/CLI.hpp"

#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"
#include "Base/Resources.h"
#include "Base/StringHelper.h"
#include "Base/FileLogger.h"
#include "EngineImpl/SimulationControllerImpl.h"
#include "EngineInterface/Serializer.h"

namespace
{

    bool writeStatistics(SimulationController const& simController, std::string const& statisticsFilename)
    {
        auto statistics = simController->getStatistics();
        std::ofstream file;
        file.open(statisticsFilename, std::ios_base::out);
        if (!file) {
            return false;
        }

        auto writeLabelAllColors = [&file](auto const& name) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                file << ", " << name << " (color " << i << ")";
            }
        };
        file << "Time step";
        writeLabelAllColors("Cells");
        writeLabelAllColors("Self-replicators");
        writeLabelAllColors("Viruses");
        writeLabelAllColors("Cell connections");
        writeLabelAllColors("Energy particles");
        writeLabelAllColors("Total energy");
        writeLabelAllColors("Total genome cells");
        writeLabelAllColors("Created cells");
        writeLabelAllColors("Attacks");
        writeLabelAllColors("Muscle activities");
        writeLabelAllColors("Transmitter activities");
        writeLabelAllColors("Defender activities");
        writeLabelAllColors("Injection activities");
        writeLabelAllColors("Completed injections");
        writeLabelAllColors("Nerve pulses");
        writeLabelAllColors("Neuron activities");
        writeLabelAllColors("Sensor activities");
        writeLabelAllColors("Sensor matches");
        file << std::endl;

        auto writeIntValueAllColors = [&file](ColorVector<int> const& values) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                file << ", " << static_cast<uint64_t>(values[i]);
            }
        };
        auto writeInt64ValueAllColors = [&file](ColorVector<uint64_t> const& values) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                file << ", " << values[i];
            }
        };
        auto writeFloatValueAllColors = [&file](ColorVector<float> const& values) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                file << ", " << values[i];
            }
        };
        file << simController->getCurrentTimestep();
        writeIntValueAllColors(statistics.timeline.timestep.numCells);
        writeIntValueAllColors(statistics.timeline.timestep.numSelfReplicators);
        writeIntValueAllColors(statistics.timeline.timestep.numViruses);
        writeIntValueAllColors(statistics.timeline.timestep.numConnections);
        writeIntValueAllColors(statistics.timeline.timestep.numParticles);
        writeFloatValueAllColors(statistics.timeline.timestep.totalEnergy);
        writeInt64ValueAllColors(statistics.timeline.timestep.numGenomeCells);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numCreatedCells);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numAttacks);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numMuscleActivities);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numTransmitterActivities);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numDefenderActivities);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numInjectionActivities);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numCompletedInjections);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numNervePulses);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numNeuronActivities);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numSensorActivities);
        writeInt64ValueAllColors(statistics.timeline.accumulated.numSensorMatches);
        file << std::endl;
        file.close();
        return true;
    }
}

int main(int argc, char** argv)
{
    try {
        FileLogger fileLogger = std::make_shared<_FileLogger>();

        CLI::App app{"Command-line interface for ALIEN v" + Const::ProgramVersion};

        //parse command line arguments
        std::string inputFilename;
        std::string outputFilename;
        std::string statisticsFilename;
        int timesteps = 0;
        app.add_option(
            "-i", inputFilename, "Specifies the name of the input file for the simulation to run. The corresponding .settings.json should also be available.");
        app.add_option("-o", outputFilename, "Specifies the name of the output file for the simulation.");
        app.add_option("-t", timesteps, "The number of time steps to be calculated.");
        app.add_option("-s", statisticsFilename, "Specifies the name of the csv-file containing the statistics.");
        CLI11_PARSE(app, argc, argv);

        //read input
        std::cout << "Reading input" << std::endl;
        if (inputFilename.empty()) {
            std::cout << "No input file given." << std::endl;
            return 1;
        }
        DeserializedSimulation simData;
        if (!Serializer::deserializeSimulationFromFiles(simData, inputFilename)) {
            std::cout << "Could not read from input files." << std::endl;
            return 1;
        }

        //run simulation
        auto startTimepoint = std::chrono::steady_clock::now();

        auto simController = std::make_shared<_SimulationControllerImpl>();
        simController->newSimulation(simData.auxiliaryData.timestep, simData.auxiliaryData.generalSettings, simData.auxiliaryData.simulationParameters);
        simController->setClusteredSimulationData(simData.mainData);

        std::cout << "Device: " << simController->getGpuName() << std::endl;
        std::cout << "Start simulation" << std::endl;
        simController->calcTimesteps(timesteps);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTimepoint).count();
        auto tps = ms != 0 ? 1000.0f * toFloat(timesteps) / toFloat(ms) : 0.0f; 
        std::cout << "Simulation finished: " << StringHelper::format(timesteps) << " time steps, " << StringHelper::format(ms) << " ms, "
                  << StringHelper::format(tps, 1) << " TPS" << std::endl;
        

        //write output simulation file
        std::cout << "Writing output" << std::endl;
        simData.auxiliaryData.timestep = static_cast<uint32_t>(simController->getCurrentTimestep());
        simData.mainData = simController->getClusteredSimulationData();
        if (outputFilename.empty()) {
            std::cout << "No output file given." << std::endl;
            return 1;
        }
        if (!Serializer::serializeSimulationToFiles(outputFilename, simData)) {
            std::cout << "Could not write to output files." << std::endl;
            return 1;
        }

        //write output statistics file
        if (!statisticsFilename.empty()) {
            if (!writeStatistics(simController, statisticsFilename)) {
                std::cout << "Could not write to statistics file." << std::endl;
                return 1;
            }
        }

        std::cout << "Finished" << std::endl;
    } catch (std::exception const& e) {
        std::cerr << "An uncaught exception occurred: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred." << std::endl;
    }
    return 0;
}
