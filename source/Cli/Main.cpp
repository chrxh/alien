#include <algorithm>
#include <iostream>

#include "CLI/CLI.hpp"

#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"
#include "Base/Resources.h"
#include "EngineImpl/SimulationControllerImpl.h"
#include "EngineInterface/Serializer.h"

int main(int argc, char** argv)
{
    try {
        CLI::App app{"Command-line interface for ALIEN v" + Const::ProgramVersion};

        //parse command line arguments
        std::string inputFilename;
        std::string outputFilename;
        std::string statisticsFilename;
        int timesteps = 0;
        app.add_option(
            "-i", inputFilename, "Specifies the name of the input file for the simulation to run. The corresponding .settings.json should also be available.");
        app.add_option("-o", outputFilename, "Specifies the name of the output file for the simulation to run.");
        app.add_option("-s", statisticsFilename, "Specifies the name of the csv-file containing the statistics.");
        CLI11_PARSE(app, argc, argv);

        //read input
        if (inputFilename.empty()) {
            std::cerr << "No input file given." << std::endl;
            return 1;
        }
        DeserializedSimulation simData;
        if (!Serializer::deserializeSimulationFromFiles(simData, inputFilename)) {
            std::cerr << "Could not read from input files." << std::endl;
            return 1;
        }

        //run simulation
        auto simController = std::make_shared<_SimulationControllerImpl>();
        simController->newSimulation(simData.auxiliaryData.timestep, simData.auxiliaryData.generalSettings, simData.auxiliaryData.simulationParameters);
        simController->setClusteredSimulationData(simData.mainData);
        for (int i = 0; i < timesteps; ++i) {
            simController->calcSingleTimestep();
        }

        //write output simulation file
        simData.auxiliaryData.timestep = static_cast<uint32_t>(simController->getCurrentTimestep());
        simData.mainData = simController->getClusteredSimulationData();
        if (outputFilename.empty()) {
            std::cerr << "No output file given." << std::endl;
            return 1;
        }
        if (!Serializer::serializeSimulationToFiles(outputFilename, simData)) {
            std::cerr << "Could not write to output files." << std::endl;
            return 1;
        }
    } catch (std::exception const& e) {
        std::cerr << "An uncaught exception occurred: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred." << std::endl;
    }
    return 0;
}
