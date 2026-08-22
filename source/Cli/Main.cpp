#include <algorithm>
#include <filesystem>
#include <iostream>

#include <CLI/CLI.hpp>

#include <Base/AlienExceptions.h>
#include <Base/FileLogger.h>
#include <Base/GlobalSettings.h>
#include <Base/KernelProfiler.h>
#include <Base/KernelTracer.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

#include <EngineImpl/SimulationFacadeImpl.h>

#include <PersisterInterface/SerializerService.h>

int main(int argc, char** argv)
{
    auto error = false;
    try {
        FileLogger fileLogger = std::make_shared<_FileLogger>();

        CLI::App app{"Command-line interface for ALIEN v" + Const::ProgramVersion};

        // Parse command line arguments
        std::string inputFilename;
        std::string outputFilename;
        int timesteps = 0;
        bool debugMode = false;
        app.add_option("-i", inputFilename, "Specifies the name of the input file for the simulation to run.");
        app.add_option("-o", outputFilename, "Specifies the name of the output file for the simulation.");
        app.add_option("-t", timesteps, "The number of time steps to be calculated.");
        app.add_flag(
            "-d,--debug",
            debugMode,
            "Enables debug mode: this bypasses CUDA graphs and synchronizes after every kernel, so the simulation runs slower than normal but each kernel can "
            "be measured and traced individually. Two files are written: '"
                + Const::ProfileFilename.string() + "' holds the accumulated wall-clock time per kernel and '" + Const::TraceFilename.string()
                + "' holds the last kernel calls, which locates a kernel that hangs or triggers a driver timeout as the entry that is still marked as "
                  "running.");
        CLI11_PARSE(app, argc, argv);

        if (debugMode) {
            GlobalSettings::get().setDebugMode(true);
            KernelProfiler::get().init(Const::ProfileFilename);
            KernelTracer::get().init(Const::TraceFilename);
            std::cout << "Debug mode: writing " << std::filesystem::absolute(Const::ProfileFilename).string() << " and "
                      << std::filesystem::absolute(Const::TraceFilename).string() << std::endl;
        }

        // Read input
        std::cout << "Reading input" << std::endl;
        if (inputFilename.empty()) {
            std::cout << "No input file given." << std::endl;
            return 1;
        }
        SimulationDesc simData;
        if (!SerializerService::get().deserializeSimulationFromFiles(simData, inputFilename)) {
            std::cout << "Could not read from input files." << std::endl;
            return 1;
        }

        // Run simulation
        auto simulationFacade = std::make_shared<_SimulationFacadeImpl>();
        simulationFacade->newSimulation(simData._timestep, simData._worldSize, simData._simulationParameters);
        simulationFacade->setSimulationData(simData._mainData);
        simulationFacade->setStatisticsHistory(simData._statistics);
        simulationFacade->setRealTime(simData._realTime);
        std::cout << "Device: " << simulationFacade->getGpuName() << std::endl;
        std::cout << "Start simulation" << std::endl;

        // Measure the simulation loop only: loading and uploading the data would otherwise distort the TPS
        auto startTimepoint = std::chrono::steady_clock::now();
        simulationFacade->calcTimesteps(timesteps);

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTimepoint).count();
        auto tps = ms != 0 ? 1000.0f * toFloat(timesteps) / toFloat(ms) : 0.0f;
        std::cout << "Simulation finished: " << StringHelper::format(timesteps) << " time steps, " << StringHelper::format(ms) << " ms, "
                  << StringHelper::format(tps, 1) << " TPS" << std::endl;

        if (debugMode) {
            std::cout << std::endl << KernelProfiler::get().getReport() << std::endl;
        }


        // Write output simulation file
        std::cout << "Writing output" << std::endl;
        simData.timestep(simulationFacade->getCurrentTimestep())
            .mainData(simulationFacade->getSimulationData())
            .simulationParameters(simulationFacade->getSimulationParameters())
            .statistics(simulationFacade->getStatisticsHistory().getCopiedData())
            .realTime(simulationFacade->getRealTime());
        if (outputFilename.empty()) {
            std::cout << "No output file given." << std::endl;
            return 1;
        }
        if (!SerializerService::get().serializeSimulationToFiles(outputFilename, simData)) {
            std::cout << "Could not write to output files." << std::endl;
            return 1;
        }

        std::cout << "Finished" << std::endl;
    } catch (AlienException const& e) {
        log(Priority::Important, std::string("An exception occurred: ") + e.what());
        log(Priority::Important, "Callstack:\n" + e.getCallstack());
        error = true;
    } catch (std::exception const& e) {
        log(Priority::Important, std::string("An exception occurred: ") + e.what());
        error = true;
    } catch (...) {
        log(Priority::Important, std::string("An unknown exception occurred: "));
        error = true;
    }
    if (error) {
        std::cerr << LoggingService::get().getLogString();
        return 1;
    }
    return 0;
}
