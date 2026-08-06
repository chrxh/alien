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
        bool profile = false;
        std::string traceFilename;  // Stays empty if --trace is given without a file name
        app.add_option("-i", inputFilename, "Specifies the name of the input file for the simulation to run.");
        app.add_option("-o", outputFilename, "Specifies the name of the output file for the simulation.");
        app.add_option("-t", timesteps, "The number of time steps to be calculated.");
        app.add_flag(
            "-p,--profile",
            profile,
            "Enables debug mode and measures the wall-clock time of each simulation kernel. A profiling report is printed after the run. This bypasses CUDA "
            "graphs and synchronizes after every kernel, so the absolute timings are slower than normal but reveal where the time is spent.");
        auto traceOption = app.add_option(
            "--trace,--trace-file",
            traceFilename,
            "Enables debug mode and writes the name and duration of every kernel call to a trace file, which keeps the last 100 calls. Intended for locating "
            "kernels that hang or trigger a driver timeout: such a kernel is the entry that is still marked as running. The entries are written unbuffered, so "
            "they survive a process that is killed instead of shut down. The file name is optional and defaults to kernel-trace.log.");
        traceOption->expected(0, 1);
        CLI11_PARSE(app, argc, argv);

        auto trace = traceOption->count() > 0;
        if (profile || trace) {
            GlobalSettings::get().setDebugMode(true);
        }
        if (profile) {
            KernelProfiler::get().setEnabled(true);
        }
        if (trace) {
            auto filename = !traceFilename.empty() ? traceFilename : std::string("kernel-trace.log");
            KernelTracer::get().init(filename);
            std::cout << "Writing kernel trace to " << std::filesystem::absolute(filename).string() << std::endl;
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
        auto startTimepoint = std::chrono::steady_clock::now();

        auto simulationFacade = std::make_shared<_SimulationFacadeImpl>();
        simulationFacade->newSimulation(simData._timestep, simData._worldSize, simData._simulationParameters);
        simulationFacade->setSimulationData(simData._mainData);
        simulationFacade->setStatisticsHistory(simData._statistics);
        simulationFacade->setRealTime(simData._realTime);
        std::cout << "Device: " << simulationFacade->getGpuName() << std::endl;
        std::cout << "Start simulation" << std::endl;

        simulationFacade->calcTimesteps(timesteps);

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTimepoint).count();
        auto tps = ms != 0 ? 1000.0f * toFloat(timesteps) / toFloat(ms) : 0.0f;
        std::cout << "Simulation finished: " << StringHelper::format(timesteps) << " time steps, " << StringHelper::format(ms) << " ms, "
                  << StringHelper::format(tps, 1) << " TPS" << std::endl;

        if (profile) {
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
