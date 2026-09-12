#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <csignal>
#include <cstdio>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <CLI/CLI.hpp>

#include <Base/AlienExceptions.h>
#include <Base/Console.h>
#include <Base/FileLogger.h>
#include <Base/GlobalSettings.h>
#include <Base/KernelProfiler.h>
#include <Base/KernelTracer.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

#include <ConsoleUi/ConsoleLiveOutput.h>
#include <ConsoleUi/ConsoleSimulationPanel.h>
#include <ConsoleUi/ConsoleWidgets.h>

#include <EngineInterface/SimulationFacade.h>

#include <EngineImpl/SimulationFacadeImpl.h>

#include <PersisterInterface/SerializerService.h>

namespace
{
    auto constexpr StatusUpdateInterval = std::chrono::milliseconds(100);
    auto constexpr MaxChunkGrowth = 4;
    auto constexpr StepLabelWidth = 10;
    auto constexpr CheckMark = "\xe2\x9c\x94";

    void printLines(std::vector<std::string> const& lines)
    {
        for (auto const& line : lines) {
            std::cout << line << std::endl;
        }
    }

    // Written directly, because a signal handler must not allocate or take the stdio lock
    void restoreCursorOnInterrupt(int signalNumber)
    {
        char constexpr ShowCursor[] = "\x1b[?25h";
        if (Console::isRichOutput()) {
#ifdef _WIN32
            _write(_fileno(stdout), ShowCursor, sizeof(ShowCursor) - 1);
#else
            auto const written = write(STDOUT_FILENO, ShowCursor, sizeof(ShowCursor) - 1);
            static_cast<void>(written);
#endif
        }
        std::signal(signalNumber, SIG_DFL);
        std::raise(signalNumber);
    }

    void printStep(std::string const& label, std::string const& value, std::string const& detail = std::string())
    {
        if (!Console::isRichOutput()) {
            std::cout << label << ": " << value << (detail.empty() ? "" : " (" + detail + ")") << std::endl;
            return;
        }
        auto padding = std::max(0, StepLabelWidth - Console::getVisibleLength(label));
        std::cout << "  " << ConsoleWidgets::createText(CheckMark, ConsolePalette::Success) << " " << ConsoleWidgets::createText(label, ConsolePalette::Label)
                  << std::string(padding, ' ') << ConsoleWidgets::createText(value, ConsolePalette::Value);
        if (!detail.empty()) {
            std::cout << "  " << ConsoleWidgets::createText(detail, ConsolePalette::Label);
        }
        std::cout << std::endl;
    }

    void printError(std::string const& message)
    {
        std::cout << std::endl << "  " << ConsoleWidgets::createText(message, ConsolePalette::Error) << std::endl;
    }

    uint64_t calcNextChunkSize(uint64_t chunkSize, std::chrono::steady_clock::duration const& chunkDuration)
    {
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(chunkDuration).count();
        if (microseconds <= 0) {
            return chunkSize * MaxChunkGrowth;
        }
        auto targetMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(StatusUpdateInterval).count();
        auto scaled = static_cast<uint64_t>(toDouble(chunkSize) * toDouble(targetMicroseconds) / toDouble(microseconds));
        return std::clamp(scaled, uint64_t(1), chunkSize * MaxChunkGrowth);
    }

    void calcTimestepsWithLiveOutput(SimulationFacade const& simulationFacade, uint64_t timesteps)
    {
        ConsoleLiveOutput liveOutput;
        ConsoleSimulationStatus status;
        status.totalTimesteps = timesteps;

        auto startTimepoint = std::chrono::steady_clock::now();
        auto lastUpdateTimepoint = startTimepoint;
        auto chunkSize = uint64_t(1);
        auto timestepsSinceUpdate = uint64_t(0);

        while (status.timestep < timesteps) {
            auto chunk = std::min(chunkSize, timesteps - status.timestep);

            auto chunkStartTimepoint = std::chrono::steady_clock::now();
            simulationFacade->calcTimesteps(chunk);
            auto now = std::chrono::steady_clock::now();

            status.timestep += chunk;
            timestepsSinceUpdate += chunk;
            chunkSize = calcNextChunkSize(chunkSize, now - chunkStartTimepoint);

            if (now - lastUpdateTimepoint < StatusUpdateInterval && status.timestep < timesteps) {
                continue;
            }
            auto intervalMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(now - lastUpdateTimepoint).count();
            lastUpdateTimepoint = now;

            status.duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTimepoint);
            status.tps = intervalMicroseconds > 0 ? toFloat(timestepsSinceUpdate) * 1.0e6f / toFloat(intervalMicroseconds) : 0.0f;
            timestepsSinceUpdate = 0;

            auto statistics = simulationFacade->getStatisticsEntry();
            status.numCells = statistics.objectStatistics.numCellObjects;
            status.numCreatures = 0;
            for (auto const& lineage : statistics.lineageEntries) {
                status.numCreatures += lineage.numCreatures;
            }
            status.numLineages = toUInt32(statistics.lineageEntries.size());

            liveOutput.update(ConsoleSimulationPanel::create(status), std::string());
        }
        liveOutput.close();
    }
}

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
        bool plainOutput = false;
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
        app.add_flag(
            "-p,--plain",
            plainOutput,
            "Disables colors, the banner and the live status panel and prints plain lines instead. This is switched on automatically when the output is "
            "redirected. The time steps are then calculated in a single run, which makes this the most accurate mode for measuring the TPS.");
        CLI11_PARSE(app, argc, argv);

        Console::init(plainOutput);
        std::signal(SIGINT, restoreCursorOnInterrupt);
        printLines(ConsoleWidgets::createBanner("artificial life environment  \xc2\xb7  v" + Const::ProgramVersion + "  \xc2\xb7  command line"));
        std::cout << std::endl;

        if (debugMode) {
            GlobalSettings::get().setDebugMode(true);
            KernelProfiler::get().init(Const::ProfileFilename);
            KernelTracer::get().init(Const::TraceFilename);
            printStep("profile", std::filesystem::absolute(Const::ProfileFilename).string());
            printStep("trace", std::filesystem::absolute(Const::TraceFilename).string());
        }

        // Read input
        if (inputFilename.empty()) {
            printError("No input file given.");
            return 1;
        }
        auto readStartTimepoint = std::chrono::steady_clock::now();
        SimulationDesc simData;
        if (!SerializerService::get().deserializeSimulationFromFiles(simData, inputFilename)) {
            printError("Could not read from input files.");
            return 1;
        }
        auto readDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - readStartTimepoint);
        printStep("input", inputFilename, StringHelper::format(readDuration.count()) + " ms");

        // Run simulation
        auto simulationFacade = std::make_shared<_SimulationFacadeImpl>();
        simulationFacade->newSimulation(simData._timestep, simData._worldSize, simData._simulationParameters);
        simulationFacade->setSimulationData(simData._mainData);
        simulationFacade->setStatisticsHistory(simData._statistics);
        simulationFacade->setRealTime(simData._realTime);
        printStep("device", simulationFacade->getGpuName());
        printStep(
            "world",
            std::to_string(simData._worldSize.x) + " x " + std::to_string(simData._worldSize.y),
            StringHelper::format(timesteps) + " time steps to calculate");
        std::cout << std::endl;

        // Measure the simulation loop only: loading and uploading the data would otherwise distort the TPS
        auto startTimepoint = std::chrono::steady_clock::now();
        if (ConsoleSimulationPanel::fitsIntoConsole()) {
            calcTimestepsWithLiveOutput(simulationFacade, timesteps);
        } else {
            simulationFacade->calcTimesteps(timesteps);
        }

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTimepoint).count();
        auto tps = ms != 0 ? 1000.0f * toFloat(timesteps) / toFloat(ms) : 0.0f;
        std::cout << std::endl;
        printStep("simulated", StringHelper::format(timesteps) + " time steps", StringHelper::format(ms) + " ms  " + StringHelper::format(tps, 1) + " TPS");

        if (debugMode) {
            std::cout << std::endl << KernelProfiler::get().getReport() << std::endl;
        }

        // Write output simulation file
        if (outputFilename.empty()) {
            printError("No output file given.");
            return 1;
        }
        auto writeStartTimepoint = std::chrono::steady_clock::now();
        simData.timestep(simulationFacade->getCurrentTimestep())
            .mainData(simulationFacade->getSimulationData())
            .simulationParameters(simulationFacade->getSimulationParameters())
            .statistics(simulationFacade->getStatisticsHistory().getCopiedData())
            .realTime(simulationFacade->getRealTime());
        if (!SerializerService::get().serializeSimulationToFiles(outputFilename, simData)) {
            printError("Could not write to output files.");
            return 1;
        }
        auto writeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - writeStartTimepoint);
        printStep("output", outputFilename, StringHelper::format(writeDuration.count()) + " ms");
        std::cout << std::endl;
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
        std::cout << Console::showCursor();
        printError("The simulation was aborted.");
        std::cerr << LoggingService::get().getLogString();
        return 1;
    }
    return 0;
}
