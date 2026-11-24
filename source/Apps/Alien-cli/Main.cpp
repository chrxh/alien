#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>

#include <Base/AlienExceptions.h>
#include <Base/Console.h>
#include <Base/CrashLogger.h>
#include <Base/ExitScopeGuard.h>
#include <Base/FileLogger.h>
#include <Base/GlobalSettings.h>
#include <Base/KernelProfiler.h>
#include <Base/KernelTracer.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

#include <ConsoleUi/ConsoleLiveOutput.h>
#include <ConsoleUi/ConsoleSimulationPanel.h>

#include <Engine/Interface/SimulationFacade.h>
#include <Engine/Interface/TemporalControlService.h>

#include <Engine/Impl/SimulationFacadeImpl.h>

#include <McpTools/Impl/McpToolsFacadeImpl.h>

#include <Network/McpServer.h>
#include <Network/NetworkService.h>
#include <Network/NetworkValidationService.h>

#include <Persister/Interface/PersisterFacade.h>
#include <Persister/Interface/SerializerService.h>

#include <Persister/Impl/PersisterFacadeImpl.h>

#include <Rendering/Interface/RenderingFacade.h>

#include <Rendering/Impl/RenderingFacadeImpl.h>

#include "CommandLineParser.h"
#include "ConsoleOutput.h"
#include "McpSession.h"
#include "PeriodicUploader.h"
#include "ServerSession.h"
#include "SimulationRunner.h"

namespace
{
    auto constexpr EmptyWorldSize = IntVector2D{1000, 500};

    void initDebugMode()
    {
        GlobalSettings::get().setDebugMode(true);
        KernelProfiler::get().init(Const::ProfileFilename);
        KernelTracer::get().init(Const::TraceFilename);
        ConsoleOutput::printStep("profile", std::filesystem::absolute(Const::ProfileFilename).string());
        ConsoleOutput::printStep("trace", std::filesystem::absolute(Const::TraceFilename).string());
    }

    bool startServerSession(std::optional<ServerSession>& serverSession, CommandLineArguments const& arguments, std::string const& gpuName)
    {
        if (!arguments.userName.empty() && arguments.password.empty()) {
            ConsoleOutput::printError("No password given. A login requires the user name via -u and the password via -p.");
            return false;
        }
        serverSession.emplace(arguments.userName, arguments.password, gpuName, !arguments.noCountAsActive);
        if (auto const& errorMessage = serverSession->getErrorMessage()) {
            ConsoleOutput::printError(*errorMessage);
            return false;
        }
        if (serverSession->isLoggedIn()) {
            ConsoleOutput::printStep("user", arguments.userName, NetworkService::get().getServerAddress());
        }
        return true;
    }

    bool isPeriodicUploadDesired(CommandLineArguments const& arguments)
    {
        return !arguments.uploadName.empty() || arguments.uploadInterval.has_value();
    }

    bool checkPeriodicUploadArguments(CommandLineArguments const& arguments, bool loggedIn)
    {
        if (!isPeriodicUploadDesired(arguments)) {
            return true;
        }
        if (arguments.uploadName.empty() || !arguments.uploadInterval.has_value()) {
            ConsoleOutput::printError("A periodic upload requires the base name via --upload-name and the interval via --upload-interval.");
            return false;
        }
        if (!loggedIn) {
            ConsoleOutput::printError("A periodic upload requires a login via -u and -p.");
            return false;
        }
        if (*arguments.uploadInterval < 1) {
            ConsoleOutput::printError("The upload interval must be at least one minute.");
            return false;
        }
        if (!NetworkValidationService::get().isStringValidForDatabase(arguments.uploadName)) {
            ConsoleOutput::printError("The upload name contains characters that are not allowed.");
            return false;
        }
        return true;
    }

    void initPersister(CommandLineArguments const& arguments)
    {
        if (!arguments.mcpServer && !isPeriodicUploadDesired(arguments)) {
            return;
        }
        _PersisterFacadeImpl::set(std::make_shared<_PersisterFacadeImpl>());
        _PersisterFacade::get()->setup();
    }

    void shutdownPersister()
    {
        if (auto persisterFacade = _PersisterFacade::get()) {
            persisterFacade->shutdown();
        }
    }

    bool startMcpSession(std::optional<McpSession>& mcpSession, ConsoleLiveOutput& liveOutput, CommandLineArguments const& arguments)
    {
        if (!arguments.mcpServer) {
            return true;
        }
        _McpToolsFacadeImpl::set(std::make_shared<_McpToolsFacadeImpl>());
        mcpSession.emplace(liveOutput, arguments.mcpPort);
        if (auto const& errorMessage = mcpSession->getErrorMessage()) {
            ConsoleOutput::printError(*errorMessage);
            return false;
        }
        ConsoleOutput::printStep("mcp", McpServer::createUrl(arguments.mcpPort));
        return true;
    }

    bool readSimulation(SimulationDesc& simData, CommandLineArguments const& arguments)
    {
        if (arguments.inputFilename.empty()) {
            if (arguments.mcpServer) {
                simData.worldSize(EmptyWorldSize).center(toRealVector2D(EmptyWorldSize) / 2);
                return true;
            }
            ConsoleOutput::printError("No input file given.");
            return false;
        }
        auto startTimepoint = std::chrono::steady_clock::now();
        if (!SerializerService::get().deserializeSimulationFromFiles(simData, arguments.inputFilename)) {
            ConsoleOutput::printError("Could not read from input files.");
            return false;
        }
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTimepoint);
        ConsoleOutput::printStep("input", arguments.inputFilename, StringHelper::format(duration.count()) + " ms");
        return true;
    }

    void initSimulation(SimulationFacade const& simulationFacade, SimulationDesc const& simData, std::optional<uint64_t> timesteps)
    {
        simulationFacade->newSimulation(simData._timestep, simData._worldSize, simData._simulationParameters);
        simulationFacade->setSimulationData(simData._mainData);
        simulationFacade->setStatisticsHistory(simData._statistics);
        simulationFacade->setRealTime(simData._realTime);
        ConsoleOutput::printStep("device", simulationFacade->getGpuName());
        ConsoleOutput::printStep(
            "world",
            std::to_string(simData._worldSize.x) + " x " + std::to_string(simData._worldSize.y),
            timesteps.has_value() ? StringHelper::format(*timesteps) + " time steps to calculate" : "no limit on the time steps");
    }

    void initPeriodicUpload(std::optional<PeriodicUploader>& periodicUploader, ConsoleLiveOutput& liveOutput, CommandLineArguments const& arguments)
    {
        if (!isPeriodicUploadDesired(arguments)) {
            return;
        }
        periodicUploader.emplace(liveOutput, arguments.uploadName, std::chrono::minutes(*arguments.uploadInterval));
        ConsoleOutput::printStep("upload", arguments.uploadName, "every " + StringHelper::format(*arguments.uploadInterval) + " min");
    }

    void runMcpSession(
        McpSession& mcpSession,
        SimulationFacade const& simulationFacade,
        SimulationDesc& simData,
        ConsoleLiveOutput& liveOutput,
        CommandLineArguments const& arguments)
    {
        std::optional<PeriodicUploader> periodicUploader;
        initPeriodicUpload(periodicUploader, liveOutput, arguments);

        ConsoleOutput::printHint(std::string("Press Q to stop the MCP server") + (arguments.outputFilename.empty() ? "." : " and write the output file."));
        ConsoleOutput::printBlankLine();

        mcpSession.setVisibleArea(simData._center, simData._zoom);
        TemporalControlService::get().createFlashback(std::move(simData._mainData));
        if (!arguments.inputFilename.empty()) {
            simulationFacade->runSimulation();
        }
        SimulationRunner::runWithMcpServer(simulationFacade, liveOutput, periodicUploader ? &*periodicUploader : nullptr);

        simData.worldSize(simulationFacade->getWorldSize()).center(mcpSession.getVisibleAreaCenter()).zoom(mcpSession.getZoomFactor());
        ConsoleOutput::printBlankLine();
    }

    std::chrono::milliseconds runSimulation(SimulationFacade const& simulationFacade, ConsoleLiveOutput& liveOutput, CommandLineArguments const& arguments)
    {
        std::optional<PeriodicUploader> periodicUploader;
        initPeriodicUpload(periodicUploader, liveOutput, arguments);

        auto singleRun = arguments.timesteps.has_value() && !ConsoleSimulationPanel::fitsIntoConsole() && !periodicUploader.has_value();
        if (!singleRun) {
            ConsoleOutput::printHint("Press Q to stop the simulation and write the output file.");
        }
        ConsoleOutput::printBlankLine();

        // Measure the simulation loop only: loading and uploading the data would otherwise distort the TPS
        auto startTimepoint = std::chrono::steady_clock::now();
        auto calculatedTimesteps = uint64_t(0);
        if (singleRun) {
            simulationFacade->calcTimesteps(*arguments.timesteps);
            calculatedTimesteps = *arguments.timesteps;
        } else {
            calculatedTimesteps = SimulationRunner::calcTimestepsWithLiveOutput(
                simulationFacade, arguments.timesteps, liveOutput, periodicUploader ? &*periodicUploader : nullptr);
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTimepoint);

        auto ms = elapsed.count();
        auto tps = ms != 0 ? 1000.0f * toFloat(calculatedTimesteps) / toFloat(ms) : 0.0f;
        ConsoleOutput::printBlankLine();
        ConsoleOutput::printStep(
            "simulated", StringHelper::format(calculatedTimesteps) + " time steps", StringHelper::format(ms) + " ms  " + StringHelper::format(tps, 1) + " TPS");
        return elapsed;
    }

    bool writeSimulation(SimulationFacade const& simulationFacade, SimulationDesc& simData, std::string const& outputFilename)
    {
        if (outputFilename.empty()) {
            ConsoleOutput::printError("No output file given.");
            return false;
        }
        auto startTimepoint = std::chrono::steady_clock::now();
        simData.timestep(simulationFacade->getCurrentTimestep())
            .mainData(simulationFacade->getSimulationData())
            .simulationParameters(simulationFacade->getSimulationParameters())
            .statistics(simulationFacade->getStatisticsHistory().getCopiedData())
            .realTime(simulationFacade->getRealTime());
        if (!SerializerService::get().serializeSimulationToFiles(outputFilename, simData)) {
            ConsoleOutput::printError("Could not write to output files.");
            return false;
        }
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTimepoint);
        ConsoleOutput::printStep("output", outputFilename, StringHelper::format(duration.count()) + " ms");
        return true;
    }
}

int main(int argc, char** argv)
{
    FileLogger fileLogger = std::make_shared<_FileLogger>();
    CrashLogger crashLogger = std::make_shared<_CrashLogger>();

    auto error = false;
    try {
        CommandLineArguments arguments;
        if (auto const& exitCode = CommandLineParser::parse(arguments, argc, argv)) {
            return *exitCode;
        }

        Console::init(arguments.plainOutput);
        ConsoleOutput::installInterruptHandler();
        ConsoleOutput::clearScreen();
        ConsoleOutput::printBanner();
        if (arguments.debugMode) {
            initDebugMode();
        }

        _RenderingFacadeImpl::set(std::make_shared<_RenderingFacadeImpl>());
        ExitScopeGuard shutdownRendering([] { _RenderingFacade::get()->shutdown(); });

        _SimulationFacadeImpl::set(std::make_shared<_SimulationFacadeImpl>());
        ExitScopeGuard closeSimulation([] { _SimulationFacade::get()->closeSimulation(); });
        auto simulationFacade = _SimulationFacade::get();

        std::optional<ServerSession> serverSession;
        if (!startServerSession(serverSession, arguments, simulationFacade->getGpuName())) {
            return 1;
        }
        if (!checkPeriodicUploadArguments(arguments, serverSession->isLoggedIn())) {
            return 1;
        }

        initPersister(arguments);
        ExitScopeGuard persisterGuard(shutdownPersister);

        ConsoleLiveOutput liveOutput;
        std::optional<McpSession> mcpSession;
        if (!startMcpSession(mcpSession, liveOutput, arguments)) {
            return 1;
        }

        SimulationDesc simData;
        if (!readSimulation(simData, arguments)) {
            return 1;
        }
        initSimulation(simulationFacade, simData, arguments.timesteps);

        if (mcpSession) {
            runMcpSession(*mcpSession, simulationFacade, simData, liveOutput, arguments);
            mcpSession.reset();
        } else {
            auto elapsed = runSimulation(simulationFacade, liveOutput, arguments);
            simulationFacade->setRealTime(simData._realTime + elapsed);
        }
        shutdownPersister();
        if (arguments.debugMode) {
            ConsoleOutput::printBlock(KernelProfiler::get().getReport());
        }

        auto outputDesired = !arguments.mcpServer || !arguments.outputFilename.empty();
        if (outputDesired && !writeSimulation(simulationFacade, simData, arguments.outputFilename)) {
            return 1;
        }
        ConsoleOutput::printBlankLine();
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
        ConsoleOutput::printError("The simulation was aborted.");
        std::cerr << LoggingService::get().getLogString();
        return 1;
    }
    return 0;
}
