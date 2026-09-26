#include "McpTemporalTools.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/TemporalControlService.h>

#include <Network/McpArguments.h>
#include <Network/McpJson.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr MaxTimesteps = 100000;
    auto constexpr MaxTpsRestriction = 1000;
    auto constexpr MaxTimestepsPerBatch = uint64_t{10};
    auto constexpr TimeBudgetPerFrame = std::chrono::milliseconds(30);
}

std::vector<McpTool> McpTemporalTools::getTools(McpToolContext& context)
{
    _context = &context;

    return {
        McpTool{
            .name = "get_time_info",
            .description = "Returns the time step, the elapsed real time, whether the simulation is running, the current time steps per second (TPS), the "
                           "speed limit, the number of time steps that can be undone with step_backward and whether a flashback exists.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return getTimeInfo(); },
        },
        McpTool{
            .name = "run_simulation",
            .description = "Starts the simulation in ALIEN. Clears the time steps that could be undone with step_backward.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return runSimulation(); },
        },
        McpTool{
            .name = "pause_simulation",
            .description = "Pauses the simulation in ALIEN.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return pauseSimulation(); },
        },
        McpTool{
            .name = "step_forward",
            .description = std::format(
                "Calculates time steps while the simulation is paused. A single time step can be undone with step_backward (the last {} single "
                "steps are kept). Calculating several time steps at once clears the steps that could be undone. The call returns when all time steps "
                "are calculated, which can take a while for large counts.",
                TemporalControlService::MaxNumPreviousTimesteps),
            .inputSchema = McpSchema::object({{"count", McpSchema::integer("Number of time steps, default: 1", 1, MaxTimesteps)}}),
            .deferredHandler = [this](boost::json::object const& arguments, McpToolCompletion const& completion) { calcTimesteps(arguments, completion); },
        },
        McpTool{
            .name = "step_backward",
            .description = "Restores the state before the last single time step made with step_forward while the simulation is paused.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return restorePreviousTimestep(); },
        },
        McpTool{
            .name = "create_flashback",
            .description = "Saves the current world content and time in memory, so that load_flashback can restore it later. There is only one "
                           "flashback: creating a new one replaces the previous one. Loading a simulation also creates a flashback.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return createFlashback(); },
        },
        McpTool{
            .name = "load_flashback",
            .description = "Restores the world content and time saved by create_flashback. Static simulation parameters are not changed, while "
                           "positions of moving layers and radiation sources are restored.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return restoreFlashback(); },
        },
        McpTool{
            .name = "set_speed_limit",
            .description = "Limits the simulation speed to a number of time steps per second or removes the limit.",
            .inputSchema = McpSchema::object({{"tps", McpSchema::integer("Maximum time steps per second, omit to remove the limit", 1, MaxTpsRestriction)}}),
            .handler = [this](boost::json::object const& arguments) { return setSpeedLimit(arguments); },
        },
    };
}

void McpTemporalTools::process()
{
    if (!_pendingTimesteps) {
        return;
    }
    auto& pending = *_pendingTimesteps;
    auto finish = [&](McpToolResult const& result) {
        auto completion = pending.completion;
        _pendingTimesteps.reset();
        completion(result);
    };

    if (_SimulationFacade::get()->getSessionId() != pending.sessionId) {
        finish({.text = "The calculation was aborted because the simulation was replaced.", .isError = true});
        return;
    }
    if (_SimulationFacade::get()->isSimulationRunning()) {
        finish(
            {.text = std::format(
                 "The calculation was aborted after {} time steps because the simulation was started.", pending.numTimesteps - pending.numRemainingTimesteps),
             .isError = true});
        return;
    }

    auto startTime = std::chrono::steady_clock::now();
    while (pending.numRemainingTimesteps > 0 && std::chrono::steady_clock::now() - startTime < TimeBudgetPerFrame) {
        auto numTimesteps = std::min(pending.numRemainingTimesteps, MaxTimestepsPerBatch);
        _SimulationFacade::get()->calcTimesteps(numTimesteps);
        pending.numRemainingTimesteps -= numTimesteps;
    }
    if (pending.numRemainingTimesteps == 0) {
        finish({.text = std::format("Calculated {} time steps. {}", pending.numTimesteps, describeTime())});
    }
}

McpToolResult McpTemporalTools::getTimeInfo() const
{
    auto tpsRestriction = _SimulationFacade::get()->getTpsRestriction();
    auto result = boost::json::object{
        {"time_step", _SimulationFacade::get()->getCurrentTimestep()},
        {"real_time", StringHelper::format(_SimulationFacade::get()->getRealTime())},
        {"running", _SimulationFacade::get()->isSimulationRunning()},
        {"tps", _SimulationFacade::get()->getTps()},
        {"speed_limit_tps", tpsRestriction ? boost::json::value(*tpsRestriction) : boost::json::value(nullptr)},
        {"undoable_time_steps", TemporalControlService::get().getNumPreviousTimesteps()},
        {"flashback_exists", TemporalControlService::get().hasFlashback()},
    };
    return {.text = McpJson::serialize(result)};
}

McpToolResult McpTemporalTools::runSimulation() const
{
    if (_SimulationFacade::get()->isSimulationRunning()) {
        return {.text = "The simulation is already running."};
    }
    if (_pendingTimesteps) {
        throw std::runtime_error("Time steps are still being calculated.");
    }
    TemporalControlService::get().clearPreviousTimesteps();
    _SimulationFacade::get()->runSimulation();
    _context->showMessage("Run");
    return {.text = "The simulation is running."};
}

McpToolResult McpTemporalTools::pauseSimulation() const
{
    if (!_SimulationFacade::get()->isSimulationRunning()) {
        return {.text = "The simulation is already paused."};
    }
    _SimulationFacade::get()->pauseSimulation();
    _context->showMessage("Pause");
    return {.text = std::format("The simulation is paused. {}", describeTime())};
}

void McpTemporalTools::calcTimesteps(boost::json::object const& arguments, McpToolCompletion const& completion)
{
    auto count = McpArguments::getOptionalInt(arguments, "count", 1, MaxTimesteps).value_or(1);
    checkPaused();
    if (_pendingTimesteps) {
        throw std::runtime_error("Time steps are still being calculated.");
    }

    if (count == 1) {
        TemporalControlService::get().calcSingleTimestep();
        completion({.text = std::format("Calculated 1 time step. {}", describeTime())});
        return;
    }
    TemporalControlService::get().clearPreviousTimesteps();
    _pendingTimesteps = PendingTimesteps{
        .numRemainingTimesteps = static_cast<uint64_t>(count),
        .numTimesteps = static_cast<uint64_t>(count),
        .sessionId = _SimulationFacade::get()->getSessionId(),
        .completion = completion,
    };
    _context->showMessage(std::format("Calculating {} time steps ...", count));
}

McpToolResult McpTemporalTools::restorePreviousTimestep() const
{
    checkPaused();
    if (!TemporalControlService::get().hasPreviousTimestep()) {
        throw std::runtime_error("There is no time step to undo. Only single time steps made with step_forward while paused can be undone.");
    }
    TemporalControlService::get().restorePreviousTimestep();
    _context->onSelectionChanged();
    _context->showMessage("Previous time step loaded");
    return {.text = std::format("Restored the previous time step. {}", describeTime())};
}

McpToolResult McpTemporalTools::createFlashback() const
{
    TemporalControlService::get().createFlashback();
    _context->showMessage("Flashback created");
    return {.text = std::format("Created a flashback. {}", describeTime())};
}

McpToolResult McpTemporalTools::restoreFlashback() const
{
    if (!TemporalControlService::get().hasFlashback()) {
        throw std::runtime_error("There is no flashback. Create one with create_flashback first.");
    }
    if (_pendingTimesteps) {
        throw std::runtime_error("Time steps are still being calculated.");
    }
    TemporalControlService::get().restoreFlashback();
    _context->onSelectionChanged();
    _context->showMessage("Flashback loaded");
    return {.text = std::format("Loaded the flashback. {}", describeTime())};
}

McpToolResult McpTemporalTools::setSpeedLimit(boost::json::object const& arguments) const
{
    auto tps = McpArguments::getOptionalInt(arguments, "tps", 1, MaxTpsRestriction);
    _SimulationFacade::get()->setTpsRestriction(tps);
    return {.text = tps ? std::format("The simulation speed is limited to {} time steps per second.", *tps) : "The speed limit is removed."};
}

void McpTemporalTools::checkPaused() const
{
    if (_SimulationFacade::get()->isSimulationRunning()) {
        throw std::runtime_error("The simulation must be paused. Call pause_simulation first.");
    }
}

std::string McpTemporalTools::describeTime() const
{
    return std::format("The current time step is {}.", _SimulationFacade::get()->getCurrentTimestep());
}
