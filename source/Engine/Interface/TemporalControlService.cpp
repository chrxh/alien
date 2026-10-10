#include "TemporalControlService.h"

#include <cmath>
#include <ranges>

#include <Base/Interface/Definitions.h>

#include "SimulationFacade.h"

void TemporalControlService::calcSingleTimestep()
{
    clearPreviousTimestepsOfOtherSessions();
    _previousTimesteps.emplace_back(createSnapshot());
    if (_previousTimesteps.size() > MaxNumPreviousTimesteps) {
        _previousTimesteps.pop_front();
    }
    _SimulationFacade::get()->calcTimesteps(1);
}

bool TemporalControlService::hasPreviousTimestep()
{
    clearPreviousTimestepsOfOtherSessions();
    return !_previousTimesteps.empty();
}

void TemporalControlService::restorePreviousTimestep()
{
    if (!hasPreviousTimestep()) {
        return;
    }
    applySnapshot(_previousTimesteps.back());
    _previousTimesteps.pop_back();
}

int TemporalControlService::getNumPreviousTimesteps()
{
    clearPreviousTimestepsOfOtherSessions();
    return toInt(_previousTimesteps.size());
}

void TemporalControlService::clearPreviousTimesteps()
{
    _previousTimesteps.clear();
}

void TemporalControlService::createFlashback()
{
    _flashback = createSnapshot();
}

bool TemporalControlService::hasFlashback() const
{
    return _flashback.has_value();
}

void TemporalControlService::restoreFlashback()
{
    if (!_flashback) {
        return;
    }
    applySnapshot(*_flashback);
    _SimulationFacade::get()->removeSelection();
    _previousTimesteps.clear();
}

TemporalControlService::Snapshot TemporalControlService::createSnapshot() const
{
    Snapshot result;
    result.timestep = _SimulationFacade::get()->getCurrentTimestep();
    result.realTime = _SimulationFacade::get()->getRealTime();
    result.data = _SimulationFacade::get()->getSimulationData();
    result.parameters = _SimulationFacade::get()->getSimulationParameters();
    return result;
}

void TemporalControlService::applySnapshot(Snapshot const& snapshot) const
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto const& origParameters = snapshot.parameters;

    if (origParameters.numLayers == parameters.numLayers) {
        for (auto i : std::views::iota(0, parameters.numLayers)) {
            restorePosition(
                parameters.layerPosition.layerValues[i],
                parameters.layerVelocity.layerValues[i],
                origParameters.layerPosition.layerValues[i],
                origParameters.layerVelocity.layerValues[i]);
        }
    }

    if (origParameters.numSources == parameters.numSources) {
        for (auto i : std::views::iota(0, parameters.numSources)) {
            restorePosition(
                parameters.sourcePosition.sourceValues[i],
                parameters.sourceVelocity.sourceValues[i],
                origParameters.sourcePosition.sourceValues[i],
                origParameters.sourceVelocity.sourceValues[i]);
        }
    }

    parameters.externalEnergy = origParameters.externalEnergy;
    auto simRunning = _SimulationFacade::get()->isSimulationRunning();
    if (simRunning) {
        _SimulationFacade::get()->pauseSimulation();
    }
    _SimulationFacade::get()->setCurrentTimestep(snapshot.timestep);
    _SimulationFacade::get()->setRealTime(snapshot.realTime);
    _SimulationFacade::get()->clear();
    _SimulationFacade::get()->setSimulationData(snapshot.data);
    _SimulationFacade::get()->setSimulationParameters(parameters);
    if (simRunning) {
        _SimulationFacade::get()->runSimulation();
    }
}

void TemporalControlService::restorePosition(
    RealVector2D& position,
    RealVector2D const& velocity,
    RealVector2D const& origPosition,
    RealVector2D const& origVelocity) const
{
    if (std::abs(velocity.x) > NEAR_ZERO || std::abs(velocity.y) > NEAR_ZERO || std::abs(origVelocity.x) > NEAR_ZERO || std::abs(origVelocity.y) > NEAR_ZERO) {
        position = origPosition;
    }
}

void TemporalControlService::clearPreviousTimestepsOfOtherSessions()
{
    auto sessionId = _SimulationFacade::get()->getSessionId();
    if (_sessionId != sessionId) {
        _previousTimesteps.clear();
    }
    _sessionId = sessionId;
}
