#include "BalancerController.h"

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/StatisticsData.h"

namespace
{
    auto constexpr AdaptionInterval = 10000;
    auto constexpr AdaptionRatio = 2;
    auto constexpr AdaptionFactor = 0.9;
}

_BalancerController::_BalancerController(SimulationController const& simController)
    : _simController(simController)
{}

void _BalancerController::process()
{
    auto const& parameters = _simController->getSimulationParameters();
    if (parameters.adaptiveCellMaxAge) {
        initializeIfNecessary();
        doAdaption();
    }

    saveLastState();
}

void _BalancerController::initializeIfNecessary()
{
    auto const& parameters = _simController->getSimulationParameters();
    auto needsInitialization = false;
    for (int i = 0; i < MAX_COLORS; ++i) {
        if (parameters.cellMaxAge[i] != _lastCellMaxAge[i]) {
            needsInitialization = true;
        }
    }
    if (parameters.adaptiveCellMaxAge != _lastAdaptiveCellMaxAge) {
        needsInitialization = true;
    }

    if (needsInitialization) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            _cellMaxAge[i] = parameters.cellMaxAge[i];
        }
        startNewMeasurement();
    }
}

void _BalancerController::doAdaption()
{
    auto const& parameters = _simController->getSimulationParameters();
    auto statistics = _simController->getStatistics();
    for (int i = 0; i < MAX_COLORS; ++i) {
        _numReplicators[i] += statistics.timeline.timestep.numSelfReplicators[i];
    }
    if (_simController->getCurrentTimestep() - *_lastTimestep > parameters.adaptiveCellMaxAgeInterval) {
        uint64_t maxReplicators = 0;
        uint64_t averageReplicators = 0;
        int colorOfMaxReplicators = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (maxReplicators < _numReplicators[i]) {
                maxReplicators = _numReplicators[i];
                colorOfMaxReplicators = i;
            }
            averageReplicators += _numReplicators[i];
        }
        averageReplicators /= MAX_COLORS;
        if (maxReplicators / averageReplicators > AdaptionRatio) {
            _cellMaxAge[colorOfMaxReplicators] *= AdaptionFactor;
            for (int i = 0; i < MAX_COLORS; ++i) {
                if (i != colorOfMaxReplicators) {
                    _cellMaxAge[i] /= std::pow(AdaptionFactor, 1.0 / (MAX_COLORS - 1));
                }
            }

            auto parameters = _simController->getSimulationParameters();
            for (int i = 0; i < MAX_COLORS; ++i) {
                parameters.cellMaxAge[i] = toInt(_cellMaxAge[i]);
            }
            _simController->setSimulationParameters(parameters);
        }
        startNewMeasurement();
    }
}

void _BalancerController::startNewMeasurement()
{
    _lastTimestep = _simController->getCurrentTimestep();
    for (int i = 0; i < MAX_COLORS; ++i) {
        _numReplicators[i] = 0;
    }
}

void _BalancerController::saveLastState()
{
    auto const& parameters = _simController->getSimulationParameters();
    for (int i = 0; i < MAX_COLORS; ++i) {
        _lastCellMaxAge[i] = parameters.cellMaxAge[i];
    }
    _lastAdaptiveCellMaxAge = parameters.adaptiveCellMaxAge;
}
