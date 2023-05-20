#include "BalancerController.h"

#include <cmath>

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/StatisticsData.h"

namespace
{
    auto constexpr AdaptionRatio = 1.3f;
    auto constexpr AdaptionFactor = 0.9;
    auto constexpr MaxCellAge = 300000;
    auto constexpr MinReplicatorsUpperValue = 100;
    auto constexpr MinReplicatorsLowerValue = 10;
}

_BalancerController::_BalancerController(SimulationController const& simController)
    : _simController(simController)
{}

void _BalancerController::process()
{
    auto const& parameters = _simController->getSimulationParameters();
    if (parameters.cellMaxAgeBalancer) {
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
    if (parameters.cellMaxAgeBalancer != _lastAdaptiveCellMaxAge) {
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
    if (_simController->getCurrentTimestep() - *_lastTimestep > parameters.cellMaxAgeBalancerInterval) {
        uint64_t maxReplicators = 0;
        uint64_t averageReplicators = 0;
        int numColors = 0;
        uint64_t minReplicators = 0;
        std::vector<int> colors;
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (maxReplicators < _numReplicators[i]) {
                maxReplicators = _numReplicators[i];
            }
            if (_numReplicators[i] > MinReplicatorsLowerValue) {
                if ((minReplicators == 0 || minReplicators > _numReplicators[i])) {
                    minReplicators = _numReplicators[i];
                }
                averageReplicators += _numReplicators[i];
                ++numColors;
                colors.emplace_back(i);
            }
        }
        if (numColors > 0) {
            averageReplicators /= numColors;
        }

        if (averageReplicators > 0) {
            for (auto const& color : colors) {
                if (_numReplicators[color] > MinReplicatorsUpperValue && _numReplicators[color] / averageReplicators > AdaptionRatio) {
                    _cellMaxAge[color] *= AdaptionFactor;
                } else if (_cellMaxAge[color] < MaxCellAge && _numReplicators[color] / averageReplicators < AdaptionRatio) {
                    _cellMaxAge[color] /= AdaptionFactor;
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
    _lastAdaptiveCellMaxAge = parameters.cellMaxAgeBalancer;
}
