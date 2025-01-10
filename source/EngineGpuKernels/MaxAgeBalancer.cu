#include "MaxAgeBalancer.cuh"

#include <vector>

#include "Base.cuh"

namespace
{
    auto constexpr AdaptionRatio = 1.3;
    auto constexpr AdaptionFactor = 1.1;
    auto constexpr MaxCellAge = 300000;
    auto constexpr MinReplicatorsUpperValue = 100;
    auto constexpr MinReplicatorsLowerValue = 20;
}

bool _MaxAgeBalancer::balance(SimulationParameters& parameters, StatisticsRawData const& statistics, uint64_t timestep)
{
    auto result = false;
    if (parameters.features.cellAgeLimiter && parameters.cellMaxAgeBalancer) {
        initializeIfNecessary(parameters, timestep);
        result |= doAdaptionIfNecessary(parameters, statistics, timestep);
    }

    saveLastState(parameters);

    return result;
}

void _MaxAgeBalancer::initializeIfNecessary(SimulationParameters const& parameters, uint64_t timestep)
{
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
        startNewMeasurement(timestep);
    }
}

bool _MaxAgeBalancer::doAdaptionIfNecessary(SimulationParameters& parameters, StatisticsRawData const& statistics, uint64_t timestep)
{
    auto result = false;
    for (int i = 0; i < MAX_COLORS; ++i) {
        _numReplicators[i] += statistics.timeline.timestep.numSelfReplicators[i];
    }
    ++_numMeasurements;
    if (timestep - *_lastTimestep > parameters.cellMaxAgeBalancerInterval) {
        uint64_t maxReplicators = 0;
        uint64_t averageReplicators = 0;
        int numAveragedReplicators = 0;
        std::vector<int> colors;
        for (int color = 0; color < MAX_COLORS; ++color) {
            if (maxReplicators < _numReplicators[color]) {
                maxReplicators = _numReplicators[color];
            }
            if (_numReplicators[color] / _numMeasurements > MinReplicatorsLowerValue) {
                averageReplicators += _numReplicators[color];
                ++numAveragedReplicators;
            }
        }
        if (numAveragedReplicators > 0) {
            averageReplicators /= numAveragedReplicators;
        }

        if (averageReplicators > 0) {
            for (int color = 0; color < MAX_COLORS; ++color) {
                if (toDouble(_numReplicators[color]) / _numMeasurements > MinReplicatorsUpperValue
                    && toDouble(_numReplicators[color]) / toDouble(averageReplicators) > AdaptionRatio) {
                    _cellMaxAge[color] /= AdaptionFactor;
                } else if (
                    _cellMaxAge[color] < MaxCellAge
                    && (_numReplicators[color] / _numMeasurements <= MinReplicatorsLowerValue
                        || toDouble(averageReplicators) / toDouble(_numReplicators[color]) > AdaptionRatio)) {
                    _cellMaxAge[color] *= AdaptionFactor;
                }
            }

            for (int i = 0; i < MAX_COLORS; ++i) {
                parameters.cellMaxAge[i] = toInt(_cellMaxAge[i]);
            }
            result = true;
        }
        startNewMeasurement(timestep);
    }
    return result;
}

void _MaxAgeBalancer::startNewMeasurement(uint64_t timestep)
{
    _lastTimestep = timestep;
    for (int i = 0; i < MAX_COLORS; ++i) {
        _numReplicators[i] = 0;
    }
    _numMeasurements = 0;
}

void _MaxAgeBalancer::saveLastState(SimulationParameters const& parameters)
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        _lastCellMaxAge[i] = parameters.cellMaxAge[i];
    }
    _lastAdaptiveCellMaxAge = parameters.cellMaxAgeBalancer;
}
