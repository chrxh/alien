#pragma once

#include <chrono>
#include <optional>

#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/StatisticsRawData.h"

class _MaxAgeBalancer
{
public:
    //returns true if parameters have been changed
    bool balance(SimulationParameters& parameters, StatisticsRawData const& statistics, uint64_t timestep);

private:
    void initializeIfNecessary(SimulationParameters const& parameters, uint64_t timestep);
    bool doAdaptionIfNecessary(SimulationParameters& parameters, StatisticsRawData const& statistics, uint64_t timestep);
    void startNewMeasurement(uint64_t timestep);
    void saveLastState(SimulationParameters const& parameters);

    ColorVector<uint64_t> _numReplicators = { 0, 0, 0, 0, 0, 0, 0 };
    int _numMeasurements = 0;
    std::optional<uint64_t> _lastTimestep;
    ColorVector<double> _cellMaxAge = { 0, 0, 0, 0, 0, 0, 0 };    //cloned parameter with double precision

    bool _lastAdaptiveCellMaxAge = false;
    ColorVector<int> _lastCellMaxAge = { 0, 0, 0, 0, 0, 0, 0 };
};
