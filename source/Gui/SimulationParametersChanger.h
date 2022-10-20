#pragma once

#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SimulationController.h"

#include "Definitions.h"

class _SimulationParametersChanger
{
public:
    _SimulationParametersChanger(SimulationController const& simController, int timestepsPerEpoch);

    void process();

    void activate();
    void deactivate();

    int getTimestepsPerEpoch() const;
    int getOriginalTimestepsPerEpoch() const;
    void setTimestepsPerEpoch(int timestepsPerEpoch);

private:
    void onChangeParameters();

    SimulationController _simController;

    enum class State
    {
        Deactivated,
        Init,
        FindEpochTarget,
        Epoch,
        Retreat,
        EmergencyRetreat
    };
    State _state = State::Deactivated;
    int _origTimestepsPerEpoch = 10000;
    int _timestepsPerEpoch = 10000;

    std::optional<uint64_t> _lastTimestep = 0;
    uint64_t _accumulatedTimesteps = 0;

    int _numRetreats = 0;

    int _measurementsSinceBeginning = 0;
    int _measurementsOfCurrentEpoch = 0;
    int _measurementsOfCurrentRetreat = 0;

    SimulationParameters _initialParameters;
    SimulationParameters _parameters;
    SimulationParametersCalculator _calculator;

    std::optional<float> _activeClustersReference;
};
