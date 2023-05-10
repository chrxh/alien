#pragma once

#include <chrono>

#include "EngineInterface/Colors.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"
#include "Definitions.h"

class _BalancerController
{
public:
    _BalancerController(SimulationController const& simController);

    bool isOn() const;
    void setOn(bool value);

    void process();

private:
    void initializeIfNecessary();
    void doAdaption();
    void startNewMeasurement();
    void saveLastState();

    SimulationController _simController;

    ColorVector<uint64_t> _numReplicators = {0, 0, 0, 0, 0, 0, 0};
    std::optional<uint64_t> _lastTimestep;
    ColorVector<double> _cellMaxAge = {0, 0, 0, 0, 0, 0, 0};    //cloned parameter with double precision

    bool _lastAdaptiveCellMaxAge = false;
    ColorVector<int> _lastCellMaxAge = {0, 0, 0, 0, 0, 0, 0};
};
