#pragma once

#include "SimulationParametersCalculator.h"
#include "SimulationChanger.h"
#include "Definitions.h"

class SimulationChangerImpl : public SimulationChanger
{
    Q_OBJECT
public:
    ~SimulationChangerImpl();

    void init(SimulationMonitor* monitor, NumberGenerator* numberGenerator);
    Q_SLOT void notifyNextTimestep() override;

    void activate(SimulationParameters const& currentParameters) override;
    void deactivate() override;

    SimulationParameters const& retrieveSimulationParameters() override;

private:
    Q_SLOT void monitorDataAvailable();

private:
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
    bool _monitorDataRequired = false;
    int _numRetreats = 0;

    int _timestepsSinceBeginning = 0;
    int _measurementsSinceBeginning = 0;
    int _measurementsOfCurrentEpoch = 0;
    int _measurementsOfCurrentRetreat = 0;

    SimulationParameters _initialParameters;
    SimulationParameters _parameters;
    SimulationMonitor* _monitor;
    NumberGenerator* _numberGenerator;
    boost::optional<SimulationParametersCalculator> _calculator;

    list<QMetaObject::Connection> _monitorConnections;

    boost::optional<int> _tokensReference;
};
