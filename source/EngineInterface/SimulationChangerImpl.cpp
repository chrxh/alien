#include "SimulationChangerImpl.h"

#include <iostream>
#include <sstream>

#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"

#include "SimulationMonitor.h"

namespace
{
    auto const TimestepsForMonitor = 1000;

    auto const InitDuration = 400;
    auto const StepDuration = 100;
    auto const RetreatDuration = 400;

    auto const RetreatStartFactor = 0.35;
    auto const RetreatEndFactor = 0.6;
    auto const EmergencyRetreatStartFactor = 0.25;
    auto const MaxRetreats = 5;
}

SimulationChangerImpl::~SimulationChangerImpl()
{
    deactivate();
}

void SimulationChangerImpl::init(SimulationMonitor * monitor, NumberGenerator* numberGenerator)
{
    if (!_monitorConnections.empty()) {
        deactivate();
    }

    _numberGenerator = numberGenerator;
    _monitor = monitor;
    for (auto const& connection : _monitorConnections) {
        disconnect(connection);
    }
    _monitorConnections.emplace_back(
        connect(_monitor, &SimulationMonitor::dataReadyToRetrieve, this, &SimulationChangerImpl::monitorDataAvailable));
}

void SimulationChangerImpl::notifyNextTimestep()
{
    ++_timestepsSinceBeginning;

    if (0 == (_timestepsSinceBeginning % TimestepsForMonitor)) {
        _monitorDataRequired = true;
        _monitor->requireData();
    }

}

void SimulationChangerImpl::activate(SimulationParameters const & currentParameters)
{
    _state = State::Init;
    _parameters = currentParameters;
    _initialParameters = currentParameters;

    _numRetreats = 0;

    _timestepsSinceBeginning = 0;
    _measurementsSinceBeginning = 0;
    _measurementsOfCurrentEpoch = 0;
    _measurementsOfCurrentRetreat = 0;
}

void SimulationChangerImpl::deactivate()
{
    if (State::Deactivated == _state) {
        return;
    }

    _state = State::Deactivated;

    //restore initial parameters
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "parameter changer: restore initial parameters");

    _parameters = _initialParameters;
    Q_EMIT simulationParametersChanged();
}

SimulationParameters const & SimulationChangerImpl::retrieveSimulationParameters()
{
    return _parameters;
}

void SimulationChangerImpl::monitorDataAvailable()
{
    if (!_monitorDataRequired) {
        return;
    }
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();

    _monitorDataRequired = false;

    ++_measurementsSinceBeginning;

    auto data = _monitor->retrieveData();
    auto activeClusters = data.numClustersWithTokens;

    if (State::Init == _state) {
        if (InitDuration == _measurementsSinceBeginning) {
            _activeClustersReference = activeClusters;

            std::stringstream stream;
            stream << "parameter changer: measurement finished: " << *_activeClustersReference << " active clusters";
            loggingService->logMessage(Priority::Important, stream.str());

            _state = State::FindEpochTarget;
            loggingService->logMessage(Priority::Important, "parameter changer: find epoch target");
        }
    }
    else if (State::FindEpochTarget == _state) {
        _calculator = SimulationParametersCalculator::createWithRandomTarget(_parameters, _numberGenerator);
        _measurementsOfCurrentEpoch = _measurementsSinceBeginning;
        _numRetreats = 0;

        _state = State::Epoch;
        loggingService->logMessage(Priority::Important, "parameter changer: start epoch");
    }
    else if (State::Epoch == _state) {
        if (activeClusters < *_activeClustersReference * RetreatStartFactor) {
            std::stringstream stream;
            stream << "parameter changer: critical number of " << activeClusters << " active clusters reached ";
            loggingService->logMessage(Priority::Important, stream.str());

            ++_numRetreats;
            _state = State::Retreat;
            loggingService->logMessage(Priority::Important, "parameter changer: start retreat");

            while (!_calculator->isSourceReached()) {
                _calculator->getPrevious();
            }
            _parameters = _calculator->getSource();
            Q_EMIT simulationParametersChanged();

            _measurementsOfCurrentRetreat = _measurementsSinceBeginning;
        }
        else if (0 == ((_measurementsSinceBeginning - _measurementsOfCurrentEpoch) % StepDuration)) {
            _parameters = _calculator->getNext();
            Q_EMIT simulationParametersChanged();
            loggingService->logMessage(Priority::Important, "parameter changer: epoch step");

            if (_calculator->isTargetReached()) {
                _state = State::FindEpochTarget;
                loggingService->logMessage(Priority::Important, "parameter changer: end epoch");
            }
        }
    }
    else if (State::Retreat == _state) {
        if (activeClusters < *_activeClustersReference * EmergencyRetreatStartFactor) {

            std::stringstream stream;
            stream << "parameter changer: very critical number of " << activeClusters << " active clusters reached";
            loggingService->logMessage(Priority::Important, stream.str());

            _state = State::EmergencyRetreat;
            loggingService->logMessage(Priority::Important, "parameter changer: start emergency retreat");

            _parameters = _initialParameters;
            Q_EMIT simulationParametersChanged();
        }
        if (activeClusters > *_activeClustersReference * RetreatEndFactor
            || _measurementsOfCurrentRetreat + RetreatDuration < _measurementsSinceBeginning) {
            loggingService->logMessage(Priority::Important, "parameter changer: end retreat");

            if (_numRetreats == MaxRetreats) {
                _state = State::FindEpochTarget;
                loggingService->logMessage(Priority::Important, "parameter changer: find epoch target");
            }
            else {
                loggingService->logMessage(Priority::Important, "parameter changer: restart epoch");
                _state = State::Epoch;
            }
        }
    }
    else if (State::EmergencyRetreat == _state) {
        if (activeClusters > *_activeClustersReference * RetreatEndFactor) {
            loggingService->logMessage(Priority::Important, "parameter changer: end emergency retreat");

            _state = State::FindEpochTarget;
            loggingService->logMessage(Priority::Important, "parameter changer: find epoch target");
        }
    }
}
