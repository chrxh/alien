#include "SimulationParametersChanger.h"

#include <iostream>
#include <sstream>

#include "Base/LoggingService.h"
#include "EngineInterface/OverallStatistics.h"
#include "SimulationParametersCalculator.h"
#include "GlobalSettings.h"

namespace
{
    auto const InitDuration = 4;
    auto const RetreatDuration = 4;

    auto const RetreatStartFactor = 0.35;
    auto const RetreatEndFactor = 0.6;
    auto const EmergencyRetreatStartFactor = 0.25;
    auto const MaxRetreats = 5;
}

_SimulationParametersChanger::_SimulationParametersChanger(SimulationController const& simController, int timestepsPerEpoch)
    : _simController(simController)
{
    _timestepsPerEpoch = timestepsPerEpoch;
    _origTimestepsPerEpoch = timestepsPerEpoch;
}

void _SimulationParametersChanger::process()
{
    if (_state == State::Deactivated) {
        return;
    }
    auto timestep = _simController->getCurrentTimestep();
    if (_lastTimestep) {
        if (*_lastTimestep <= timestep) {
            _accumulatedTimesteps += timestep - *_lastTimestep;

            if (_accumulatedTimesteps > _timestepsPerEpoch) {
                onChangeParameters();
                _accumulatedTimesteps -= _timestepsPerEpoch;
            }
        } else {
            _lastTimestep = std::nullopt;
        }
    }

    _lastTimestep = timestep;
}

void _SimulationParametersChanger::activate()
{
    _state = State::Init;
    _parameters = _simController->getSimulationParameters();
    _initialParameters = _parameters;

    _numRetreats = 0;

    _lastTimestep = std::nullopt;
    _accumulatedTimesteps = 0;

    _measurementsSinceBeginning = 0;
    _measurementsOfCurrentEpoch = 0;
    _measurementsOfCurrentRetreat = 0;
    log(Priority::Important, "parameter changer: activated, start measurement");
}

void _SimulationParametersChanger::deactivate()
{
    _state = State::Deactivated;
    log(Priority::Important, "parameter changer: deactivated");
}

int _SimulationParametersChanger::getTimestepsPerEpoch() const
{
    return _timestepsPerEpoch;
}

int _SimulationParametersChanger::getOriginalTimestepsPerEpoch() const
{
    return _origTimestepsPerEpoch;
}

void _SimulationParametersChanger::setTimestepsPerEpoch(int timestepsPerEpoch)
{
    _timestepsPerEpoch = timestepsPerEpoch;
    _accumulatedTimesteps = 0;
}

void _SimulationParametersChanger::onChangeParameters()
{
    ++_measurementsSinceBeginning;

    auto statistics = _simController->getStatistics();
    auto activeClusters = statistics.numTokens;

    if (State::Init == _state) {
        if (InitDuration == _measurementsSinceBeginning) {
            _activeClustersReference = activeClusters;

            std::stringstream stream;
            stream << "parameter changer: measurement finished: " << *_activeClustersReference << " active clusters";
            log(Priority::Important, stream.str());

            _state = State::FindEpochTarget;
            log(Priority::Important, "parameter changer: find epoch target");
        }
    } else if (State::FindEpochTarget == _state) {
        _calculator = _SimulationParametersCalculator::createWithRandomTarget(_parameters);
        _measurementsOfCurrentEpoch = _measurementsSinceBeginning;
        _numRetreats = 0;

        _state = State::Epoch;
        log(Priority::Important, "parameter changer: start epoch");
    } else if (State::Epoch == _state) {
        if (activeClusters < *_activeClustersReference * RetreatStartFactor) {
            std::stringstream stream;
            stream << "parameter changer: critical number of " << activeClusters << " active clusters reached ";
            log(Priority::Important, stream.str());

            ++_numRetreats;
            _state = State::Retreat;
            log(Priority::Important, "parameter changer: start retreat");

            while (!_calculator->isSourceReached()) {
                _calculator->getPrevious();
            }
            _parameters = _calculator->getSource();
            _simController->setSimulationParameters_async(_parameters);

            _measurementsOfCurrentRetreat = _measurementsSinceBeginning;
        } else {
            _parameters = _calculator->getNext();
            _simController->setSimulationParameters_async(_parameters);
            log(Priority::Important, "parameter changer: epoch step");

            if (_calculator->isTargetReached()) {
                _state = State::FindEpochTarget;
                log(Priority::Important, "parameter changer: end epoch");
            }
        }
    } else if (State::Retreat == _state) {
        if (activeClusters < *_activeClustersReference * EmergencyRetreatStartFactor) {

            std::stringstream stream;
            stream << "parameter changer: very critical number of " << activeClusters << " active clusters reached";
            log(Priority::Important, stream.str());

            _state = State::EmergencyRetreat;
            log(Priority::Important, "parameter changer: start emergency retreat");

            _parameters = _initialParameters;
            _simController->setSimulationParameters_async(_parameters);
        }
        if (activeClusters > *_activeClustersReference * RetreatEndFactor || _measurementsOfCurrentRetreat + RetreatDuration < _measurementsSinceBeginning) {
            log(Priority::Important, "parameter changer: end retreat");

            if (_numRetreats == MaxRetreats) {
                _state = State::FindEpochTarget;
                log(Priority::Important, "parameter changer: find epoch target");
            } else {
                log(Priority::Important, "parameter changer: restart epoch");
                _state = State::Epoch;
            }
        }
    } else if (State::EmergencyRetreat == _state) {
        if (activeClusters > *_activeClustersReference * RetreatEndFactor) {
            log(Priority::Important, "parameter changer: end emergency retreat");

            _state = State::FindEpochTarget;
            log(Priority::Important, "parameter changer: find epoch target");
        }
    }
}
