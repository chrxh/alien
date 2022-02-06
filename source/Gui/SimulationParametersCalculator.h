#pragma once

#include "EngineInterface/SimulationParameters.h"
#include "Definitions.h"

class _SimulationParametersCalculator
{
public:
    static SimulationParametersCalculator create(SimulationParameters const& source, SimulationParameters const& target);
    static SimulationParametersCalculator createWithRandomTarget(SimulationParameters const& source);

    bool isTargetReached() const;
    bool isSourceReached() const;

    SimulationParameters getNext();
    SimulationParameters getPrevious();

    SimulationParameters const& getSource() const;

protected:
    _SimulationParametersCalculator(SimulationParameters const& source, SimulationParameters const& target);

    SimulationParameters calcCurrentParameters() const;
    float calcCurrentParameter(float source, float target) const;
    int calcCurrentParameter(int source, int target) const;

private:
    SimulationParameters _source;
    SimulationParameters _target;
    int _step = 0;
};
