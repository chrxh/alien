#include "SimulationParametersCalculator.h"

#include "Base/NumberGenerator.h"

namespace
{
    auto const MaxSteps = 10;
}

SimulationParametersCalculator _SimulationParametersCalculator::create(SimulationParameters const& source, SimulationParameters const& target)
{
    return std::shared_ptr<_SimulationParametersCalculator>(new _SimulationParametersCalculator(source, target));
}

SimulationParametersCalculator _SimulationParametersCalculator::createWithRandomTarget(SimulationParameters const& source)
{
    auto target = source;
    auto& numberGenerator = NumberGenerator::getInstance();

    target.baseValues.friction = numberGenerator.getRandomFloat(0.0f, 4.0f) / 1000;
    target.baseValues.rigidity =numberGenerator.getRandomFloat(0.0f, 1.0f);
    target.baseValues.radiationFactor = numberGenerator.getRandomFloat(0.0f, 0.7f) / 1000;
    target.baseValues.cellMaxForce = numberGenerator.getRandomFloat(0.1f, 1.0f);
    target.baseValues.cellMinEnergy = numberGenerator.getRandomFloat(25.0f, 80.0f);
    target.baseValues.cellFusionVelocity = numberGenerator.getRandomFloat(0.0f, 1.0f);
    target.baseValues.cellFunctionAttackerEnergyCost = numberGenerator.getRandomFloat(0.0f, 3.0f);
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            target.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j] = numberGenerator.getRandomFloat(0.0f, 1.0f);
        }
    }
    target.baseValues.cellFunctionAttackerGeometryDeviationExponent = numberGenerator.getRandomFloat(0.0f, 4.0f);

    return std::shared_ptr<_SimulationParametersCalculator>(new _SimulationParametersCalculator(source, target));
}

bool _SimulationParametersCalculator::isTargetReached() const
{
    return MaxSteps == _step;
}

bool _SimulationParametersCalculator::isSourceReached() const
{
    return 0 == _step;
}

SimulationParameters _SimulationParametersCalculator::getNext()
{
    _step = std::min(MaxSteps, _step + 1);
    return calcCurrentParameters();
}

SimulationParameters _SimulationParametersCalculator::getPrevious()
{
    _step = std::max(0, _step - 1);
    return calcCurrentParameters();
}

SimulationParameters const& _SimulationParametersCalculator::getSource() const
{
    return _source;
}

_SimulationParametersCalculator::_SimulationParametersCalculator(SimulationParameters const& source, SimulationParameters const& target)
    : _source(source)
    , _target(target)
{}

SimulationParameters _SimulationParametersCalculator::calcCurrentParameters() const
{
    auto result = _source;
    result.baseValues.friction = calcCurrentParameter(_source.baseValues.friction, _target.baseValues.friction);
    result.baseValues.rigidity = calcCurrentParameter(_source.baseValues.rigidity, _target.baseValues.rigidity);
    result.baseValues.radiationFactor = calcCurrentParameter(_source.baseValues.radiationFactor, _target.baseValues.radiationFactor);
    result.baseValues.cellMaxForce = calcCurrentParameter(_source.baseValues.cellMaxForce, _target.baseValues.cellMaxForce);
    result.baseValues.cellMinEnergy = calcCurrentParameter(_source.baseValues.cellMinEnergy, _target.baseValues.cellMinEnergy);
    result.baseValues.cellFusionVelocity = calcCurrentParameter(_source.baseValues.cellFusionVelocity, _target.baseValues.cellFusionVelocity);
    result.baseValues.cellFunctionAttackerEnergyCost =
        calcCurrentParameter(_source.baseValues.cellFunctionAttackerEnergyCost, _target.baseValues.cellFunctionAttackerEnergyCost);

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            result.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                calcCurrentParameter(_source.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j], _target.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j]);
        }
    }
    result.baseValues.cellFunctionAttackerGeometryDeviationExponent =
        calcCurrentParameter(_source.baseValues.cellFunctionAttackerGeometryDeviationExponent, _target.baseValues.cellFunctionAttackerGeometryDeviationExponent);
    return result;
}

float _SimulationParametersCalculator::calcCurrentParameter(float source, float target) const
{
    auto const factor = toFloat(_step) / MaxSteps;
    return source * (1.0f - factor) + target * factor;
}

int _SimulationParametersCalculator::calcCurrentParameter(int source, int target) const
{
    return toInt(calcCurrentParameter(toFloat(source), toFloat(target)));
}
