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

    target.spotValues.friction = numberGenerator.getRandomFloat(0.0f, 4.0f) / 1000;
    target.spotValues.rigidity =numberGenerator.getRandomFloat(0.0f, 1.0f);
    target.spotValues.radiationFactor = numberGenerator.getRandomFloat(0.0f, 0.7f) / 1000;
    target.spotValues.cellMaxForce = numberGenerator.getRandomFloat(0.1f, 1.0f);
    target.spotValues.cellMinEnergy = numberGenerator.getRandomFloat(25.0f, 80.0f);
    target.spotValues.cellBindingForce = numberGenerator.getRandomFloat(0.3f, 2.0f);
    target.spotValues.cellFusionVelocity = numberGenerator.getRandomFloat(0.0f, 1.0f);
    target.spotValues.cellFunctionWeaponEnergyCost = numberGenerator.getRandomFloat(0.0f, 3.0f);
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            target.spotValues.cellFunctionWeaponFoodChainColorMatrix[i][j] = numberGenerator.getRandomFloat(0.0f, 1.0f);
        }
    }
    target.spotValues.cellFunctionWeaponGeometryDeviationExponent = numberGenerator.getRandomFloat(0.0f, 4.0f);

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
    result.spotValues.friction = calcCurrentParameter(_source.spotValues.friction, _target.spotValues.friction);
    result.spotValues.rigidity = calcCurrentParameter(_source.spotValues.rigidity, _target.spotValues.rigidity);
    result.spotValues.radiationFactor = calcCurrentParameter(_source.spotValues.radiationFactor, _target.spotValues.radiationFactor);
    result.spotValues.cellMaxForce = calcCurrentParameter(_source.spotValues.cellMaxForce, _target.spotValues.cellMaxForce);
    result.spotValues.cellMinEnergy = calcCurrentParameter(_source.spotValues.cellMinEnergy, _target.spotValues.cellMinEnergy);
    result.spotValues.cellBindingForce = calcCurrentParameter(_source.spotValues.cellBindingForce, _target.spotValues.cellBindingForce);
    result.spotValues.cellFusionVelocity = calcCurrentParameter(_source.spotValues.cellFusionVelocity, _target.spotValues.cellFusionVelocity);
    result.spotValues.cellFunctionWeaponEnergyCost =
        calcCurrentParameter(_source.spotValues.cellFunctionWeaponEnergyCost, _target.spotValues.cellFunctionWeaponEnergyCost);

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            result.spotValues.cellFunctionWeaponFoodChainColorMatrix[i][j] =
                calcCurrentParameter(_source.spotValues.cellFunctionWeaponFoodChainColorMatrix[i][j], _target.spotValues.cellFunctionWeaponFoodChainColorMatrix[i][j]);
        }
    }
    result.spotValues.cellFunctionWeaponGeometryDeviationExponent =
        calcCurrentParameter(_source.spotValues.cellFunctionWeaponGeometryDeviationExponent, _target.spotValues.cellFunctionWeaponGeometryDeviationExponent);
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
