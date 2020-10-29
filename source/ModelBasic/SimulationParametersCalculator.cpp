#include "Base/NumberGenerator.h"

#include "SimulationParametersCalculator.h"

namespace
{
    auto const MaxSteps = 10;
}

SimulationParametersCalculator SimulationParametersCalculator::create(
    SimulationParameters const& source,
    SimulationParameters const& target)
{
    return SimulationParametersCalculator(source, target);
}

SimulationParametersCalculator SimulationParametersCalculator::createWithRandomTarget(
    SimulationParameters const& source, NumberGenerator* numberGenerator)
{
    auto target = source;
    target.cellMaxForce = numberGenerator->getRandomReal(0.1, 1.0);
    target.cellMinTokenUsages = numberGenerator->getRandomInt(40000, 400000);
    target.cellTokenUsageDecayProb = numberGenerator->getRandomReal(0.01f, 0.1f) / 10000.0;
    target.cellMinEnergy = numberGenerator->getRandomReal(25.0, 80);
    target.cellFusionVelocity = numberGenerator->getRandomReal(0.2, 0.8);
    target.cellFunctionWeaponStrength = numberGenerator->getRandomReal(0.2, 2.7);
    target.cellFunctionWeaponEnergyCost = numberGenerator->getRandomReal(0, 4);
    target.cellFunctionConstructorOffspringCellEnergy = numberGenerator->getRandomReal(70, 130);
    target.cellFunctionConstructorOffspringTokenEnergy = numberGenerator->getRandomReal(30, 100);
    target.cellFunctionConstructorTokenDataMutationProb = numberGenerator->getRandomReal(0, 0.02);
    target.cellFunctionConstructorCellDataMutationProb = numberGenerator->getRandomReal(0, 0.02);
    target.cellFunctionConstructorCellPropertyMutationProb = numberGenerator->getRandomReal(0, 0.02);
    target.cellFunctionConstructorCellStructureMutationProb = numberGenerator->getRandomReal(0, 0.02);
    target.radiationFactor = numberGenerator->getRandomReal(0.05, 0.7) / 1000.0;

    return SimulationParametersCalculator(source, target);
}

bool SimulationParametersCalculator::isTargetReached() const
{
    return MaxSteps == _step;
}

bool SimulationParametersCalculator::isSourceReached() const
{
    return 0 == _step;
}

SimulationParameters SimulationParametersCalculator::getNext()
{
    _step = std::min(MaxSteps, _step + 1);
    return calcCurrentParameters();
}

SimulationParameters SimulationParametersCalculator::getPrevious()
{
    _step = std::max(0, _step - 1);
    return calcCurrentParameters();
}

SimulationParameters const & SimulationParametersCalculator::getSource() const
{
    return _source;
}

SimulationParametersCalculator::SimulationParametersCalculator(
    SimulationParameters const& source,
    SimulationParameters const& target)
    : _source(source)
    , _target(target)
{
}

SimulationParameters SimulationParametersCalculator::calcCurrentParameters() const
{
    auto result = _source;
    result.cellMaxForce = calcCurrentParameter(_source.cellMaxForce, _target.cellMaxForce);
    result.cellMinTokenUsages = calcCurrentParameter(_source.cellMinTokenUsages, _target.cellMinTokenUsages);  //int
    result.cellTokenUsageDecayProb =
        calcCurrentParameter(_source.cellTokenUsageDecayProb, _target.cellTokenUsageDecayProb);
    result.cellMinEnergy = calcCurrentParameter(_source.cellMinEnergy, _target.cellMinEnergy);
    result.cellFusionVelocity = calcCurrentParameter(_source.cellFusionVelocity, _target.cellFusionVelocity);
    result.cellFunctionWeaponStrength =
        calcCurrentParameter(_source.cellFunctionWeaponStrength, _target.cellFunctionWeaponStrength);
    result.cellFunctionWeaponEnergyCost =
        calcCurrentParameter(_source.cellFunctionWeaponEnergyCost, _target.cellFunctionWeaponEnergyCost);
    result.cellFunctionConstructorOffspringCellEnergy = calcCurrentParameter(
        _source.cellFunctionConstructorOffspringCellEnergy, _target.cellFunctionConstructorOffspringCellEnergy);
    result.cellFunctionConstructorOffspringTokenEnergy = calcCurrentParameter(
        _source.cellFunctionConstructorOffspringTokenEnergy, _target.cellFunctionConstructorOffspringTokenEnergy);
    result.cellFunctionConstructorTokenDataMutationProb = calcCurrentParameter(
        _source.cellFunctionConstructorTokenDataMutationProb, _target.cellFunctionConstructorTokenDataMutationProb);
    result.cellFunctionConstructorCellDataMutationProb = calcCurrentParameter(
        _source.cellFunctionConstructorCellDataMutationProb, _target.cellFunctionConstructorCellDataMutationProb);
    result.cellFunctionConstructorCellPropertyMutationProb = calcCurrentParameter(
        _source.cellFunctionConstructorCellPropertyMutationProb,
        _target.cellFunctionConstructorCellPropertyMutationProb);
    result.cellFunctionConstructorCellStructureMutationProb = calcCurrentParameter(
        _source.cellFunctionConstructorCellStructureMutationProb,
        _target.cellFunctionConstructorCellStructureMutationProb);
    result.radiationFactor = calcCurrentParameter(_source.radiationFactor, _target.radiationFactor);
    return result;
}

float SimulationParametersCalculator::calcCurrentParameter(float source, float target) const
{
    auto const factor = static_cast<float>(_step) / static_cast<float>(MaxSteps);
    return source * (1.0f - factor) + target * factor;
}

int SimulationParametersCalculator::calcCurrentParameter(int source, int target) const
{
    return static_cast<int>(calcCurrentParameter(static_cast<float>(source), static_cast<float>(target)));
}
