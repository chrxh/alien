#include "SimulationParametersParser.h"

namespace
{
    std::string toString(int value) { return QString("%1").arg(value).toStdString(); }
    std::string toString(float value) { return QString("%1").arg(value).toStdString(); }
    std::string toString(bool value) { return value ? "true" : "false"; }
}

boost::property_tree::ptree SimulationParametersParser::encode(SimulationParameters const& parameters)
{
    boost::property_tree::ptree tree;
    tree.add("cell.minDistance", toString(parameters.cellMinDistance));
    tree.add("cell.maxDistance", toString(parameters.cellMaxDistance));
    tree.add("cell.maxForce", toString(parameters.cellMaxForce));
    tree.add("cell.maxForceDecayProbability", toString(parameters.cellMaxForceDecayProb));
    tree.add("cell.minTokenUsages", toString(parameters.cellMinTokenUsages));
    tree.add("cell.tokenUsageDecayProbability", toString(parameters.cellTokenUsageDecayProb));
    tree.add("cell.maxBonds", toString(parameters.cellMaxBonds));
    tree.add("cell.maxToken", toString(parameters.cellMaxToken));
    tree.add("cell.maxTokenBranchNumber", toString(parameters.cellMaxTokenBranchNumber));
    tree.add("cell.minEnergy", toString(parameters.cellMinEnergy));
    tree.add("cell.transformationProb", toString(parameters.cellTransformationProb));
    tree.add("cell.fusionVelocity", toString(parameters.cellFusionVelocity));
    tree.add("cell.function.computer.maxInstructions", toString(parameters.cellFunctionComputerMaxInstructions));
    tree.add("cell.function.computer.memorySize", toString(parameters.cellFunctionComputerCellMemorySize));
    tree.add("cell.function.weapon.strength", toString(parameters.cellFunctionWeaponStrength));
    tree.add("cell.function.weapon.energyCost", toString(parameters.cellFunctionWeaponEnergyCost));
    tree.add(
        "cell.function.constructor.offspringCellEnergy",
        toString(parameters.cellFunctionConstructorOffspringCellEnergy));
    tree.add(
        "cell.function.constructor.offspringCellDistance",
        toString(parameters.cellFunctionConstructorOffspringCellDistance));
    tree.add(
        "cell.function.constructor.offspringTokenEnergy",
        toString(parameters.cellFunctionConstructorOffspringTokenEnergy));
    tree.add(
        "cell.function.constructor.offspringTokenSuppressMemoryCopy",
        toString(parameters.cellFunctionConstructorOffspringTokenSuppressMemoryCopy));
    tree.add(
        "cell.function.constructor.tokenDataMutationProbability",
        toString(parameters.cellFunctionConstructorTokenDataMutationProb));
    tree.add(
        "cell.function.constructor.cellDataMutationProbability",
        toString(parameters.cellFunctionConstructorCellDataMutationProb));
    tree.add(
        "cell.function.constructor.cellPropertyMutationProbability",
        toString(parameters.cellFunctionConstructorCellPropertyMutationProb));
    tree.add(
        "cell.function.constructor.cellStructureMutationProbability",
        toString(parameters.cellFunctionConstructorCellStructureMutationProb));
    tree.add("cell.function.sensor.range", toString(parameters.cellFunctionSensorRange));
    tree.add("cell.function.communicator.range", toString(parameters.cellFunctionCommunicatorRange));
    tree.add("token.memorySize", toString(parameters.tokenMemorySize));
    tree.add("token.minEnergy", toString(parameters.tokenMinEnergy));
    tree.add("radiation.exponent", toString(parameters.radiationExponent));
    tree.add("radiation.factor", toString(parameters.radiationFactor));
    tree.add("radiation.probability", toString(parameters.radiationProb));
    tree.add("radiation.velocityMultiplier", toString(parameters.radiationVelocityMultiplier));
    tree.add("radiation.velocityPerturbation", toString(parameters.radiationVelocityPerturbation));

    return tree;
}

SimulationParameters SimulationParametersParser::decode(boost::property_tree::ptree const& tree)
{
    SimulationParameters result;
    result.cellMinDistance = tree.get<float>("cell.minDistance");
    result.cellMaxDistance = tree.get<float>("cell.maxDistance");
    result.cellMaxForce = tree.get<float>("cell.maxForce");
    result.cellMaxForceDecayProb = tree.get<float>("cell.maxForceDecayProbability");
    result.cellMinTokenUsages = tree.get<int>("cell.minTokenUsages");
    result.cellTokenUsageDecayProb = tree.get<float>("cell.tokenUsageDecayProbability");
    result.cellMaxBonds = tree.get<int>("cell.maxBonds");
    result.cellMaxToken = tree.get<int>("cell.maxToken");
    result.cellMaxTokenBranchNumber = tree.get<int>("cell.maxTokenBranchNumber");
    result.cellMinEnergy = tree.get<float>("cell.minEnergy");
    result.cellTransformationProb = tree.get<float>("cell.transformationProb");
    result.cellFusionVelocity = tree.get<float>("cell.fusionVelocity");
    result.cellFunctionComputerMaxInstructions = tree.get<int>("cell.function.computer.maxInstructions");
    result.cellFunctionComputerCellMemorySize = tree.get<int>("cell.function.computer.memorySize");
    result.cellFunctionWeaponStrength = tree.get<float>("cell.function.weapon.strength");
    result.cellFunctionWeaponEnergyCost = tree.get<float>("cell.function.weapon.energyCost");
    result.cellFunctionConstructorOffspringCellEnergy =
        tree.get<float>("cell.function.constructor.offspringCellEnergy");
    result.cellFunctionConstructorOffspringCellDistance =
        tree.get<float>(
        "cell.function.constructor.offspringCellDistance");
    result.cellFunctionConstructorOffspringTokenEnergy =
        tree.get<float>("cell.function.constructor.offspringTokenEnergy");
    result.cellFunctionConstructorOffspringTokenSuppressMemoryCopy=
        tree.get<bool>("cell.function.constructor.offspringTokenSuppressMemoryCopy", false);
    result.cellFunctionConstructorTokenDataMutationProb =
        tree.get<float>(
        "cell.function.constructor.tokenDataMutationProbability");
    result.cellFunctionConstructorCellDataMutationProb =
        tree.get<float>(
        "cell.function.constructor.cellDataMutationProbability");
    result.cellFunctionConstructorCellPropertyMutationProb =
        tree.get<float>(
        "cell.function.constructor.cellPropertyMutationProbability");
    result.cellFunctionConstructorCellStructureMutationProb =
        tree.get<float>(
        "cell.function.constructor.cellStructureMutationProbability");
    result.cellFunctionSensorRange = tree.get<float>("cell.function.sensor.range");
    result.cellFunctionCommunicatorRange = tree.get<float>("cell.function.communicator.range");
    result.tokenMemorySize = tree.get<int>("token.memorySize");
    result.tokenMinEnergy = tree.get<float>("token.minEnergy");
    result.radiationExponent = tree.get<float>("radiation.exponent");
    result.radiationFactor = tree.get<float>("radiation.factor");
    result.radiationProb = tree.get<float>("radiation.probability");
    result.radiationVelocityMultiplier = tree.get<float>("radiation.velocityMultiplier");
    result.radiationVelocityPerturbation = tree.get<float>("radiation.velocityPerturbation");

    return result;
}
