#include "SimulationParametersParser.h"

boost::property_tree::ptree SimulationParametersParser::encode(SimulationParameters const& parameters)
{
    boost::property_tree::ptree tree;
    tree.add("cell.minDistance", parameters.cellMinDistance);
    tree.add("cell.maxDistance", parameters.cellMaxDistance);
    tree.add("cell.maxForce", parameters.cellMaxForce);
    tree.add("cell.maxForceDecayProbability", parameters.cellMaxForceDecayProb);
    tree.add("cell.minTokenUsages", parameters.cellMinTokenUsages);
    tree.add("cell.tokenUsageDecayProbability", parameters.cellTokenUsageDecayProb);
    tree.add("cell.maxBonds", parameters.cellMaxBonds);
    tree.add("cell.maxToken", parameters.cellMaxToken);
    tree.add("cell.maxTokenBranchNumber", parameters.cellMaxTokenBranchNumber);
    tree.add("cell.minEnergy", parameters.cellMinEnergy);
    tree.add("cell.transformationProb", parameters.cellTransformationProb);
    tree.add("cell.fusionVelocity", parameters.cellFusionVelocity);
    tree.add("cell.function.computer.maxInstructions", parameters.cellFunctionComputerMaxInstructions);
    tree.add("cell.function.computer.memorySize", parameters.cellFunctionComputerCellMemorySize);
    tree.add("cell.function.weapon.strength", parameters.cellFunctionWeaponStrength);
    tree.add("cell.function.weapon.energyCost", parameters.cellFunctionWeaponEnergyCost);
    tree.add("cell.function.constructor.offspringCellEnergy", parameters.cellFunctionConstructorOffspringCellEnergy);
    tree.add(
        "cell.function.constructor.offspringCellDistance", parameters.cellFunctionConstructorOffspringCellDistance);
    tree.add("cell.function.constructor.offspringTokenEnergy", parameters.cellFunctionConstructorOffspringTokenEnergy);
    tree.add("cell.function.constructor.offspringTokenSuppressMemoryCopy", parameters.cellFunctionConstructorOffspringTokenSuppressMemoryCopy);
    tree.add(
        "cell.function.constructor.tokenDataMutationProbability",
        parameters.cellFunctionConstructorTokenDataMutationProb);
    tree.add(
        "cell.function.constructor.cellDataMutationProbability",
        parameters.cellFunctionConstructorCellDataMutationProb);
    tree.add(
        "cell.function.constructor.cellPropertyMutationProbability",
        parameters.cellFunctionConstructorCellPropertyMutationProb);
    tree.add(
        "cell.function.constructor.cellStructureMutationProbability",
        parameters.cellFunctionConstructorCellStructureMutationProb);
    tree.add("cell.function.sensor.range", parameters.cellFunctionSensorRange);
    tree.add("cell.function.communicator.range", parameters.cellFunctionCommunicatorRange);
    tree.add("token.memorySize", parameters.tokenMemorySize);
    tree.add("token.minEnergy", parameters.tokenMinEnergy);
    tree.add("radiation.exponent", parameters.radiationExponent);
    tree.add("radiation.factor", parameters.radiationFactor);
    tree.add("radiation.probability", parameters.radiationProb);
    tree.add("radiation.velocityMultiplier", parameters.radiationVelocityMultiplier);
    tree.add("radiation.velocityPerturbation", parameters.radiationVelocityPerturbation);

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
