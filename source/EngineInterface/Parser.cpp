#include "Parser.h"

#include "EngineInterfaceSettings.h"
#include "GeneralSettings.h"

namespace
{
    std::string toString(int value) { return std::to_string(value); }
    std::string toString(float value) { return std::to_string(value); }
    std::string toString(bool value) { return value ? "true" : "false"; }
}

boost::property_tree::ptree Parser::encode(SimulationParameters const& parameters)
{
    boost::property_tree::ptree tree;
    tree.add("cell.min distance", toString(parameters.cellMinDistance));
    tree.add("cell.max distance", toString(parameters.cellMaxCollisionDistance));
    tree.add("cell.max force", toString(parameters.cellMaxForce));
    tree.add("cell.max force decay probability", toString(parameters.cellMaxForceDecayProb));
    tree.add("cell.min token usages", toString(parameters.cellMinTokenUsages));
    tree.add("cell.token usage decay probability", toString(parameters.cellTokenUsageDecayProb));
    tree.add("cell.max bonds", toString(parameters.cellMaxBonds));
    tree.add("cell.max token", toString(parameters.cellMaxToken));
    tree.add("cell.max token branch number", toString(parameters.cellMaxTokenBranchNumber));
    tree.add("cell.min energy", toString(parameters.cellMinEnergy));
    tree.add("cell.transformation probability", toString(parameters.cellTransformationProb));
    tree.add("cell.fusion velocity", toString(parameters.cellFusionVelocity));
    tree.add("cell.function.computer.max instructions", toString(parameters.cellFunctionComputerMaxInstructions));
    tree.add("cell.function.computer.memory size", toString(parameters.cellFunctionComputerCellMemorySize));
    tree.add("cell.function.weapon.strength", toString(parameters.cellFunctionWeaponStrength));
    tree.add("cell.function.weapon.energy cost", toString(parameters.cellFunctionWeaponEnergyCost));
    tree.add("cell.function.weapon.geometry deviation exponent", toString(parameters.cellFunctionWeaponGeometryDeviationExponent));
    tree.add(
        "cell.function.weapon.inhomogeneous color factor",
        toString(parameters.cellFunctionWeaponInhomogeneousColorFactor));
    tree.add(
        "cell.function.constructor.offspring.cell energy",
        toString(parameters.cellFunctionConstructorOffspringCellEnergy));
    tree.add(
        "cell.function.constructor.offspring.cell distance",
        toString(parameters.cellFunctionConstructorOffspringCellDistance));
    tree.add(
        "cell.function.constructor.offspring.token energy",
        toString(parameters.cellFunctionConstructorOffspringTokenEnergy));
    tree.add(
        "cell.function.constructor.offspring.token suppress memory copy",
        toString(parameters.cellFunctionConstructorOffspringTokenSuppressMemoryCopy));
    tree.add(
        "cell.function.constructor.mutation probability.token data",
        toString(parameters.cellFunctionConstructorTokenDataMutationProb));
    tree.add(
        "cell.function.constructor.mutation probability.cell data",
        toString(parameters.cellFunctionConstructorCellDataMutationProb));
    tree.add(
        "cell.function.constructor.mutation probability.cell property",
        toString(parameters.cellFunctionConstructorCellPropertyMutationProb));
    tree.add(
        "cell.function.constructor.mutation probability.cell structure",
        toString(parameters.cellFunctionConstructorCellStructureMutationProb));
    tree.add("cell.function.sensor.range", toString(parameters.cellFunctionSensorRange));
    tree.add("cell.function.communicator.range", toString(parameters.cellFunctionCommunicatorRange));
    tree.add("token.memory size", toString(parameters.tokenMemorySize));
    tree.add("token.min energy", toString(parameters.tokenMinEnergy));
    tree.add("radiation.exponent", toString(parameters.radiationExponent));
    tree.add("radiation.factor", toString(parameters.radiationFactor));
    tree.add("radiation.probability", toString(parameters.radiationProb));
    tree.add("radiation.velocity multiplier", toString(parameters.radiationVelocityMultiplier));
    tree.add("radiation.velocity perturbation", toString(parameters.radiationVelocityPerturbation));

    return tree;
}

SimulationParameters Parser::decodeSimulationParameters(boost::property_tree::ptree const& tree)
{
    auto defaultParameters = SimulationParameters();

    SimulationParameters result;
    result.cellMinDistance = tree.get<float>("cell.min distance");
    result.cellMaxCollisionDistance = tree.get<float>("cell.max distance");
    result.cellMaxForce = tree.get<float>("cell.max force");
    result.cellMaxForceDecayProb = tree.get<float>("cell.max force decay probability");
    result.cellMinTokenUsages = tree.get<int>("cell.min token usages");
    result.cellTokenUsageDecayProb = tree.get<float>("cell.token usage decay probability");
    result.cellMaxBonds = tree.get<int>("cell.max bonds");
    result.cellMaxToken = tree.get<int>("cell.max token");
    result.cellMaxTokenBranchNumber = tree.get<int>("cell.max token branch number");
    result.cellMinEnergy = tree.get<float>("cell.min energy");
    result.cellTransformationProb = tree.get<float>("cell.transformation probability");
    result.cellFusionVelocity = tree.get<float>("cell.fusion velocity");
    result.cellFunctionComputerMaxInstructions = tree.get<int>("cell.function.computer.max instructions");
    result.cellFunctionComputerCellMemorySize = tree.get<int>("cell.function.computer.memory size");
    result.cellFunctionWeaponStrength = tree.get<float>("cell.function.weapon.strength");
    result.cellFunctionWeaponEnergyCost = tree.get<float>("cell.function.weapon.energy cost");
    result.cellFunctionWeaponGeometryDeviationExponent = tree.get<float>(
        "cell.function.weapon.geometry deviation exponent",
        defaultParameters.cellFunctionWeaponGeometryDeviationExponent);
    result.cellFunctionWeaponInhomogeneousColorFactor = tree.get<float>(
        "cell.function.weapon.inhomogeneous color factor",
        defaultParameters.cellFunctionWeaponInhomogeneousColorFactor);
    result.cellFunctionConstructorOffspringCellEnergy =
        tree.get<float>("cell.function.constructor.offspring.cell energy");
    result.cellFunctionConstructorOffspringCellDistance =
        tree.get<float>(
        "cell.function.constructor.offspring.cell distance");
    result.cellFunctionConstructorOffspringTokenEnergy =
        tree.get<float>("cell.function.constructor.offspring.token energy");
    result.cellFunctionConstructorOffspringTokenSuppressMemoryCopy=
        tree.get<bool>("cell.function.constructor.offspring.token suppress memory copy", false);
    result.cellFunctionConstructorTokenDataMutationProb =
        tree.get<float>(
        "cell.function.constructor.mutation probability.token data");
    result.cellFunctionConstructorCellDataMutationProb =
        tree.get<float>(
        "cell.function.constructor.mutation probability.cell data");
    result.cellFunctionConstructorCellPropertyMutationProb =
        tree.get<float>(
        "cell.function.constructor.mutation probability.cell property");
    result.cellFunctionConstructorCellStructureMutationProb =
        tree.get<float>(
        "cell.function.constructor.mutation probability.cell structure");
    result.cellFunctionSensorRange = tree.get<float>("cell.function.sensor.range");
    result.cellFunctionCommunicatorRange = tree.get<float>("cell.function.communicator.range");
    result.tokenMemorySize = tree.get<int>("token.memory size");
    result.tokenMinEnergy = tree.get<float>("token.min energy");
    result.radiationExponent = tree.get<float>("radiation.exponent");
    result.radiationFactor = tree.get<float>("radiation.factor");
    result.radiationProb = tree.get<float>("radiation.probability");
    result.radiationVelocityMultiplier = tree.get<float>("radiation.velocity multiplier");
    result.radiationVelocityPerturbation = tree.get<float>("radiation.velocity perturbation");

    return result;
}

boost::property_tree::ptree Parser::encode(GeneralSettings const& parameters)
{
    boost::property_tree::ptree tree;
    tree.add("worldSize.x", toString(parameters.worldSize.x));
    tree.add("worldSize.y", toString(parameters.worldSize.y));

    tree.add("cudaSettings.numBlocks", toString(parameters.gpuConstants.NUM_BLOCKS));
    tree.add("cudaSettings.numThreadsPerBlock", toString(parameters.gpuConstants.NUM_THREADS_PER_BLOCK));
    return tree;
}

GeneralSettings Parser::decodeGeneralSettings(boost::property_tree::ptree const& tree)
{
    GeneralSettings result;
    result.worldSize.x = tree.get<int>("worldSize.x");
    result.worldSize.y = tree.get<int>("worldSize.y");

    result.gpuConstants.NUM_BLOCKS = tree.get<int>("cudaSettings.numBlocks");
    result.gpuConstants.NUM_THREADS_PER_BLOCK = tree.get<int>("cudaSettings.numThreadsPerBlock");

    return result;
}
