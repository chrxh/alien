#include "Parser.h"

#include "EngineInterfaceSettings.h"
#include "GeneralSettings.h"
#include "Settings.h"

namespace
{
    std::string toString(int value) { return std::to_string(value); }
    std::string toString(uint64_t value) { return std::to_string(value); }
    std::string toString(float value) { return std::to_string(value); }
    std::string toString(bool value) { return value ? "true" : "false"; }
}

boost::property_tree::ptree Parser::encode(SimulationParameters const& parameters)
{
    boost::property_tree::ptree tree;
    tree.add("time step size", toString(parameters.timestepSize));
    tree.add("friction", toString(parameters.friction));
    tree.add("cell.binding force", toString(parameters.cellBindingForce));
    tree.add("cell.max velocity", toString(parameters.cellMaxVel));
    tree.add("cell.max binding distance", toString(parameters.cellMaxBindingDistance));
    tree.add("cell.repulsion strength", toString(parameters.cellRepulsionStrength));
    tree.add("token.mutation rate", toString(parameters.tokenMutationRate));

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
    result.timestepSize = tree.get<float>("time step size", defaultParameters.timestepSize);
    result.friction = tree.get<float>("friction", defaultParameters.friction);
    result.cellBindingForce = tree.get<float>("cell.binding force", defaultParameters.cellBindingForce);
    result.cellMaxVel= tree.get<float>("cell.max velocity", defaultParameters.cellMaxVel);
    result.cellMaxBindingDistance = tree.get<float>("cell.max binding distance", defaultParameters.cellMaxBindingDistance);
    result.cellRepulsionStrength = tree.get<float>("cell.repulsion strength", defaultParameters.cellRepulsionStrength);
    result.tokenMutationRate = tree.get<float>("token mutation rate", defaultParameters.tokenMutationRate);

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
    return tree;
}

GeneralSettings Parser::decodeGeneralSettings(boost::property_tree::ptree const& tree)
{
    GeneralSettings result;
    result.worldSize.x = tree.get<int>("worldSize.x");
    result.worldSize.y = tree.get<int>("worldSize.y");
    return result;
}

boost::property_tree::ptree Parser::encode(uint64_t timestep, Settings const& settings)
{
    boost::property_tree::ptree tree;

    //general
    tree.add("general.time step", toString(timestep));
    tree.add("general.world size.x", toString(settings.generalSettings.worldSize.x));
    tree.add("general.world size.y", toString(settings.generalSettings.worldSize.y));

    //simulation parameters
    tree.add("simulation parameters.time step size", toString(settings.simulationParameters.timestepSize));
    tree.add("simulation parameters.friction", toString(settings.simulationParameters.friction));
    tree.add("simulation parameters.cell.binding force", toString(settings.simulationParameters.cellBindingForce));
    tree.add("simulation parameters.cell.max velocity", toString(settings.simulationParameters.cellMaxVel));
    tree.add("simulation parameters.cell.max binding distance", toString(settings.simulationParameters.cellMaxBindingDistance));
    tree.add("simulation parameters.cell.repulsion strength", toString(settings.simulationParameters.cellRepulsionStrength));
    tree.add("simulation parameters.token.mutation rate", toString(settings.simulationParameters.tokenMutationRate));

    tree.add("simulation parameters.cell.min distance", toString(settings.simulationParameters.cellMinDistance));
    tree.add("simulation parameters.cell.max distance", toString(settings.simulationParameters.cellMaxCollisionDistance));
    tree.add("simulation parameters.cell.max force", toString(settings.simulationParameters.cellMaxForce));
    tree.add("simulation parameters.cell.max force decay probability", toString(settings.simulationParameters.cellMaxForceDecayProb));
    tree.add("simulation parameters.cell.min token usages", toString(settings.simulationParameters.cellMinTokenUsages));
    tree.add("simulation parameters.cell.token usage decay probability", toString(settings.simulationParameters.cellTokenUsageDecayProb));
    tree.add("simulation parameters.cell.max bonds", toString(settings.simulationParameters.cellMaxBonds));
    tree.add("simulation parameters.cell.max token", toString(settings.simulationParameters.cellMaxToken));
    tree.add("simulation parameters.cell.max token branch number", toString(settings.simulationParameters.cellMaxTokenBranchNumber));
    tree.add("simulation parameters.cell.min energy", toString(settings.simulationParameters.cellMinEnergy));
    tree.add("simulation parameters.cell.transformation probability", toString(settings.simulationParameters.cellTransformationProb));
    tree.add("simulation parameters.cell.fusion velocity", toString(settings.simulationParameters.cellFusionVelocity));
    tree.add("simulation parameters.cell.function.computer.max instructions", toString(settings.simulationParameters.cellFunctionComputerMaxInstructions));
    tree.add("simulation parameters.cell.function.computer.memory size", toString(settings.simulationParameters.cellFunctionComputerCellMemorySize));
    tree.add("simulation parameters.cell.function.weapon.strength", toString(settings.simulationParameters.cellFunctionWeaponStrength));
    tree.add("simulation parameters.cell.function.weapon.energy cost", toString(settings.simulationParameters.cellFunctionWeaponEnergyCost));
    tree.add(
        "simulation parameters.cell.function.weapon.geometry deviation exponent",
        toString(settings.simulationParameters.cellFunctionWeaponGeometryDeviationExponent));
    tree.add(
        "simulation parameters.cell.function.weapon.inhomogeneous color factor",
        toString(settings.simulationParameters.cellFunctionWeaponInhomogeneousColorFactor));
    tree.add(
        "simulation parameters.cell.function.constructor.offspring.cell energy",
        toString(settings.simulationParameters.cellFunctionConstructorOffspringCellEnergy));
    tree.add(
        "simulation parameters.cell.function.constructor.offspring.cell distance",
        toString(settings.simulationParameters.cellFunctionConstructorOffspringCellDistance));
    tree.add(
        "simulation parameters.cell.function.constructor.offspring.token energy",
        toString(settings.simulationParameters.cellFunctionConstructorOffspringTokenEnergy));
    tree.add(
        "simulation parameters.cell.function.constructor.offspring.token suppress memory copy",
        toString(settings.simulationParameters.cellFunctionConstructorOffspringTokenSuppressMemoryCopy));
    tree.add(
        "simulation parameters.cell.function.constructor.mutation probability.token data",
        toString(settings.simulationParameters.cellFunctionConstructorTokenDataMutationProb));
    tree.add(
        "simulation parameters.cell.function.constructor.mutation probability.cell data",
        toString(settings.simulationParameters.cellFunctionConstructorCellDataMutationProb));
    tree.add(
        "simulation parameters.cell.function.constructor.mutation probability.cell property",
        toString(settings.simulationParameters.cellFunctionConstructorCellPropertyMutationProb));
    tree.add(
        "simulation parameters.cell.function.constructor.mutation probability.cell structure",
        toString(settings.simulationParameters.cellFunctionConstructorCellStructureMutationProb));
    tree.add("simulation parameters.cell.function.sensor.range", toString(settings.simulationParameters.cellFunctionSensorRange));
    tree.add("simulation parameters.cell.function.communicator.range", toString(settings.simulationParameters.cellFunctionCommunicatorRange));
    tree.add("simulation parameters.token.memory size", toString(settings.simulationParameters.tokenMemorySize));
    tree.add("simulation parameters.token.min energy", toString(settings.simulationParameters.tokenMinEnergy));
    tree.add("simulation parameters.radiation.exponent", toString(settings.simulationParameters.radiationExponent));
    tree.add("simulation parameters.radiation.factor", toString(settings.simulationParameters.radiationFactor));
    tree.add("simulation parameters.radiation.probability", toString(settings.simulationParameters.radiationProb));
    tree.add("simulation parameters.radiation.velocity multiplier", toString(settings.simulationParameters.radiationVelocityMultiplier));
    tree.add("simulation parameters.radiation.velocity perturbation", toString(settings.simulationParameters.radiationVelocityPerturbation));

    //flow field settings
    tree.add("flow field.active", toString(settings.flowFieldSettings.active));
    tree.add("flow field.num centers", toString(settings.flowFieldSettings.numCenters));
    for (int i = 0; i < 2; ++i) {
        RadialFlowCenterData const& radialData = settings.flowFieldSettings.radialFlowCenters[i];
        std::string node = "flow field.center" + toString(i) + ".";
        tree.add(node + "position.x", toString(radialData.posX));
        tree.add(node + "position.y", toString(radialData.posY));
        tree.add(node + "radius", toString(radialData.radius));
        tree.add(node + "strength", toString(radialData.strength));
    }

    return tree;
}

std::pair<uint64_t, Settings> Parser::decodeTimestepAndSettings(
    boost::property_tree::ptree const& tree)
{
    uint64_t timestep = 0;
    Settings settings;

    //general
    timestep = tree.get<uint64_t>("general.time step", timestep);
    settings.generalSettings.worldSize.x =
        tree.get<int>("general.world size.x");
    settings.generalSettings.worldSize.y =
        tree.get<int>("general.world size.y");

    //simulation parameters
    settings.simulationParameters.timestepSize =
        tree.get<float>("time step size", settings.simulationParameters.timestepSize);
    settings.simulationParameters.friction = tree.get<float>("friction", settings.simulationParameters.friction);
    settings.simulationParameters.cellBindingForce =
        tree.get<float>("cell.binding force", settings.simulationParameters.cellBindingForce);
    settings.simulationParameters.cellMaxVel =
        tree.get<float>("cell.max velocity", settings.simulationParameters.cellMaxVel);
    settings.simulationParameters.cellMaxBindingDistance =
        tree.get<float>("cell.max binding distance", settings.simulationParameters.cellMaxBindingDistance);
    settings.simulationParameters.cellRepulsionStrength =
        tree.get<float>("cell.repulsion strength", settings.simulationParameters.cellRepulsionStrength);
    settings.simulationParameters.tokenMutationRate =
        tree.get<float>("token mutation rate", settings.simulationParameters.tokenMutationRate);

    settings.simulationParameters.cellMinDistance = tree.get<float>("cell.min distance");
    settings.simulationParameters.cellMaxCollisionDistance = tree.get<float>("cell.max distance");
    settings.simulationParameters.cellMaxForce = tree.get<float>("cell.max force");
    settings.simulationParameters.cellMaxForceDecayProb = tree.get<float>("cell.max force decay probability");
    settings.simulationParameters.cellMinTokenUsages = tree.get<int>("cell.min token usages");
    settings.simulationParameters.cellTokenUsageDecayProb = tree.get<float>("cell.token usage decay probability");
    settings.simulationParameters.cellMaxBonds = tree.get<int>("cell.max bonds");
    settings.simulationParameters.cellMaxToken = tree.get<int>("cell.max token");
    settings.simulationParameters.cellMaxTokenBranchNumber = tree.get<int>("cell.max token branch number");
    settings.simulationParameters.cellMinEnergy = tree.get<float>("cell.min energy");
    settings.simulationParameters.cellTransformationProb = tree.get<float>("cell.transformation probability");
    settings.simulationParameters.cellFusionVelocity = tree.get<float>("cell.fusion velocity");
    settings.simulationParameters.cellFunctionComputerMaxInstructions = tree.get<int>("cell.function.computer.max instructions");
    settings.simulationParameters.cellFunctionComputerCellMemorySize = tree.get<int>("cell.function.computer.memory size");
    settings.simulationParameters.cellFunctionWeaponStrength = tree.get<float>("cell.function.weapon.strength");
    settings.simulationParameters.cellFunctionWeaponEnergyCost = tree.get<float>("cell.function.weapon.energy cost");
    settings.simulationParameters.cellFunctionWeaponGeometryDeviationExponent = tree.get<float>(
        "cell.function.weapon.geometry deviation exponent",
        settings.simulationParameters.cellFunctionWeaponGeometryDeviationExponent);
    settings.simulationParameters.cellFunctionWeaponInhomogeneousColorFactor = tree.get<float>(
        "cell.function.weapon.inhomogeneous color factor",
        settings.simulationParameters.cellFunctionWeaponInhomogeneousColorFactor);
    settings.simulationParameters.cellFunctionConstructorOffspringCellEnergy =
        tree.get<float>("cell.function.constructor.offspring.cell energy");
    settings.simulationParameters.cellFunctionConstructorOffspringCellDistance =
        tree.get<float>("cell.function.constructor.offspring.cell distance");
    settings.simulationParameters.cellFunctionConstructorOffspringTokenEnergy =
        tree.get<float>("cell.function.constructor.offspring.token energy");
    settings.simulationParameters.cellFunctionConstructorOffspringTokenSuppressMemoryCopy =
        tree.get<bool>("cell.function.constructor.offspring.token suppress memory copy", false);
    settings.simulationParameters.cellFunctionConstructorTokenDataMutationProb =
        tree.get<float>("cell.function.constructor.mutation probability.token data");
    settings.simulationParameters.cellFunctionConstructorCellDataMutationProb =
        tree.get<float>("cell.function.constructor.mutation probability.cell data");
    settings.simulationParameters.cellFunctionConstructorCellPropertyMutationProb =
        tree.get<float>("cell.function.constructor.mutation probability.cell property");
    settings.simulationParameters.cellFunctionConstructorCellStructureMutationProb =
        tree.get<float>("cell.function.constructor.mutation probability.cell structure");
    settings.simulationParameters.cellFunctionSensorRange = tree.get<float>("cell.function.sensor.range");
    settings.simulationParameters.cellFunctionCommunicatorRange = tree.get<float>("cell.function.communicator.range");
    settings.simulationParameters.tokenMemorySize = tree.get<int>("token.memory size");
    settings.simulationParameters.tokenMinEnergy = tree.get<float>("token.min energy");
    settings.simulationParameters.radiationExponent = tree.get<float>("radiation.exponent");
    settings.simulationParameters.radiationFactor = tree.get<float>("radiation.factor");
    settings.simulationParameters.radiationProb = tree.get<float>("radiation.probability");
    settings.simulationParameters.radiationVelocityMultiplier = tree.get<float>("radiation.velocity multiplier");
    settings.simulationParameters.radiationVelocityPerturbation = tree.get<float>("radiation.velocity perturbation");

    //flow field settings
    settings.flowFieldSettings.active = tree.get<bool>("flow field.active");
    settings.flowFieldSettings.numCenters = tree.get<int>("flow field.num centers");
    for (int i = 0; i < 2; ++i) {
        RadialFlowCenterData& radialData = settings.flowFieldSettings.radialFlowCenters[i];
        std::string node = "flow field.center" + toString(i) + ".";

        radialData.posX = tree.get<float>(node + "position.x");
        radialData.posY = tree.get<float>(node + "position.y");
        radialData.radius = tree.get<float>(node + "radius");
        radialData.strength = tree.get<float>(node + "strength");
    }

    return std::make_pair(timestep, settings);
}
