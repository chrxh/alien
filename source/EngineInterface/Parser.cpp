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

boost::property_tree::ptree Parser::encode(uint64_t timestep, Settings const& settings)
{
    boost::property_tree::ptree tree;

    //general
    tree.add("general.time step", toString(timestep));
    tree.add("general.world size.x", toString(settings.generalSettings.worldSizeX));
    tree.add("general.world size.y", toString(settings.generalSettings.worldSizeY));

    //simulation parameters
    tree.add("simulation parameters.time step size", toString(settings.simulationParameters.timestepSize));
    tree.add("simulation parameters.friction", toString(settings.simulationParameters.spotValues.friction));
    tree.add(
        "simulation parameters.cell.binding force",
        toString(settings.simulationParameters.spotValues.cellBindingForce));
    tree.add("simulation parameters.cell.max velocity", toString(settings.simulationParameters.cellMaxVel));
    tree.add("simulation parameters.cell.max binding distance", toString(settings.simulationParameters.cellMaxBindingDistance));
    tree.add("simulation parameters.cell.repulsion strength", toString(settings.simulationParameters.cellRepulsionStrength));
    tree.add(
        "simulation parameters.token.mutation rate",
        toString(settings.simulationParameters.spotValues.tokenMutationRate));

    tree.add("simulation parameters.cell.min distance", toString(settings.simulationParameters.cellMinDistance));
    tree.add("simulation parameters.cell.max distance", toString(settings.simulationParameters.cellMaxCollisionDistance));
    tree.add("simulation parameters.cell.max force", toString(settings.simulationParameters.spotValues.cellMaxForce));
    tree.add("simulation parameters.cell.max force decay probability", toString(settings.simulationParameters.cellMaxForceDecayProb));
    tree.add("simulation parameters.cell.min token usages", toString(settings.simulationParameters.cellMinTokenUsages));
    tree.add("simulation parameters.cell.token usage decay probability", toString(settings.simulationParameters.cellTokenUsageDecayProb));
    tree.add("simulation parameters.cell.max bonds", toString(settings.simulationParameters.cellMaxBonds));
    tree.add("simulation parameters.cell.max token", toString(settings.simulationParameters.cellMaxToken));
    tree.add("simulation parameters.cell.max token branch number", toString(settings.simulationParameters.cellMaxTokenBranchNumber));
    tree.add("simulation parameters.cell.min energy", toString(settings.simulationParameters.spotValues.cellMinEnergy));
    tree.add("simulation parameters.cell.transformation probability", toString(settings.simulationParameters.cellTransformationProb));
    tree.add(
        "simulation parameters.cell.fusion velocity",
        toString(settings.simulationParameters.spotValues.cellFusionVelocity));
    tree.add("simulation parameters.cell.function.computer.max instructions", toString(settings.simulationParameters.cellFunctionComputerMaxInstructions));
    tree.add("simulation parameters.cell.function.computer.memory size", toString(settings.simulationParameters.cellFunctionComputerCellMemorySize));
    tree.add("simulation parameters.cell.function.weapon.strength", toString(settings.simulationParameters.cellFunctionWeaponStrength));
    tree.add(
        "simulation parameters.cell.function.weapon.energy cost",
        toString(settings.simulationParameters.spotValues.cellFunctionWeaponEnergyCost));
    tree.add(
        "simulation parameters.cell.function.weapon.geometry deviation exponent",
        toString(settings.simulationParameters.spotValues.cellFunctionWeaponGeometryDeviationExponent));
    tree.add(
        "simulation parameters.cell.function.weapon.color penalty",
        toString(settings.simulationParameters.spotValues.cellFunctionWeaponColorPenalty));
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
    tree.add(
        "simulation parameters.radiation.factor", toString(settings.simulationParameters.spotValues.radiationFactor));
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
    settings.generalSettings.worldSizeX =
        tree.get<int>("general.world size.x");
    settings.generalSettings.worldSizeY =
        tree.get<int>("general.world size.y");

    //simulation parameters
    settings.simulationParameters.timestepSize =
        tree.get<float>("simulation parameters.time step size", settings.simulationParameters.timestepSize);
    settings.simulationParameters.spotValues.friction =
        tree.get<float>("simulation parameters.friction", settings.simulationParameters.spotValues.friction);
    settings.simulationParameters.spotValues.cellBindingForce = tree.get<float>(
        "simulation parameters.cell.binding force", settings.simulationParameters.spotValues.cellBindingForce);
    settings.simulationParameters.cellMaxVel =
        tree.get<float>("simulation parameters.cell.max velocity", settings.simulationParameters.cellMaxVel);
    settings.simulationParameters.cellMaxBindingDistance =
        tree.get<float>("simulation parameters.cell.max binding distance", settings.simulationParameters.cellMaxBindingDistance);
    settings.simulationParameters.cellRepulsionStrength =
        tree.get<float>("simulation parameters.cell.repulsion strength", settings.simulationParameters.cellRepulsionStrength);
    settings.simulationParameters.spotValues.tokenMutationRate = tree.get<float>(
        "simulation parameters.token mutation rate", settings.simulationParameters.spotValues.tokenMutationRate);

    settings.simulationParameters.cellMinDistance = tree.get<float>("simulation parameters.cell.min distance");
    settings.simulationParameters.cellMaxCollisionDistance = tree.get<float>("simulation parameters.cell.max distance");
    settings.simulationParameters.spotValues.cellMaxForce = tree.get<float>("simulation parameters.cell.max force");
    settings.simulationParameters.cellMaxForceDecayProb = tree.get<float>("simulation parameters.cell.max force decay probability");
    settings.simulationParameters.cellMinTokenUsages = tree.get<int>("simulation parameters.cell.min token usages");
    settings.simulationParameters.cellTokenUsageDecayProb = tree.get<float>("simulation parameters.cell.token usage decay probability");
    settings.simulationParameters.cellMaxBonds = tree.get<int>("simulation parameters.cell.max bonds");
    settings.simulationParameters.cellMaxToken = tree.get<int>("simulation parameters.cell.max token");
    settings.simulationParameters.cellMaxTokenBranchNumber = tree.get<int>("simulation parameters.cell.max token branch number");
    settings.simulationParameters.spotValues.cellMinEnergy = tree.get<float>("simulation parameters.cell.min energy");
    settings.simulationParameters.cellTransformationProb = tree.get<float>("simulation parameters.cell.transformation probability");
    settings.simulationParameters.spotValues.cellFusionVelocity =
        tree.get<float>("simulation parameters.cell.fusion velocity");
    settings.simulationParameters.cellFunctionComputerMaxInstructions = tree.get<int>("simulation parameters.cell.function.computer.max instructions");
    settings.simulationParameters.cellFunctionComputerCellMemorySize = tree.get<int>("simulation parameters.cell.function.computer.memory size");
    settings.simulationParameters.cellFunctionWeaponStrength = tree.get<float>("simulation parameters.cell.function.weapon.strength");
    settings.simulationParameters.spotValues.cellFunctionWeaponEnergyCost =
        tree.get<float>("simulation parameters.cell.function.weapon.energy cost");
    settings.simulationParameters.spotValues.cellFunctionWeaponGeometryDeviationExponent = tree.get<float>(
        "simulation parameters.cell.function.weapon.geometry deviation exponent",
        settings.simulationParameters.spotValues.cellFunctionWeaponGeometryDeviationExponent);
    settings.simulationParameters.spotValues.cellFunctionWeaponColorPenalty = tree.get<float>(
        "simulation parameters.cell.function.weapon.color penalty",
        settings.simulationParameters.spotValues.cellFunctionWeaponColorPenalty);
    settings.simulationParameters.cellFunctionConstructorOffspringCellEnergy =
        tree.get<float>("simulation parameters.cell.function.constructor.offspring.cell energy");
    settings.simulationParameters.cellFunctionConstructorOffspringCellDistance =
        tree.get<float>("simulation parameters.cell.function.constructor.offspring.cell distance");
    settings.simulationParameters.cellFunctionConstructorOffspringTokenEnergy =
        tree.get<float>("simulation parameters.cell.function.constructor.offspring.token energy");
    settings.simulationParameters.cellFunctionConstructorOffspringTokenSuppressMemoryCopy =
        tree.get<bool>("simulation parameters.cell.function.constructor.offspring.token suppress memory copy", false);
    settings.simulationParameters.cellFunctionConstructorTokenDataMutationProb =
        tree.get<float>("simulation parameters.cell.function.constructor.mutation probability.token data");
    settings.simulationParameters.cellFunctionConstructorCellDataMutationProb =
        tree.get<float>("simulation parameters.cell.function.constructor.mutation probability.cell data");
    settings.simulationParameters.cellFunctionConstructorCellPropertyMutationProb =
        tree.get<float>("simulation parameters.cell.function.constructor.mutation probability.cell property");
    settings.simulationParameters.cellFunctionConstructorCellStructureMutationProb =
        tree.get<float>("simulation parameters.cell.function.constructor.mutation probability.cell structure");
    settings.simulationParameters.cellFunctionSensorRange = tree.get<float>("simulation parameters.cell.function.sensor.range");
    settings.simulationParameters.cellFunctionCommunicatorRange = tree.get<float>("simulation parameters.cell.function.communicator.range");
    settings.simulationParameters.tokenMemorySize = tree.get<int>("simulation parameters.token.memory size");
    settings.simulationParameters.tokenMinEnergy = tree.get<float>("simulation parameters.token.min energy");
    settings.simulationParameters.radiationExponent = tree.get<float>("simulation parameters.radiation.exponent");
    settings.simulationParameters.spotValues.radiationFactor =
        tree.get<float>("simulation parameters.radiation.factor");
    settings.simulationParameters.radiationProb = tree.get<float>("simulation parameters.radiation.probability");
    settings.simulationParameters.radiationVelocityMultiplier = tree.get<float>("simulation parameters.radiation.velocity multiplier");
    settings.simulationParameters.radiationVelocityPerturbation = tree.get<float>("simulation parameters.radiation.velocity perturbation");

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
