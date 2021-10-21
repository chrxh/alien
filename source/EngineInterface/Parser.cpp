#include "Parser.h"

#include "EngineInterfaceSettings.h"
#include "GeneralSettings.h"
#include "Settings.h"

namespace
{
    template<typename T>
    std::string toString(T value) { return std::to_string(value); }

    template <>
    std::string toString<bool>(bool value)
    {
        return value ? "true" : "false";
    }
}

boost::property_tree::ptree Parser::encode(uint64_t timestep, Settings settings)
{
    boost::property_tree::ptree tree;
    encodeDecode(tree, timestep, settings, Task::Encode);
    return tree;
}

std::pair<uint64_t, Settings> Parser::decodeTimestepAndSettings(
    boost::property_tree::ptree tree)
{
    uint64_t timestep;
    Settings settings;
    encodeDecode(tree, timestep, settings, Task::Decode);
    return std::make_pair(timestep, settings);
}

void Parser::encodeDecode(boost::property_tree::ptree& tree, uint64_t& timestep, Settings& settings, Task task)
{
    Settings defaultSettings;

    //general settings
    encodeDecode(tree, timestep, uint64_t(0), "general.time step", task);
    encodeDecode(
        tree,
        settings.generalSettings.worldSizeX,
        defaultSettings.generalSettings.worldSizeX,
        "general.world size.x",
        task);
    encodeDecode(
        tree,
        settings.generalSettings.worldSizeY,
        defaultSettings.generalSettings.worldSizeY,
        "general.world size.y",
        task);

    //simulation parameters
    auto& simPar = settings.simulationParameters;
    auto& defaultPar = defaultSettings.simulationParameters;
    encodeDecode(tree, simPar.timestepSize, defaultPar.timestepSize, "simulation parameters.time step size", task);
    encodeDecode(
        tree, simPar.spotValues.friction, defaultPar.spotValues.friction, "simulation parameters.friction", task);
    encodeDecode(
        tree,
        simPar.spotValues.cellBindingForce,
        defaultPar.spotValues.cellBindingForce,
        "simulation parameters.cell.binding force",
        task);
    encodeDecode(tree, simPar.cellMaxVel, defaultPar.cellMaxVel, "simulation parameters.cell.max velocity", task);
    encodeDecode(
        tree,
        simPar.cellMaxBindingDistance,
        defaultPar.cellMaxBindingDistance,
        "simulation parameters.cell.max binding distance",
        task);
    encodeDecode(
        tree,
        simPar.cellRepulsionStrength,
        defaultPar.cellRepulsionStrength,
        "simulation parameters.cell.repulsion strength",
        task);
    encodeDecode(
        tree,
        simPar.spotValues.tokenMutationRate,
        defaultPar.spotValues.tokenMutationRate,
        "simulation parameters.token.mutation rate",
        task);

    encodeDecode(
        tree, simPar.cellMinDistance, defaultPar.cellMinDistance, "simulation parameters.cell.min distance", task);
    encodeDecode(
        tree,
        simPar.cellMaxCollisionDistance,
        defaultPar.cellMaxCollisionDistance,
        "simulation parameters.cell.max distance",
        task);
    encodeDecode(
        tree,
        simPar.spotValues.cellMaxForce,
        defaultPar.spotValues.cellMaxForce,
        "simulation parameters.cell.max force",
        task);
    encodeDecode(
        tree,
        simPar.cellMaxForceDecayProb,
        defaultPar.cellMaxForceDecayProb,
        "simulation parameters.cell.max force decay probability",
        task);
    encodeDecode(
        tree,
        simPar.cellMinTokenUsages,
        defaultPar.cellMinTokenUsages,
        "simulation parameters.cell.min token usages",
        task);
    encodeDecode(
        tree,
        simPar.cellTokenUsageDecayProb,
        defaultPar.cellTokenUsageDecayProb,
        "simulation parameters.cell.token usage decay probability",
        task);
    encodeDecode(tree, simPar.cellMaxBonds, defaultPar.cellMaxBonds, "simulation parameters.cell.max bonds", task);
    encodeDecode(tree, simPar.cellMaxToken, defaultPar.cellMaxToken, "simulation parameters.cell.max token", task);
    encodeDecode(
        tree,
        simPar.cellMaxTokenBranchNumber,
        defaultPar.cellMaxTokenBranchNumber,
        "simulation parameters.cell.max token branch number",
        task);
    encodeDecode(
        tree,
        simPar.spotValues.cellMinEnergy,
        defaultPar.spotValues.cellMinEnergy,
        "simulation parameters.cell.min energy",
        task);
    encodeDecode(
        tree,
        simPar.cellTransformationProb,
        defaultPar.cellTransformationProb,
        "simulation parameters.cell.transformation probability",
        task);
    encodeDecode(
        tree,
        simPar.spotValues.cellFusionVelocity,
        defaultPar.spotValues.cellFusionVelocity,
        "simulation parameters.cell.fusion velocity",
        task);
    encodeDecode(
        tree,
        simPar.cellFunctionComputerMaxInstructions,
        defaultPar.cellFunctionComputerMaxInstructions,
        "simulation parameters.cell.function.computer.max instructions",
        task);
    encodeDecode(
        tree,
        simPar.cellFunctionComputerCellMemorySize,
        defaultPar.cellFunctionComputerCellMemorySize,
        "simulation parameters.cell.function.computer.memory size",
        task);
    encodeDecode(
        tree,
        simPar.cellFunctionWeaponStrength,
        defaultPar.cellFunctionWeaponStrength,
        "simulation parameters.cell.function.weapon.strength",
        task);
    encodeDecode(
        tree,
        simPar.spotValues.cellFunctionWeaponEnergyCost,
        defaultPar.spotValues.cellFunctionWeaponEnergyCost,
        "simulation parameters.cell.function.weapon.energy cost",
        task);
    encodeDecode(
        tree,
        simPar.spotValues.cellFunctionWeaponGeometryDeviationExponent,
        defaultPar.spotValues.cellFunctionWeaponGeometryDeviationExponent,
        "simulation parameters.cell.function.weapon.geometry deviation exponent",
        task);
    encodeDecode(
        tree,
        simPar.spotValues.cellFunctionWeaponColorPenalty,
        defaultPar.spotValues.cellFunctionWeaponColorPenalty,
        "simulation parameters.cell.function.weapon.color penalty",
        task);
    encodeDecode(
        tree,
        simPar.cellFunctionConstructorOffspringCellEnergy,
        defaultPar.cellFunctionConstructorOffspringCellEnergy,
        "simulation parameters.cell.function.constructor.offspring.cell energy",
        task);
    encodeDecode(
        tree,
        simPar.cellFunctionConstructorOffspringCellDistance,
        defaultPar.cellFunctionConstructorOffspringCellDistance,
        "simulation parameters.cell.function.constructor.offspring.cell distance",
        task);
    encodeDecode(
        tree,
        simPar.cellFunctionConstructorOffspringTokenEnergy,
        defaultPar.cellFunctionConstructorOffspringTokenEnergy,
        "simulation parameters.cell.function.constructor.offspring.token energy",
        task);
    encodeDecode(
        tree,
        simPar.cellFunctionConstructorOffspringTokenSuppressMemoryCopy,
        defaultPar.cellFunctionConstructorOffspringTokenSuppressMemoryCopy,
        "simulation parameters.cell.function.constructor.offspring.token suppress memory copy",
        task);
    encodeDecode(
        tree,
        simPar.cellFunctionConstructorTokenDataMutationProb,
        defaultPar.cellFunctionConstructorTokenDataMutationProb,
        "simulation parameters.cell.function.constructor.offspring.token suppress memory copy",
        task);
    /*
    settings.simulationParameters.cellFunctionConstructorTokenDataMutationProb =
        tree.get<float>("simulation parameters.cell.function.constructor.mutation probability.token data");
    settings.simulationParameters.cellFunctionConstructorCellDataMutationProb =
        tree.get<float>("simulation parameters.cell.function.constructor.mutation probability.cell data");
    settings.simulationParameters.cellFunctionConstructorCellPropertyMutationProb =
        tree.get<float>("simulation parameters.cell.function.constructor.mutation probability.cell property");
    settings.simulationParameters.cellFunctionConstructorCellStructureMutationProb =
        tree.get<float>("simulation parameters.cell.function.constructor.mutation probability.cell structure");
*/
    encodeDecode(
        tree,
        simPar.cellFunctionSensorRange,
        defaultPar.cellFunctionSensorRange,
        "simulation parameters.cell.function.sensor.range",
        task);
    encodeDecode(
        tree,
        simPar.cellFunctionCommunicatorRange,
        defaultPar.cellFunctionCommunicatorRange,
        "simulation parameters.cell.function.communicator.range",
        task);
    encodeDecode(
        tree, simPar.tokenMemorySize, defaultPar.tokenMemorySize, "simulation parameters.token.memory size", task);
    encodeDecode(
        tree, simPar.tokenMinEnergy, defaultPar.tokenMinEnergy, "simulation parameters.token.min energy", task);
    encodeDecode(
        tree, simPar.radiationExponent, defaultPar.radiationExponent, "simulation parameters.radiation.exponent", task);
    encodeDecode(
        tree,
        simPar.spotValues.radiationFactor,
        defaultPar.spotValues.radiationFactor,
        "simulation parameters.radiation.factor",
        task);
    encodeDecode(
        tree, simPar.radiationProb, defaultPar.radiationProb, "simulation parameters.radiation.probability", task);
    encodeDecode(
        tree,
        simPar.radiationVelocityMultiplier,
        defaultPar.radiationVelocityMultiplier,
        "simulation parameters.radiation.velocity multiplier",
        task);
    encodeDecode(
        tree,
        simPar.radiationVelocityPerturbation,
        defaultPar.radiationVelocityPerturbation,
        "simulation parameters.radiation.velocity perturbation",
        task);

    //spots
    auto& spots = settings.simulationParametersSpots;
    auto& defaultSpots = defaultSettings.simulationParametersSpots;
    encodeDecode(tree, spots.numSpots, defaultSpots.numSpots, "simulation parameters.spots.num spots", task);
    for (int index = 0; index <= 1; ++index) {
        std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
        auto& spot = spots.spots[index];
        auto& defaultSpot = defaultSpots.spots[index];
        encodeDecode(tree, spot.color, defaultSpot.color, base + "color", task);
        encodeDecode(tree, spot.posX, defaultSpot.posX, base + "pos.x", task);
        encodeDecode(tree, spot.posY, defaultSpot.posY, base + "pos.y", task);
        encodeDecode(tree, spot.coreRadius, defaultSpot.coreRadius, base + "core radius", task);
        encodeDecode(tree, spot.fadeoutRadius, defaultSpot.fadeoutRadius, base + "fadeout radius", task);
        encodeDecode(tree, spot.values.friction, defaultSpot.values.friction, base + "friction", task);
        encodeDecode(
            tree, spot.values.radiationFactor, defaultSpot.values.radiationFactor, base + "radiation.factor", task);
        encodeDecode(tree, spot.values.cellMaxForce, defaultSpot.values.cellMaxForce, base + "cell.max force", task);
        encodeDecode(tree, spot.values.cellMinEnergy, defaultSpot.values.cellMinEnergy, base + "cell.min energy", task);

        encodeDecode(
            tree, spot.values.cellBindingForce, defaultSpot.values.cellBindingForce, base + "cell.binding force", task);
        encodeDecode(
            tree,
            spot.values.cellFusionVelocity,
            defaultSpot.values.cellFusionVelocity,
            base + "cell.fusion velocity",
            task);
        encodeDecode(
            tree,
            spot.values.tokenMutationRate,
            defaultSpot.values.tokenMutationRate,
            base + "token.mutation rate",
            task);
        encodeDecode(
            tree,
            spot.values.cellFunctionWeaponEnergyCost,
            defaultSpot.values.cellFunctionWeaponEnergyCost,
            base + "cell.function.weapon.energy cost",
            task);
        encodeDecode(
            tree,
            spot.values.cellFunctionWeaponColorPenalty,
            defaultSpot.values.cellFunctionWeaponColorPenalty,
            base + "cell.function.weapon.color penalty",
            task);
        encodeDecode(
            tree,
            spot.values.cellFunctionWeaponGeometryDeviationExponent,
            defaultSpot.values.cellFunctionWeaponGeometryDeviationExponent,
            base + "cell.function.weapon.geometry deviation exponent",
            task);
    }

    //flow field settings
    encodeDecode(
        tree, settings.flowFieldSettings.active, defaultSettings.flowFieldSettings.active, "flow field.active", task);
    encodeDecode(
        tree,
        settings.flowFieldSettings.numCenters,
        defaultSettings.flowFieldSettings.numCenters,
        "flow field.num centers",
        task);
    for (int i = 0; i < 2; ++i) {
        std::string node = "flow field.center" + toString(i) + ".";
        auto& radialData = settings.flowFieldSettings.centers[i];
        auto& defaultRadialData = defaultSettings.flowFieldSettings.centers[i];
        encodeDecode(tree, radialData.posX, defaultRadialData.posX, node + "pos.x", task);
        encodeDecode(tree, radialData.posY, defaultRadialData.posY, node + "pos.y", task);
        encodeDecode(tree, radialData.radius, defaultRadialData.radius, node + "radius", task);
        encodeDecode(tree, radialData.strength, defaultRadialData.strength, node + "strength", task);
    }
}

template <typename T>
void Parser::encodeDecode(
    boost::property_tree::ptree& tree,
    T& parameter,
    T const& defaultValue,
    std::string const& node,
    Task task)
{
    if (Task::Encode == task) {
        tree.add(node, toString(parameter));
    } else {
        parameter = tree.get<T>(node, defaultValue);
    }
}
