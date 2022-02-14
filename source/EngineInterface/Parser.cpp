#include "Parser.h"

#include "GeneralSettings.h"
#include "Settings.h"

boost::property_tree::ptree Parser::encode(uint64_t timestep, Settings settings)
{
    boost::property_tree::ptree tree;
    encodeDecode(tree, timestep, settings, ParserTask::Encode);
    return tree;
}

std::pair<uint64_t, Settings> Parser::decodeTimestepAndSettings(
    boost::property_tree::ptree tree)
{
    uint64_t timestep;
    Settings settings;
    encodeDecode(tree, timestep, settings, ParserTask::Decode);
    return std::make_pair(timestep, settings);
}

namespace
{
    std::unordered_map<SpotShape, std::string> shapeStringMap = {{SpotShape::Circular, "Circular"}, {SpotShape::Rectangular, "Rectangular"}};
    std::unordered_map<std::string, SpotShape> shapeEnumMap = {{"Circular", SpotShape::Circular}, {"Rectangular", SpotShape::Rectangular}};
}

void Parser::encodeDecode(boost::property_tree::ptree& tree, uint64_t& timestep, Settings& settings, ParserTask ParserTask)
{
    Settings defaultSettings;

    //general settings
    JsonParser::encodeDecode(tree, timestep, uint64_t(0), "general.time step", ParserTask);
    JsonParser::encodeDecode(
        tree,
        settings.generalSettings.worldSizeX,
        defaultSettings.generalSettings.worldSizeX,
        "general.world size.x",
        ParserTask);
    JsonParser::encodeDecode(
        tree,
        settings.generalSettings.worldSizeY,
        defaultSettings.generalSettings.worldSizeY,
        "general.world size.y",
        ParserTask);

    //simulation parameters
    auto& simPar = settings.simulationParameters;
    auto& defaultPar = defaultSettings.simulationParameters;
    JsonParser::encodeDecode(tree, simPar.timestepSize, defaultPar.timestepSize, "simulation parameters.time step size", ParserTask);
    JsonParser::encodeDecode(tree, simPar.spotValues.friction, defaultPar.spotValues.friction, "simulation parameters.friction", ParserTask);
    JsonParser::encodeDecode(tree, simPar.spotValues.rigidity, defaultPar.spotValues.rigidity, "simulation parameters.rigidity", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.spotValues.cellBindingForce, defaultPar.spotValues.cellBindingForce, "simulation parameters.cell.binding force", ParserTask);
    JsonParser::encodeDecode(tree, simPar.cellMaxVel, defaultPar.cellMaxVel, "simulation parameters.cell.max velocity", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellMaxBindingDistance, defaultPar.cellMaxBindingDistance, "simulation parameters.cell.max binding distance", ParserTask);
    JsonParser::encodeDecode(tree, simPar.cellRepulsionStrength, defaultPar.cellRepulsionStrength, "simulation parameters.cell.repulsion strength", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.spotValues.tokenMutationRate, defaultPar.spotValues.tokenMutationRate, "simulation parameters.token.mutation rate", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.spotValues.cellMutationRate, defaultPar.spotValues.cellMutationRate, "simulation parameters.cell.mutation rate", ParserTask);

    JsonParser::encodeDecode(tree, simPar.cellMinDistance, defaultPar.cellMinDistance, "simulation parameters.cell.min distance", ParserTask);
    JsonParser::encodeDecode(tree, simPar.cellMaxCollisionDistance, defaultPar.cellMaxCollisionDistance, "simulation parameters.cell.max distance", ParserTask);
    JsonParser::encodeDecode(tree, simPar.spotValues.cellMaxForce, defaultPar.spotValues.cellMaxForce, "simulation parameters.cell.max force", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellMaxForceDecayProb, defaultPar.cellMaxForceDecayProb, "simulation parameters.cell.max force decay probability", ParserTask);
    JsonParser::encodeDecode(tree, simPar.cellMinTokenUsages, defaultPar.cellMinTokenUsages, "simulation parameters.cell.min token usages", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellTokenUsageDecayProb, defaultPar.cellTokenUsageDecayProb, "simulation parameters.cell.token usage decay probability", ParserTask);
    JsonParser::encodeDecode(tree, simPar.cellMaxBonds, defaultPar.cellMaxBonds, "simulation parameters.cell.max bonds", ParserTask);
    JsonParser::encodeDecode(tree, simPar.cellMaxToken, defaultPar.cellMaxToken, "simulation parameters.cell.max token", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellMaxTokenBranchNumber, defaultPar.cellMaxTokenBranchNumber, "simulation parameters.cell.max token branch number", ParserTask);
    JsonParser::encodeDecode(tree, simPar.spotValues.cellMinEnergy, defaultPar.spotValues.cellMinEnergy, "simulation parameters.cell.min energy", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellTransformationProb, defaultPar.cellTransformationProb, "simulation parameters.cell.transformation probability", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.spotValues.cellFusionVelocity, defaultPar.spotValues.cellFusionVelocity, "simulation parameters.cell.fusion velocity", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.spotValues.cellMaxBindingEnergy, simPar.spotValues.cellMaxBindingEnergy, "simulation parameters.cell.max binding energy", ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionComputerMaxInstructions,
        defaultPar.cellFunctionComputerMaxInstructions,
        "simulation parameters.cell.function.computer.max instructions",
        ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionComputerCellMemorySize,
        defaultPar.cellFunctionComputerCellMemorySize,
        "simulation parameters.cell.function.computer.memory size",
        ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellFunctionWeaponStrength, defaultPar.cellFunctionWeaponStrength, "simulation parameters.cell.function.weapon.strength", ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionWeaponEnergyCost,
        defaultPar.spotValues.cellFunctionWeaponEnergyCost,
        "simulation parameters.cell.function.weapon.energy cost",
        ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionWeaponGeometryDeviationExponent,
        defaultPar.spotValues.cellFunctionWeaponGeometryDeviationExponent,
        "simulation parameters.cell.function.weapon.geometry deviation exponent",
        ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionWeaponColorPenalty,
        defaultPar.spotValues.cellFunctionWeaponColorPenalty,
        "simulation parameters.cell.function.weapon.color penalty",
        ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionConstructorOffspringCellEnergy,
        defaultPar.cellFunctionConstructorOffspringCellEnergy,
        "simulation parameters.cell.function.constructor.offspring.cell energy",
        ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionConstructorOffspringCellDistance,
        defaultPar.cellFunctionConstructorOffspringCellDistance,
        "simulation parameters.cell.function.constructor.offspring.cell distance",
        ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionConstructorOffspringTokenEnergy,
        defaultPar.cellFunctionConstructorOffspringTokenEnergy,
        "simulation parameters.cell.function.constructor.offspring.token energy",
        ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionConstructorOffspringTokenSuppressMemoryCopy,
        defaultPar.cellFunctionConstructorOffspringTokenSuppressMemoryCopy,
        "simulation parameters.cell.function.constructor.offspring.token suppress memory copy",
        ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionConstructorTokenDataMutationProb,
        defaultPar.cellFunctionConstructorTokenDataMutationProb,
        "simulation parameters.cell.function.constructor.offspring.token suppress memory copy",
        ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellFunctionSensorRange, defaultPar.cellFunctionSensorRange, "simulation parameters.cell.function.sensor.range", ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionCommunicatorRange,
        defaultPar.cellFunctionCommunicatorRange,
        "simulation parameters.cell.function.communicator.range",
        ParserTask);
    JsonParser::encodeDecode(tree, simPar.tokenMemorySize, defaultPar.tokenMemorySize, "simulation parameters.token.memory size", ParserTask);
    JsonParser::encodeDecode(tree, simPar.tokenMinEnergy, defaultPar.tokenMinEnergy, "simulation parameters.token.min energy", ParserTask);
    JsonParser::encodeDecode(tree, simPar.radiationExponent, defaultPar.radiationExponent, "simulation parameters.radiation.exponent", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.spotValues.radiationFactor, defaultPar.spotValues.radiationFactor, "simulation parameters.radiation.factor", ParserTask);
    JsonParser::encodeDecode(tree, simPar.radiationProb, defaultPar.radiationProb, "simulation parameters.radiation.probability", ParserTask);
    JsonParser::encodeDecode(
        tree, simPar.radiationVelocityMultiplier, defaultPar.radiationVelocityMultiplier, "simulation parameters.radiation.velocity multiplier", ParserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.radiationVelocityPerturbation,
        defaultPar.radiationVelocityPerturbation,
        "simulation parameters.radiation.velocity perturbation",
        ParserTask);

    //spots
    auto& spots = settings.simulationParametersSpots;
    auto& defaultSpots = defaultSettings.simulationParametersSpots;
    JsonParser::encodeDecode(tree, spots.numSpots, defaultSpots.numSpots, "simulation parameters.spots.num spots", ParserTask);
    for (int index = 0; index <= 1; ++index) {
        std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
        auto& spot = spots.spots[index];
        auto& defaultSpot = defaultSpots.spots[index];
        JsonParser::encodeDecode(tree, spot.color, defaultSpot.color, base + "color", ParserTask);
        JsonParser::encodeDecode(tree, spot.posX, defaultSpot.posX, base + "pos.x", ParserTask);
        JsonParser::encodeDecode(tree, spot.posY, defaultSpot.posY, base + "pos.y", ParserTask);

        auto shapeString = shapeStringMap.at(spot.shape);
        JsonParser::encodeDecode(tree, shapeString, shapeStringMap.at(defaultSpot.shape), base + "shape", ParserTask);
        spot.shape = shapeEnumMap.at(shapeString);

        JsonParser::encodeDecode(tree, spot.width, defaultSpot.width, base + "core width", ParserTask);
        JsonParser::encodeDecode(tree, spot.height, defaultSpot.height, base + "core height", ParserTask);
        JsonParser::encodeDecode(tree, spot.coreRadius, defaultSpot.coreRadius, base + "core radius", ParserTask);
        JsonParser::encodeDecode(tree, spot.fadeoutRadius, defaultSpot.fadeoutRadius, base + "fadeout radius", ParserTask);
        JsonParser::encodeDecode(tree, spot.values.friction, defaultSpot.values.friction, base + "friction", ParserTask);
        JsonParser::encodeDecode(tree, spot.values.rigidity, defaultSpot.values.rigidity, "rigidity", ParserTask);
        JsonParser::encodeDecode(tree, spot.values.radiationFactor, defaultSpot.values.radiationFactor, base + "radiation.factor", ParserTask);
        JsonParser::encodeDecode(tree, spot.values.cellMaxForce, defaultSpot.values.cellMaxForce, base + "cell.max force", ParserTask);
        JsonParser::encodeDecode(tree, spot.values.cellMinEnergy, defaultSpot.values.cellMinEnergy, base + "cell.min energy", ParserTask);

        JsonParser::encodeDecode(tree, spot.values.cellBindingForce, defaultSpot.values.cellBindingForce, base + "cell.binding force", ParserTask);
        JsonParser::encodeDecode(tree, spot.values.cellFusionVelocity, defaultSpot.values.cellFusionVelocity, base + "cell.fusion velocity", ParserTask);
        JsonParser::encodeDecode(tree, spot.values.cellMaxBindingEnergy, defaultSpot.values.cellMaxBindingEnergy, base + "cell.max binding energy", ParserTask);
        JsonParser::encodeDecode(tree, spot.values.cellMutationRate, defaultSpot.values.cellMutationRate, base + "cell.mutation rate", ParserTask);
        JsonParser::encodeDecode(tree, spot.values.tokenMutationRate, defaultSpot.values.tokenMutationRate, base + "token.mutation rate", ParserTask);
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionWeaponEnergyCost,
            defaultSpot.values.cellFunctionWeaponEnergyCost,
            base + "cell.function.weapon.energy cost",
            ParserTask);
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionWeaponColorPenalty,
            defaultSpot.values.cellFunctionWeaponColorPenalty,
            base + "cell.function.weapon.color penalty",
            ParserTask);
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionWeaponGeometryDeviationExponent,
            defaultSpot.values.cellFunctionWeaponGeometryDeviationExponent,
            base + "cell.function.weapon.geometry deviation exponent",
            ParserTask);
    }

    //flow field settings
    JsonParser::encodeDecode(tree, settings.flowFieldSettings.active, defaultSettings.flowFieldSettings.active, "flow field.active", ParserTask);
    JsonParser::encodeDecode(tree, settings.flowFieldSettings.numCenters, defaultSettings.flowFieldSettings.numCenters, "flow field.num centers", ParserTask);
    for (int i = 0; i < 2; ++i) {
        std::string node = "flow field.center" + std::to_string(i) + ".";
        auto& radialData = settings.flowFieldSettings.centers[i];
        auto& defaultRadialData = defaultSettings.flowFieldSettings.centers[i];
        JsonParser::encodeDecode(tree, radialData.posX, defaultRadialData.posX, node + "pos.x", ParserTask);
        JsonParser::encodeDecode(tree, radialData.posY, defaultRadialData.posY, node + "pos.y", ParserTask);
        JsonParser::encodeDecode(tree, radialData.radius, defaultRadialData.radius, node + "radius", ParserTask);
        JsonParser::encodeDecode(tree, radialData.strength, defaultRadialData.strength, node + "strength", ParserTask);
    }
}
