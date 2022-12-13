#include "SettingsParser.h"

#include "GeneralSettings.h"
#include "Settings.h"

boost::property_tree::ptree SettingsParser::encode(uint64_t timestep, Settings settings)
{
    boost::property_tree::ptree tree;
    encodeDecode(tree, timestep, settings, ParserTask::Encode);
    return tree;
}

std::pair<uint64_t, Settings> SettingsParser::decodeTimestepAndSettings(boost::property_tree::ptree tree)
{
    uint64_t timestep;
    Settings settings;
    encodeDecode(tree, timestep, settings, ParserTask::Decode);
    return std::make_pair(timestep, settings);
}

namespace
{
    std::unordered_map<SpotShape, std::string> shapeStringMap = {{SpotShape::Circular, "circular"}, {SpotShape::Rectangular, "rectangular"}};
    std::unordered_map<std::string, SpotShape> shapeEnumMap = {{"circular", SpotShape::Circular}, {"rectangular", SpotShape::Rectangular}};
}

void SettingsParser::encodeDecode(boost::property_tree::ptree& tree, uint64_t& timestep, Settings& settings, ParserTask parserTask)
{
    Settings defaultSettings;

    //general settings
    JsonParser::encodeDecode(tree, timestep, uint64_t(0), "general.time step", parserTask);
    JsonParser::encodeDecode(tree, settings.generalSettings.worldSizeX, defaultSettings.generalSettings.worldSizeX, "general.world size.x", parserTask);
    JsonParser::encodeDecode(tree, settings.generalSettings.worldSizeY, defaultSettings.generalSettings.worldSizeY, "general.world size.y", parserTask);

    //simulation parameters
    auto& simPar = settings.simulationParameters;
    auto& defaultPar = defaultSettings.simulationParameters;
    JsonParser::encodeDecode(tree, simPar.timestepSize, defaultPar.timestepSize, "simulation parameters.time step size", parserTask);
    JsonParser::encodeDecode(tree, simPar.spotValues.friction, defaultPar.spotValues.friction, "simulation parameters.friction", parserTask);
    JsonParser::encodeDecode(tree, simPar.spotValues.rigidity, defaultPar.spotValues.rigidity, "simulation parameters.rigidity", parserTask);
    JsonParser::encodeDecode(
        tree, simPar.spotValues.cellBindingForce, defaultPar.spotValues.cellBindingForce, "simulation parameters.cell.binding force", parserTask);
    JsonParser::encodeDecode(tree, simPar.cellMaxVelocity, defaultPar.cellMaxVelocity, "simulation parameters.cell.max velocity", parserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellMaxBindingDistance, defaultPar.cellMaxBindingDistance, "simulation parameters.cell.max binding distance", parserTask);
    JsonParser::encodeDecode(tree, simPar.cellRepulsionStrength, defaultPar.cellRepulsionStrength, "simulation parameters.cell.repulsion strength", parserTask);
    JsonParser::encodeDecode(tree, simPar.cellNormalEnergy, defaultPar.cellNormalEnergy, "simulation parameters.cell.normal energy", parserTask);

    JsonParser::encodeDecode(tree, simPar.cellMinDistance, defaultPar.cellMinDistance, "simulation parameters.cell.min distance", parserTask);
    JsonParser::encodeDecode(tree, simPar.cellMaxCollisionDistance, defaultPar.cellMaxCollisionDistance, "simulation parameters.cell.max distance", parserTask);
    JsonParser::encodeDecode(tree, simPar.spotValues.cellMaxForce, defaultPar.spotValues.cellMaxForce, "simulation parameters.cell.max force", parserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellMaxForceDecayProb, defaultPar.cellMaxForceDecayProb, "simulation parameters.cell.max force decay probability", parserTask);
    JsonParser::encodeDecode(tree, simPar.cellMaxBonds, defaultPar.cellMaxBonds, "simulation parameters.cell.max bonds", parserTask);
    JsonParser::encodeDecode(
        tree, simPar.cellMaxExecutionOrderNumbers, defaultPar.cellMaxExecutionOrderNumbers, "simulation parameters.cell.max execution order number", parserTask);
    JsonParser::encodeDecode(tree, simPar.spotValues.cellMinEnergy, defaultPar.spotValues.cellMinEnergy, "simulation parameters.cell.min energy", parserTask);
    JsonParser::encodeDecode(
        tree, simPar.spotValues.cellFusionVelocity, defaultPar.spotValues.cellFusionVelocity, "simulation parameters.cell.fusion velocity", parserTask);
    JsonParser::encodeDecode(
        tree, simPar.spotValues.cellMaxBindingEnergy, simPar.spotValues.cellMaxBindingEnergy, "simulation parameters.cell.max binding energy", parserTask);
    for (int i = 0; i < 7; ++i) {
        JsonParser::encodeDecode(
            tree,
            simPar.spotValues.cellColorTransitionDuration[i],
            defaultPar.spotValues.cellColorTransitionDuration[i],
            "simulation parameters.cell.color transition rules.duration[" + std::to_string(i) + "]",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            simPar.spotValues.cellColorTransitionTargetColor[i],
            defaultPar.spotValues.cellColorTransitionTargetColor[i],
            "simulation parameters.cell.color transition rules.target color[" + std::to_string(i) + "]",
            parserTask);
    }
    JsonParser::encodeDecode(
        tree, simPar.spotValues.radiationFactor, defaultPar.spotValues.radiationFactor, "simulation parameters.radiation.factor", parserTask);
    JsonParser::encodeDecode(tree, simPar.radiationProb, defaultPar.radiationProb, "simulation parameters.radiation.probability", parserTask);
    JsonParser::encodeDecode(
        tree, simPar.radiationVelocityMultiplier, defaultPar.radiationVelocityMultiplier, "simulation parameters.radiation.velocity multiplier", parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.radiationVelocityPerturbation,
        defaultPar.radiationVelocityPerturbation,
        "simulation parameters.radiation.velocity perturbation",
        parserTask);
    for (int i = 0; i < 7; ++i) {
        JsonParser::encodeDecode(
            tree,
            simPar.radiationAbsorptionByCellColor[i],
            defaultPar.radiationAbsorptionByCellColor[i],
            "simulation parameters.radiation.absorption by cell color[" + std::to_string(i) + "]",
            parserTask);
    }
    JsonParser::encodeDecode(
        tree, simPar.radiationMinCellEnergy, defaultPar.radiationMinCellEnergy,
        "simulation parameters.radiation.min cell energy",
        parserTask);
    JsonParser::encodeDecode(tree, simPar.radiationMinCellAge, defaultPar.radiationMinCellAge, "simulation parameters.radiation.min cell age", parserTask);

    JsonParser::encodeDecode(tree, simPar.clusterDecay, defaultPar.clusterDecay, "simulation parameters.cluster.decay", parserTask);
    JsonParser::encodeDecode(tree, simPar.clusterDecayProb, defaultPar.clusterDecayProb, "simulation parameters.cluster.decay probability", parserTask);

    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionConstructionInheritColor,
        defaultPar.cellFunctionConstructionInheritColor,
        "simulation parameters.cell.function.constructor.inherit color",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionConstructorOffspringDistance,
        defaultPar.cellFunctionConstructorOffspringDistance,
        "simulation parameters.cell.function.constructor.offspring distance",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionConstructorConnectingCellMaxDistance,
        defaultPar.cellFunctionConstructorConnectingCellMaxDistance,
        "simulation parameters.cell.function.constructor.connecting cell max distance",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionConstructorActivityThreshold,
        defaultPar.cellFunctionConstructorActivityThreshold,
        "simulation parameters.cell.function.constructor.activity threshold",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionConstructorMutationNeuronDataProbability,
        defaultPar.spotValues.cellFunctionConstructorMutationNeuronDataProbability,
        "simulation parameters.cell.function.constructor.mutation probability.neuron data",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionConstructorMutationDataProbability,
        defaultPar.spotValues.cellFunctionConstructorMutationDataProbability,
        "simulation parameters.cell.function.constructor.mutation probability.data",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionConstructorMutationCellFunctionProbability,
        defaultPar.spotValues.cellFunctionConstructorMutationCellFunctionProbability,
        "simulation parameters.cell.function.constructor.mutation probability.cell function",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionConstructorMutationInsertionProbability,
        defaultPar.spotValues.cellFunctionConstructorMutationInsertionProbability,
        "simulation parameters.cell.function.constructor.mutation probability.insertion",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionConstructorMutationDeletionProbability,
        defaultPar.spotValues.cellFunctionConstructorMutationDeletionProbability,
        "simulation parameters.cell.function.constructor.mutation probability.deletion",
        parserTask);


    JsonParser::encodeDecode(
        tree, simPar.cellFunctionAttackerRadius, defaultPar.cellFunctionAttackerRadius,
        "simulation parameters.cell.function.attacker.radius",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionAttackerStrength,
        defaultPar.cellFunctionAttackerStrength,
        "simulation parameters.cell.function.attacker.strength",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionAttackerEnergyDistributionRadius,
        defaultPar.cellFunctionAttackerEnergyDistributionRadius,
        "simulation parameters.cell.function.attacker.energy distribution radius",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionAttackerEnergyDistributionSameColor,
        defaultPar.cellFunctionAttackerEnergyDistributionSameColor,
        "simulation parameters.cell.function.attacker.energy distribution same color",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionAttackerEnergyDistributionValue,
        defaultPar.cellFunctionAttackerEnergyDistributionValue,
        "simulation parameters.cell.function.attacker.energy distribution value",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionAttackerInhomogeneityBonusFactor,
        defaultPar.cellFunctionAttackerInhomogeneityBonusFactor,
        "simulation parameters.cell.function.attacker.inhomogeneity bonus",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionAttackerActivityThreshold,
        defaultPar.cellFunctionAttackerActivityThreshold,
        "simulation parameters.cell.function.attacker.activity threshold",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionAttackerEnergyCost,
        defaultPar.spotValues.cellFunctionAttackerEnergyCost,
        "simulation parameters.cell.function.attacker.energy cost",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionAttackerGeometryDeviationExponent,
        defaultPar.spotValues.cellFunctionAttackerGeometryDeviationExponent,
        "simulation parameters.cell.function.attacker.geometry deviation exponent",
        parserTask);
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            JsonParser::encodeDecode(
                tree,
                simPar.spotValues.cellFunctionAttackerFoodChainColorMatrix[i][j],
                defaultPar.spotValues.cellFunctionAttackerFoodChainColorMatrix[i][j],
                "simulation parameters.cell.function.attacker.food chain color matrix[" + std::to_string(i) + ", " + std::to_string(j) + "]",
                parserTask);
        }
    }
    JsonParser::encodeDecode(
        tree,
        simPar.spotValues.cellFunctionAttackerConnectionsMismatchPenalty,
        defaultPar.spotValues.cellFunctionAttackerConnectionsMismatchPenalty,
        "simulation parameters.cell.function.attacker.connections mismatch penalty",
        parserTask);

    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionTransmitterEnergyDistributionSameColor,
        defaultPar.cellFunctionTransmitterEnergyDistributionSameColor,
        "simulation parameters.cell.function.transmitter.energy distribution same color",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionTransmitterEnergyDistributionRadius,
        defaultPar.cellFunctionTransmitterEnergyDistributionRadius,
        "simulation parameters.cell.function.transmitter.energy distribution radius",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionTransmitterEnergyDistributionValue,
        defaultPar.cellFunctionTransmitterEnergyDistributionValue,
        "simulation parameters.cell.function.transmitter.energy distribution value",
        parserTask);

    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionMuscleContractionExpansionDelta,
        defaultPar.cellFunctionMuscleContractionExpansionDelta,
        "simulation parameters.cell.function.muscle.contraction expansion delta",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionMuscleMovementDelta,
        defaultPar.cellFunctionMuscleMovementDelta,
        "simulation parameters.cell.function.muscle.movement delta",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionMuscleBendingAngle,
        defaultPar.cellFunctionMuscleBendingAngle,
        "simulation parameters.cell.function.muscle.bending angle",
        parserTask);

    JsonParser::encodeDecode(
        tree,
        simPar.particleTransformationAllowed,
        defaultPar.particleTransformationAllowed,
        "simulation parameters.particle.transformation allowed",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.particleTransformationRandomCellFunction,
        defaultPar.particleTransformationRandomCellFunction,
        "simulation parameters.particle.transformation.random cell function",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.particleTransformationMaxGenomeSize,
        defaultPar.particleTransformationMaxGenomeSize,
        "simulation parameters.particle.transformation.max genome size",
        parserTask);

    JsonParser::encodeDecode(
        tree, simPar.cellFunctionSensorRange, defaultPar.cellFunctionSensorRange, "simulation parameters.cell.function.sensor.range", parserTask);
    JsonParser::encodeDecode(
        tree,
        simPar.cellFunctionSensorActivityThreshold,
        defaultPar.cellFunctionSensorActivityThreshold,
        "simulation parameters.cell.function.sensor.activity threshold",
        parserTask);

    //spots
    auto& spots = settings.simulationParametersSpots;
    auto& defaultSpots = defaultSettings.simulationParametersSpots;
    JsonParser::encodeDecode(tree, spots.numSpots, defaultSpots.numSpots, "simulation parameters.spots.num spots", parserTask);
    for (int index = 0; index <= 1; ++index) {
        std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
        auto& spot = spots.spots[index];
        auto& defaultSpot = defaultSpots.spots[index];
        JsonParser::encodeDecode(tree, spot.color, defaultSpot.color, base + "color", parserTask);
        JsonParser::encodeDecode(tree, spot.posX, defaultSpot.posX, base + "pos.x", parserTask);
        JsonParser::encodeDecode(tree, spot.posY, defaultSpot.posY, base + "pos.y", parserTask);

        auto shapeString = shapeStringMap.at(spot.shape);
        JsonParser::encodeDecode(tree, shapeString, shapeStringMap.at(defaultSpot.shape), base + "shape", parserTask);
        spot.shape = shapeEnumMap.at(shapeString);

        JsonParser::encodeDecode(tree, spot.width, defaultSpot.width, base + "core width", parserTask);
        JsonParser::encodeDecode(tree, spot.height, defaultSpot.height, base + "core height", parserTask);
        JsonParser::encodeDecode(tree, spot.coreRadius, defaultSpot.coreRadius, base + "core radius", parserTask);
        JsonParser::encodeDecode(tree, spot.fadeoutRadius, defaultSpot.fadeoutRadius, base + "fadeout radius", parserTask);
        JsonParser::encodeDecode(tree, spot.values.friction, defaultSpot.values.friction, base + "friction", parserTask);
        JsonParser::encodeDecode(tree, spot.values.rigidity, defaultSpot.values.rigidity, base + "rigidity", parserTask);
        JsonParser::encodeDecode(tree, spot.values.radiationFactor, defaultSpot.values.radiationFactor, base + "radiation.factor", parserTask);
        JsonParser::encodeDecode(tree, spot.values.cellMaxForce, defaultSpot.values.cellMaxForce, base + "cell.max force", parserTask);
        JsonParser::encodeDecode(tree, spot.values.cellMinEnergy, defaultSpot.values.cellMinEnergy, base + "cell.min energy", parserTask);

        JsonParser::encodeDecode(tree, spot.values.cellBindingForce, defaultSpot.values.cellBindingForce, base + "cell.binding force", parserTask);
        JsonParser::encodeDecode(tree, spot.values.cellFusionVelocity, defaultSpot.values.cellFusionVelocity, base + "cell.fusion velocity", parserTask);
        JsonParser::encodeDecode(tree, spot.values.cellMaxBindingEnergy, defaultSpot.values.cellMaxBindingEnergy, base + "cell.max binding energy", parserTask);

        for (int i = 0; i < 7; ++i) {
            JsonParser::encodeDecode(
                tree,
                spot.values.cellColorTransitionDuration[i],
                defaultSpot.values.cellColorTransitionDuration[i],
                base + "cell.color transition rules.duration[" + std::to_string(i) + "]",
                parserTask);
            JsonParser::encodeDecode(
                tree,
                spot.values.cellColorTransitionTargetColor[i],
                defaultSpot.values.cellColorTransitionTargetColor[i],
                base + "cell.color transition rules.target color[" + std::to_string(i) + "]",
                parserTask);
        }

        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionAttackerEnergyCost,
            defaultSpot.values.cellFunctionAttackerEnergyCost,
            base + "cell.function.attacker.energy cost",
            parserTask);
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                JsonParser::encodeDecode(
                    tree,
                    spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j],
                    defaultSpot.values.cellFunctionAttackerFoodChainColorMatrix[i][j],
                    base + "function.attacker.color matrix[" + std::to_string(i) + ", " + std::to_string(j) + "]",
                    parserTask);
            }
        }
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionAttackerGeometryDeviationExponent,
            defaultSpot.values.cellFunctionAttackerGeometryDeviationExponent,
            base + "cell.function.attacker.geometry deviation exponent",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionAttackerConnectionsMismatchPenalty,
            defaultSpot.values.cellFunctionAttackerConnectionsMismatchPenalty,
            base + "cell.function.attacker.connections mismatch penalty",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionConstructorMutationNeuronDataProbability,
            defaultSpot.values.cellFunctionConstructorMutationNeuronDataProbability,
            base + "cell.function.constructor.mutation probability.neuron data",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionConstructorMutationDataProbability,
            defaultSpot.values.cellFunctionConstructorMutationDataProbability,
            base + " cell.function.constructor.mutation probability.data ",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionConstructorMutationCellFunctionProbability,
            defaultSpot.values.cellFunctionConstructorMutationCellFunctionProbability,
            base + "cell.function.constructor.mutation probability.cell function",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionConstructorMutationInsertionProbability,
            defaultSpot.values.cellFunctionConstructorMutationInsertionProbability,
            base + "cell.function.constructor.mutation probability.insertion",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            spot.values.cellFunctionConstructorMutationDeletionProbability,
            defaultSpot.values.cellFunctionConstructorMutationDeletionProbability,
            base + "cell.function.constructor.mutation probability.deletion",
            parserTask);
    }

    //flow field settings
    JsonParser::encodeDecode(tree, settings.flowFieldSettings.active, defaultSettings.flowFieldSettings.active, "flow field.active", parserTask);
    JsonParser::encodeDecode(tree, settings.flowFieldSettings.numCenters, defaultSettings.flowFieldSettings.numCenters, "flow field.num centers", parserTask);
    for (int i = 0; i < 2; ++i) {
        std::string node = "flow field.center" + std::to_string(i) + ".";
        auto& radialData = settings.flowFieldSettings.centers[i];
        auto& defaultRadialData = defaultSettings.flowFieldSettings.centers[i];
        JsonParser::encodeDecode(tree, radialData.posX, defaultRadialData.posX, node + "pos.x", parserTask);
        JsonParser::encodeDecode(tree, radialData.posY, defaultRadialData.posY, node + "pos.y", parserTask);
        JsonParser::encodeDecode(tree, radialData.radius, defaultRadialData.radius, node + "radius", parserTask);
        JsonParser::encodeDecode(tree, radialData.strength, defaultRadialData.strength, node + "strength", parserTask);
    }
}
