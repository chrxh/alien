#include "AuxiliaryDataParser.h"

#include "GeneralSettings.h"
#include "Settings.h"

boost::property_tree::ptree AuxiliaryDataParser::encodeAuxiliaryData(AuxiliaryData const& data)
{
    boost::property_tree::ptree tree;
    encodeDecode(tree, const_cast<AuxiliaryData&>(data), ParserTask::Encode);
    return tree;
}

AuxiliaryData AuxiliaryDataParser::decodeAuxiliaryData(boost::property_tree::ptree tree)
{
    AuxiliaryData result;
    encodeDecode(tree, result, ParserTask::Decode);
    return result;
}

boost::property_tree::ptree AuxiliaryDataParser::encodeSimulationParameters(SimulationParameters const& data)
{
    boost::property_tree::ptree tree;
    encodeDecode(tree, const_cast<SimulationParameters&>(data), ParserTask::Encode);
    return tree;
}

SimulationParameters AuxiliaryDataParser::decodeSimulationParameters(boost::property_tree::ptree tree)
{
    SimulationParameters result;
    encodeDecode(tree, result, ParserTask::Decode);
    return result;
}

void AuxiliaryDataParser::encodeDecode(boost::property_tree::ptree& tree, AuxiliaryData& data, ParserTask parserTask)
{
    AuxiliaryData defaultSettings;

    //general settings
    JsonParser::encodeDecode(tree, data.timestep, uint64_t(0), "general.time step", parserTask);
    JsonParser::encodeDecode(tree, data.zoom, 4.0f, "general.zoom", parserTask);
    JsonParser::encodeDecode(tree, data.center.x, 0.0f, "general.center.x", parserTask);
    JsonParser::encodeDecode(tree, data.center.y, 0.0f, "general.center.y", parserTask);
    JsonParser::encodeDecode(tree, data.generalSettings.worldSizeX, defaultSettings.generalSettings.worldSizeX, "general.world size.x", parserTask);
    JsonParser::encodeDecode(tree, data.generalSettings.worldSizeY, defaultSettings.generalSettings.worldSizeY, "general.world size.y", parserTask);

    encodeDecode(tree, data.simulationParameters, parserTask);
}

void AuxiliaryDataParser::encodeDecode(boost::property_tree::ptree& tree, SimulationParameters& parameters, ParserTask parserTask)
{
    //simulation parameters
    SimulationParameters defaultParameters;
    JsonParser::encodeDecode(tree, parameters.backgroundColor, defaultParameters.backgroundColor, "simulation parameters.background color", parserTask);
    JsonParser::encodeDecode(tree, parameters.timestepSize, defaultParameters.timestepSize, "simulation parameters.time step size", parserTask);

    JsonParser::encodeDecode(tree, parameters.motionType, defaultParameters.motionType, "simulation parameters.motion.type", parserTask);
    if (parameters.motionType == MotionType_Fluid) {
        JsonParser::encodeDecode(
            tree,
            parameters.motionData.fluidMotion.smoothingLength,
            defaultParameters.motionData.fluidMotion.smoothingLength,
            "simulation parameters.fluid.smoothing length",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            parameters.motionData.fluidMotion.pressureStrength,
            defaultParameters.motionData.fluidMotion.pressureStrength,
            "simulation parameters.fluid.pressure strength",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            parameters.motionData.fluidMotion.viscosityStrength,
            defaultParameters.motionData.fluidMotion.viscosityStrength,
            "simulation parameters.fluid.viscosity strength",
            parserTask);
    } else {
        JsonParser::encodeDecode(
            tree,
            parameters.motionData.collisionMotion.cellMaxCollisionDistance,
            defaultParameters.motionData.collisionMotion.cellMaxCollisionDistance,
            "simulation parameters.motion.collision.max distance",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            parameters.motionData.collisionMotion.cellRepulsionStrength,
            defaultParameters.motionData.collisionMotion.cellRepulsionStrength,
            "simulation parameters.motion.collision.repulsion strength",
            parserTask);
    }

    JsonParser::encodeDecode(tree, parameters.baseValues.friction, defaultParameters.baseValues.friction, "simulation parameters.friction", parserTask);
    JsonParser::encodeDecode(tree, parameters.baseValues.rigidity, defaultParameters.baseValues.rigidity, "simulation parameters.rigidity", parserTask);
    JsonParser::encodeDecode(tree, parameters.cellMaxVelocity, defaultParameters.cellMaxVelocity, "simulation parameters.cell.max velocity", parserTask);
    JsonParser::encodeDecode(
        tree, parameters.cellMaxBindingDistance, defaultParameters.cellMaxBindingDistance, "simulation parameters.cell.max binding distance", parserTask);
    JsonParser::encodeDecode(tree, parameters.cellNormalEnergy, defaultParameters.cellNormalEnergy, "simulation parameters.cell.normal energy", parserTask);

    JsonParser::encodeDecode(tree, parameters.cellMinDistance, defaultParameters.cellMinDistance, "simulation parameters.cell.min distance", parserTask);
    JsonParser::encodeDecode(tree, parameters.baseValues.cellMaxForce, defaultParameters.baseValues.cellMaxForce, "simulation parameters.cell.max force", parserTask);
    JsonParser::encodeDecode(
        tree, parameters.cellMaxForceDecayProb, defaultParameters.cellMaxForceDecayProb, "simulation parameters.cell.max force decay probability", parserTask);
    JsonParser::encodeDecode(
        tree, parameters.cellNumExecutionOrderNumbers, defaultParameters.cellNumExecutionOrderNumbers, "simulation parameters.cell.max execution order number", parserTask);
    JsonParser::encodeDecode(tree, parameters.baseValues.cellMinEnergy, defaultParameters.baseValues.cellMinEnergy, "simulation parameters.cell.min energy", parserTask);
    JsonParser::encodeDecode(
        tree, parameters.baseValues.cellFusionVelocity, defaultParameters.baseValues.cellFusionVelocity, "simulation parameters.cell.fusion velocity", parserTask);
    JsonParser::encodeDecode(
        tree, parameters.baseValues.cellMaxBindingEnergy, parameters.baseValues.cellMaxBindingEnergy, "simulation parameters.cell.max binding energy", parserTask);
    for (int i = 0; i < MAX_COLORS; ++i) {
        JsonParser::encodeDecode(
            tree,
            parameters.baseValues.cellColorTransitionDuration[i],
            defaultParameters.baseValues.cellColorTransitionDuration[i],
            "simulation parameters.cell.color transition rules.duration[" + std::to_string(i) + "]",
            parserTask);
        JsonParser::encodeDecode(
            tree,
            parameters.baseValues.cellColorTransitionTargetColor[i],
            defaultParameters.baseValues.cellColorTransitionTargetColor[i],
            "simulation parameters.cell.color transition rules.target color[" + std::to_string(i) + "]",
            parserTask);
    }
    JsonParser::encodeDecode(
        tree, parameters.baseValues.radiationFactor, defaultParameters.baseValues.radiationFactor, "simulation parameters.radiation.factor", parserTask);
    JsonParser::encodeDecode(tree, parameters.radiationProb, defaultParameters.radiationProb, "simulation parameters.radiation.probability", parserTask);
    JsonParser::encodeDecode(
        tree, parameters.radiationVelocityMultiplier, defaultParameters.radiationVelocityMultiplier, "simulation parameters.radiation.velocity multiplier", parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.radiationVelocityPerturbation,
        defaultParameters.radiationVelocityPerturbation,
        "simulation parameters.radiation.velocity perturbation",
        parserTask);
    for (int i = 0; i < MAX_COLORS; ++i) {
        JsonParser::encodeDecode(
            tree,
            parameters.radiationAbsorptionByCellColor[i],
            defaultParameters.radiationAbsorptionByCellColor[i],
            "simulation parameters.radiation.absorption by cell color[" + std::to_string(i) + "]",
            parserTask);
    }
    JsonParser::encodeDecode(
        tree, parameters.highRadiationMinCellEnergy, defaultParameters.highRadiationMinCellEnergy,
        "simulation parameters.high radiation.min cell energy",
        parserTask);
    JsonParser::encodeDecode(
        tree, parameters.highRadiationFactor, defaultParameters.highRadiationFactor,
        "simulation parameters.high radiation.factor",
        parserTask);
    encodeDecodeColorDependentProperty(
        tree,
        parameters.radiationMinCellAgeByCellColor,
        parameters.radiationMinCellAgeColorDependent,
        defaultParameters.radiationMinCellAgeByCellColor,
        "simulation parameters.radiation.min cell age",
        parserTask);

    JsonParser::encodeDecode(tree, parameters.clusterDecay, defaultParameters.clusterDecay, "simulation parameters.cluster.decay", parserTask);
    JsonParser::encodeDecode(tree, parameters.clusterDecayProb, defaultParameters.clusterDecayProb, "simulation parameters.cluster.decay probability", parserTask);

    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionConstructorOffspringDistance,
        defaultParameters.cellFunctionConstructorOffspringDistance,
        "simulation parameters.cell.function.constructor.offspring distance",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionConstructorConnectingCellMaxDistance,
        defaultParameters.cellFunctionConstructorConnectingCellMaxDistance,
        "simulation parameters.cell.function.constructor.connecting cell max distance",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionConstructorActivityThreshold,
        defaultParameters.cellFunctionConstructorActivityThreshold,
        "simulation parameters.cell.function.constructor.activity threshold",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability,
        defaultParameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability,
        "simulation parameters.cell.function.constructor.mutation probability.neuron data",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionConstructorMutationDataProbability,
        defaultParameters.baseValues.cellFunctionConstructorMutationDataProbability,
        "simulation parameters.cell.function.constructor.mutation probability.data",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability,
        defaultParameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability,
        "simulation parameters.cell.function.constructor.mutation probability.cell function",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionConstructorMutationInsertionProbability,
        defaultParameters.baseValues.cellFunctionConstructorMutationInsertionProbability,
        "simulation parameters.cell.function.constructor.mutation probability.insertion",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionConstructorMutationDeletionProbability,
        defaultParameters.baseValues.cellFunctionConstructorMutationDeletionProbability,
        "simulation parameters.cell.function.constructor.mutation probability.deletion",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionConstructorMutationTranslationProbability,
        defaultParameters.baseValues.cellFunctionConstructorMutationTranslationProbability,
        "simulation parameters.cell.function.constructor.mutation probability.translation",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionConstructorMutationDuplicationProbability,
        defaultParameters.baseValues.cellFunctionConstructorMutationDuplicationProbability,
        "simulation parameters.cell.function.constructor.mutation probability.duplication",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionConstructorMutationColor,
        defaultParameters.cellFunctionConstructorMutationColor,
        "simulation parameters.cell.function.constructor.mutation color",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionConstructorMutationSelfReplication,
        defaultParameters.cellFunctionConstructorMutationSelfReplication,
        "simulation parameters.cell.function.constructor.mutation self replication",
        parserTask);

    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionInjectorRadius,
        defaultParameters.cellFunctionInjectorRadius,
        "simulation parameters.cell.function.injector.radius",
        parserTask);
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            JsonParser::encodeDecode(
                tree,
                parameters.cellFunctionInjectorDurationColorMatrix[i][j],
                defaultParameters.cellFunctionInjectorDurationColorMatrix[i][j],
                "simulation parameters.cell.function.injector.duration[" + std::to_string(i) + ", " + std::to_string(j) + "]",
                parserTask);
        }
    }

    JsonParser::encodeDecode(
        tree, parameters.cellFunctionAttackerRadius, defaultParameters.cellFunctionAttackerRadius,
        "simulation parameters.cell.function.attacker.radius",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionAttackerStrength,
        defaultParameters.cellFunctionAttackerStrength,
        "simulation parameters.cell.function.attacker.strength",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionAttackerEnergyDistributionRadius,
        defaultParameters.cellFunctionAttackerEnergyDistributionRadius,
        "simulation parameters.cell.function.attacker.energy distribution radius",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionAttackerEnergyDistributionSameColor,
        defaultParameters.cellFunctionAttackerEnergyDistributionSameColor,
        "simulation parameters.cell.function.attacker.energy distribution same color",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionAttackerEnergyDistributionValue,
        defaultParameters.cellFunctionAttackerEnergyDistributionValue,
        "simulation parameters.cell.function.attacker.energy distribution value",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionAttackerColorInhomogeneityFactor,
        defaultParameters.cellFunctionAttackerColorInhomogeneityFactor,
        "simulation parameters.cell.function.attacker.color inhomogeneity factor",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionAttackerActivityThreshold,
        defaultParameters.cellFunctionAttackerActivityThreshold,
        "simulation parameters.cell.function.attacker.activity threshold",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionAttackerEnergyCost,
        defaultParameters.baseValues.cellFunctionAttackerEnergyCost,
        "simulation parameters.cell.function.attacker.energy cost",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionAttackerVelocityPenalty,
        defaultParameters.cellFunctionAttackerVelocityPenalty,
        "simulation parameters.cell.function.attacker.velocity penalty",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent,
        defaultParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent,
        "simulation parameters.cell.function.attacker.geometry deviation exponent",
        parserTask);
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            JsonParser::encodeDecode(
                tree,
                parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j],
                defaultParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j],
                "simulation parameters.cell.function.attacker.food chain color matrix[" + std::to_string(i) + ", " + std::to_string(j) + "]",
                parserTask);
        }
    }
    JsonParser::encodeDecode(
        tree,
        parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty,
        defaultParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty,
        "simulation parameters.cell.function.attacker.connections mismatch penalty",
        parserTask);

    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionDefenderAgainstAttackerStrength,
        defaultParameters.cellFunctionDefenderAgainstAttackerStrength,
        "simulation parameters.cell.function.defender.against attacker strength",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionDefenderAgainstInjectorStrength,
        defaultParameters.cellFunctionDefenderAgainstInjectorStrength,
        "simulation parameters.cell.function.defender.against injector strength",
        parserTask);

    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionTransmitterEnergyDistributionSameColor,
        defaultParameters.cellFunctionTransmitterEnergyDistributionSameColor,
        "simulation parameters.cell.function.transmitter.energy distribution same color",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionTransmitterEnergyDistributionRadius,
        defaultParameters.cellFunctionTransmitterEnergyDistributionRadius,
        "simulation parameters.cell.function.transmitter.energy distribution radius",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionTransmitterEnergyDistributionValue,
        defaultParameters.cellFunctionTransmitterEnergyDistributionValue,
        "simulation parameters.cell.function.transmitter.energy distribution value",
        parserTask);

    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionMuscleContractionExpansionDelta,
        defaultParameters.cellFunctionMuscleContractionExpansionDelta,
        "simulation parameters.cell.function.muscle.contraction expansion delta",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionMuscleMovementAcceleration,
        defaultParameters.cellFunctionMuscleMovementAcceleration,
        "simulation parameters.cell.function.muscle.movement acceleration",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionMuscleBendingAngle,
        defaultParameters.cellFunctionMuscleBendingAngle,
        "simulation parameters.cell.function.muscle.bending angle",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionMuscleBendingAcceleration,
        defaultParameters.cellFunctionMuscleBendingAcceleration,
        "simulation parameters.cell.function.muscle.bending acceleration",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionMuscleBendingAccelerationThreshold,
        defaultParameters.cellFunctionMuscleBendingAccelerationThreshold,
        "simulation parameters.cell.function.muscle.bending acceleration threshold",
        parserTask);

    JsonParser::encodeDecode(
        tree,
        parameters.particleTransformationAllowed,
        defaultParameters.particleTransformationAllowed,
        "simulation parameters.particle.transformation allowed",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.particleTransformationRandomCellFunction,
        defaultParameters.particleTransformationRandomCellFunction,
        "simulation parameters.particle.transformation.random cell function",
        parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.particleTransformationMaxGenomeSize,
        defaultParameters.particleTransformationMaxGenomeSize,
        "simulation parameters.particle.transformation.max genome size",
        parserTask);

    JsonParser::encodeDecode(
        tree, parameters.cellFunctionSensorRange, defaultParameters.cellFunctionSensorRange, "simulation parameters.cell.function.sensor.range", parserTask);
    JsonParser::encodeDecode(
        tree,
        parameters.cellFunctionSensorActivityThreshold,
        defaultParameters.cellFunctionSensorActivityThreshold,
        "simulation parameters.cell.function.sensor.activity threshold",
        parserTask);

    //particle sources
    JsonParser::encodeDecode(tree, parameters.numParticleSources, defaultParameters.numParticleSources, "simulation parameters.particle sources.num sources", parserTask);
    for (int index = 0; index < parameters.numParticleSources; ++index) {
        std::string base = "simulation parameters.particle sources." + std::to_string(index) + ".";
        auto& source = parameters.particleSources[index];
        auto& defaultSource = defaultParameters.particleSources[index];
        JsonParser::encodeDecode(tree, source.posX, defaultSource.posX, base + "pos.x", parserTask);
        JsonParser::encodeDecode(tree, source.posY, defaultSource.posY, base + "pos.y", parserTask);
    }

    //spots
    JsonParser::encodeDecode(tree, parameters.numSpots, defaultParameters.numSpots, "simulation parameters.spots.num spots", parserTask);
    for (int index = 0; index < parameters.numSpots; ++index) {
        std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
        auto& spot = parameters.spots[index];
        auto& defaultSpot = defaultParameters.spots[index];
        JsonParser::encodeDecode(tree, spot.color, defaultSpot.color, base + "color", parserTask);
        JsonParser::encodeDecode(tree, spot.posX, defaultSpot.posX, base + "pos.x", parserTask);
        JsonParser::encodeDecode(tree, spot.posY, defaultSpot.posY, base + "pos.y", parserTask);

        JsonParser::encodeDecode(tree, spot.shapeType, defaultSpot.shapeType, base + "shape.type", parserTask);
        if (spot.shapeType == ShapeType_Circular) {
            JsonParser::encodeDecode(
                tree, spot.shapeData.circularSpot.coreRadius, defaultSpot.shapeData.circularSpot.coreRadius, base + "shape.circular.core radius", parserTask);
        }
        if (spot.shapeType == ShapeType_Rectangular) {
            JsonParser::encodeDecode(tree, spot.shapeData.rectangularSpot.width, defaultSpot.shapeData.rectangularSpot.width, base + "shape.rectangular.core width", parserTask);
            JsonParser::encodeDecode(
                tree, spot.shapeData.rectangularSpot.height, defaultSpot.shapeData.rectangularSpot.height, base + "shape.rectangular.core height", parserTask);
        }
        JsonParser::encodeDecode(tree, spot.flowType, defaultSpot.flowType, base + "flow.type", parserTask);
        if (spot.flowType == FlowType_Radial) {
            JsonParser::encodeDecode(
                tree, spot.flowData.radialFlow.orientation, defaultSpot.flowData.radialFlow.orientation, base + "flow.radial.orientation", parserTask);
            JsonParser::encodeDecode(
                tree, spot.flowData.radialFlow.strength, defaultSpot.flowData.radialFlow.strength, base + "flow.radial.strength", parserTask);
            JsonParser::encodeDecode(
                tree, spot.flowData.radialFlow.driftAngle, defaultSpot.flowData.radialFlow.driftAngle, base + "flow.radial.drift angle", parserTask);
        }
        if (spot.flowType == FlowType_Central) {
            JsonParser::encodeDecode(
                tree, spot.flowData.centralFlow.strength, defaultSpot.flowData.centralFlow.strength, base + "flow.central.strength", parserTask);
        }
        if (spot.flowType == FlowType_Linear) {
            JsonParser::encodeDecode(tree, spot.flowData.linearFlow.angle, defaultSpot.flowData.linearFlow.angle, base + "flow.linear.angle", parserTask);
            JsonParser::encodeDecode(
                tree, spot.flowData.linearFlow.strength, defaultSpot.flowData.linearFlow.strength, base + "flow.linear.strength", parserTask);
        }
        JsonParser::encodeDecode(tree, spot.fadeoutRadius, defaultSpot.fadeoutRadius, base + "fadeout radius", parserTask);

        encodeDecodeSpotProperty(tree, spot.values.friction, spot.activatedValues.friction, defaultSpot.values.friction, base + "friction", parserTask);
        encodeDecodeSpotProperty(tree, spot.values.rigidity, spot.activatedValues.rigidity, defaultSpot.values.rigidity, base + "rigidity", parserTask);
        encodeDecodeSpotProperty(
            tree, spot.values.radiationFactor, spot.activatedValues.radiationFactor, defaultSpot.values.radiationFactor, base + "radiation.factor", parserTask);
        encodeDecodeSpotProperty(
            tree, spot.values.cellMaxForce, spot.activatedValues.cellMaxForce, defaultSpot.values.cellMaxForce, base + "cell.max force", parserTask);
        encodeDecodeSpotProperty(
            tree, spot.values.cellMinEnergy, spot.activatedValues.cellMinEnergy, defaultSpot.values.cellMinEnergy, base + "cell.min energy", parserTask);

        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFusionVelocity,
            spot.activatedValues.cellFusionVelocity,
            defaultSpot.values.cellFusionVelocity,
            base + "cell.fusion velocity",
            parserTask);
        encodeDecodeSpotProperty(
            tree,
            spot.values.cellMaxBindingEnergy,
            spot.activatedValues.cellMaxBindingEnergy,
            defaultSpot.values.cellMaxBindingEnergy,
            base + "cell.max binding energy",
            parserTask);

        JsonParser::encodeDecode(tree, spot.activatedValues.cellColorTransition, false, base + "cell.color transition rules.activated", parserTask);
        for (int i = 0; i < MAX_COLORS; ++i) {
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

        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionAttackerEnergyCost,
            spot.activatedValues.cellFunctionAttackerEnergyCost,
            defaultSpot.values.cellFunctionAttackerEnergyCost,
            base + "cell.function.attacker.energy cost",
            parserTask);
        JsonParser::encodeDecode(
            tree, spot.activatedValues.cellFunctionAttackerFoodChainColorMatrix, false, base + "cell.function.attacker.food chain color matrix.activated", parserTask);
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                JsonParser::encodeDecode(
                    tree,
                    spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j],
                    defaultSpot.values.cellFunctionAttackerFoodChainColorMatrix[i][j],
                    base + "cell.function.attacker.food chain color matrix[" + std::to_string(i) + ", " + std::to_string(j) + "]",
                    parserTask);
            }
        }
        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionAttackerGeometryDeviationExponent,
            spot.activatedValues.cellFunctionAttackerGeometryDeviationExponent,
            defaultSpot.values.cellFunctionAttackerGeometryDeviationExponent,
            base + "cell.function.attacker.geometry deviation exponent",
            parserTask);
        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionAttackerConnectionsMismatchPenalty,
            spot.activatedValues.cellFunctionAttackerConnectionsMismatchPenalty,
            defaultSpot.values.cellFunctionAttackerConnectionsMismatchPenalty,
            base + "cell.function.attacker.connections mismatch penalty",
            parserTask);

        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionConstructorMutationNeuronDataProbability,
            spot.activatedValues.cellFunctionConstructorMutationNeuronDataProbability,
            defaultSpot.values.cellFunctionConstructorMutationNeuronDataProbability,
            base + "cell.function.constructor.mutation probability.neuron data",
            parserTask);
        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionConstructorMutationDataProbability,
            spot.activatedValues.cellFunctionConstructorMutationDataProbability,
            defaultSpot.values.cellFunctionConstructorMutationDataProbability,
            base + " cell.function.constructor.mutation probability.data ",
            parserTask);
        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionConstructorMutationCellFunctionProbability,
            spot.activatedValues.cellFunctionConstructorMutationCellFunctionProbability,
            defaultSpot.values.cellFunctionConstructorMutationCellFunctionProbability,
            base + "cell.function.constructor.mutation probability.cell function",
            parserTask);
        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionConstructorMutationInsertionProbability,
            spot.activatedValues.cellFunctionConstructorMutationInsertionProbability,
            defaultSpot.values.cellFunctionConstructorMutationInsertionProbability,
            base + "cell.function.constructor.mutation probability.insertion",
            parserTask);
        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionConstructorMutationDeletionProbability,
            spot.activatedValues.cellFunctionConstructorMutationDeletionProbability,
            defaultSpot.values.cellFunctionConstructorMutationDeletionProbability,
            base + "cell.function.constructor.mutation probability.deletion",
            parserTask);
        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionConstructorMutationTranslationProbability,
            spot.activatedValues.cellFunctionConstructorMutationTranslationProbability,
            defaultSpot.values.cellFunctionConstructorMutationTranslationProbability,
            "cell.function.constructor.mutation probability.translation",
            parserTask);
        encodeDecodeSpotProperty(
            tree,
            spot.values.cellFunctionConstructorMutationDuplicationProbability,
            spot.activatedValues.cellFunctionConstructorMutationDuplicationProbability,
            defaultSpot.values.cellFunctionConstructorMutationDuplicationProbability,
            "cell.function.constructor.mutation probability.duplication",
            parserTask);
    }
}

template <typename T>
void AuxiliaryDataParser::encodeDecodeSpotProperty(
    boost::property_tree::ptree& tree,
    T& parameter,
    bool& isActivated,
    T const& defaultValue,
    std::string const& node,
    ParserTask task)
{
    JsonParser::encodeDecode(tree, isActivated, false, node + ".activated", task);
    JsonParser::encodeDecode(tree, parameter, defaultValue, node + ".value", task);
}

template <typename T>
void AuxiliaryDataParser::encodeDecodeColorDependentProperty(
    boost::property_tree::ptree& tree,
    T& parameter,
    bool& isColorDependent,
    T const& defaultValue,
    std::string const& node,
    ParserTask task)
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        JsonParser::encodeDecode(tree, parameter[i], defaultValue[i], "simulation parameters.radiation.absorption.color[" + std::to_string(i) + "]", task);
    }
    if (task == ParserTask::Decode) {
        isColorDependent = false;
        for (int i = 1; i < MAX_COLORS; ++i) {
            if (parameter[i] != parameter[0]) {
                isColorDependent = true;
            }
        }
    }
}
