#include "AuxiliaryDataParser.h"

#include "GeneralSettings.h"
#include "Settings.h"

namespace
{
    template <typename T>
    void encodeDecodeProperty(boost::property_tree::ptree& tree, T& parameter, T const& defaultValue, std::string const& node, ParserTask task)
    {
        JsonParser::encodeDecode(tree, parameter, defaultValue, node, task);
    }

    template <>
    void encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        ColorVector<float>& parameter,
        ColorVector<float> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            encodeDecodeProperty(tree, parameter[i], defaultValue[i], node + "[" + std::to_string(i) + "]", task);
        }
    }

    template <>
    void encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        ColorVector<int>& parameter,
        ColorVector<int> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            encodeDecodeProperty(tree, parameter[i], defaultValue[i], node + "[" + std::to_string(i) + "]", task);
        }
    }

    template <>
    void encodeDecodeProperty<ColorMatrix<float>>(
        boost::property_tree::ptree& tree,
        ColorMatrix<float>& parameter,
        ColorMatrix<float> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                encodeDecodeProperty(tree, parameter[i][j], defaultValue[i][j], node + "[" + std::to_string(i) + ", " + std::to_string(j) + "]", task);
            }
        }
    }

    template <>
    void encodeDecodeProperty<ColorMatrix<int>>(
        boost::property_tree::ptree& tree,
        ColorMatrix<int>& parameter,
        ColorMatrix<int> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                encodeDecodeProperty(tree, parameter[i][j], defaultValue[i][j], node + "[" + std::to_string(i) + ", " + std::to_string(j) + "]", task);
            }
        }
    }

    template <>
    void encodeDecodeProperty<ColorMatrix<bool>>(
        boost::property_tree::ptree& tree,
        ColorMatrix<bool>& parameter,
        ColorMatrix<bool> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                encodeDecodeProperty(tree, parameter[i][j], defaultValue[i][j], node + "[" + std::to_string(i) + ", " + std::to_string(j) + "]", task);
            }
        }
    }

    template <typename T>
    void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        T& parameter,
        bool& isActivated,
        T const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        encodeDecodeProperty(tree, isActivated, false, node + ".activated", task);
        encodeDecodeProperty(tree, parameter, defaultValue, node + ".value", task);
    }

    template <>
    void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        ColorVector<float>& parameter,
        bool& isActivated,
        ColorVector<float> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        encodeDecodeProperty(tree, isActivated, false, node + ".activated", task);
        encodeDecodeProperty(tree, parameter, defaultValue, node, task);
    }

    template <>
    void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        ColorVector<int>& parameter,
        bool& isActivated,
        ColorVector<int> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        encodeDecodeProperty(tree, isActivated, false, node + ".activated", task);
        encodeDecodeProperty(tree, parameter, defaultValue, node, task);
    }

    void encodeDecode(boost::property_tree::ptree& tree, SimulationParameters& parameters, ParserTask parserTask)
    {
        //simulation parameters
        SimulationParameters defaultParameters;
        encodeDecodeProperty(tree, parameters.backgroundColor, defaultParameters.backgroundColor, "simulation parameters.background color", parserTask);
        encodeDecodeProperty(tree, parameters.cellColorization, defaultParameters.cellColorization, "simulation parameters.cell colorization", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.zoomLevelNeuronalActivity,
            defaultParameters.zoomLevelNeuronalActivity,
            "simulation parameters.zoom level.neural activity",
            parserTask);
        encodeDecodeProperty(tree, parameters.timestepSize, defaultParameters.timestepSize, "simulation parameters.time step size", parserTask);

        encodeDecodeProperty(tree, parameters.motionType, defaultParameters.motionType, "simulation parameters.motion.type", parserTask);
        if (parameters.motionType == MotionType_Fluid) {
            encodeDecodeProperty(
                tree,
                parameters.motionData.fluidMotion.smoothingLength,
                defaultParameters.motionData.fluidMotion.smoothingLength,
                "simulation parameters.fluid.smoothing length",
                parserTask);
            encodeDecodeProperty(
                tree,
                parameters.motionData.fluidMotion.pressureStrength,
                defaultParameters.motionData.fluidMotion.pressureStrength,
                "simulation parameters.fluid.pressure strength",
                parserTask);
            encodeDecodeProperty(
                tree,
                parameters.motionData.fluidMotion.viscosityStrength,
                defaultParameters.motionData.fluidMotion.viscosityStrength,
                "simulation parameters.fluid.viscosity strength",
                parserTask);
        } else {
            encodeDecodeProperty(
                tree,
                parameters.motionData.collisionMotion.cellMaxCollisionDistance,
                defaultParameters.motionData.collisionMotion.cellMaxCollisionDistance,
                "simulation parameters.motion.collision.max distance",
                parserTask);
            encodeDecodeProperty(
                tree,
                parameters.motionData.collisionMotion.cellRepulsionStrength,
                defaultParameters.motionData.collisionMotion.cellRepulsionStrength,
                "simulation parameters.motion.collision.repulsion strength",
                parserTask);
        }

        encodeDecodeProperty(tree, parameters.baseValues.friction, defaultParameters.baseValues.friction, "simulation parameters.friction", parserTask);
        encodeDecodeProperty(tree, parameters.baseValues.rigidity, defaultParameters.baseValues.rigidity, "simulation parameters.rigidity", parserTask);
        encodeDecodeProperty(tree, parameters.cellMaxVelocity, defaultParameters.cellMaxVelocity, "simulation parameters.cell.max velocity", parserTask);
        encodeDecodeProperty(
            tree, parameters.cellMaxBindingDistance, defaultParameters.cellMaxBindingDistance, "simulation parameters.cell.max binding distance", parserTask);
        encodeDecodeProperty(tree, parameters.cellNormalEnergy, defaultParameters.cellNormalEnergy, "simulation parameters.cell.normal energy", parserTask);

        encodeDecodeProperty(tree, parameters.cellMinDistance, defaultParameters.cellMinDistance, "simulation parameters.cell.min distance", parserTask);
        encodeDecodeProperty(
            tree, parameters.baseValues.cellMaxForce, defaultParameters.baseValues.cellMaxForce, "simulation parameters.cell.max force", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellMaxForceDecayProb,
            defaultParameters.cellMaxForceDecayProb,
            "simulation parameters.cell.max force decay probability",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellNumExecutionOrderNumbers,
            defaultParameters.cellNumExecutionOrderNumbers,
            "simulation parameters.cell.max execution order number",
            parserTask);
        encodeDecodeProperty(
            tree, parameters.baseValues.cellMinEnergy, defaultParameters.baseValues.cellMinEnergy, "simulation parameters.cell.min energy", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFusionVelocity,
            defaultParameters.baseValues.cellFusionVelocity,
            "simulation parameters.cell.fusion velocity",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellMaxBindingEnergy,
            parameters.baseValues.cellMaxBindingEnergy,
            "simulation parameters.cell.max binding energy",
            parserTask);
        encodeDecodeProperty(tree, parameters.cellMaxAge, defaultParameters.cellMaxAge, "simulation parameters.cell.max age", parserTask);
        encodeDecodeProperty(
            tree, parameters.cellMaxAgeBalancer, defaultParameters.cellMaxAgeBalancer, "simulation parameters.cell.max age.balance.enabled", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellMaxAgeBalancerInterval,
            defaultParameters.cellMaxAgeBalancerInterval,
            "simulation parameters.cell.max age.balance.interval",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellColorTransitionDuration,
            defaultParameters.baseValues.cellColorTransitionDuration,
            "simulation parameters.cell.color transition rules.duration",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellColorTransitionTargetColor,
            defaultParameters.baseValues.cellColorTransitionTargetColor,
            "simulation parameters.cell.color transition rules.target color",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.radiationCellAgeStrength,
            defaultParameters.baseValues.radiationCellAgeStrength,
            "simulation parameters.radiation.factor",
            parserTask);
        encodeDecodeProperty(tree, parameters.radiationProb, defaultParameters.radiationProb, "simulation parameters.radiation.probability", parserTask);
        encodeDecodeProperty(
            tree,
            parameters.radiationVelocityMultiplier,
            defaultParameters.radiationVelocityMultiplier,
            "simulation parameters.radiation.velocity multiplier",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.radiationVelocityPerturbation,
            defaultParameters.radiationVelocityPerturbation,
            "simulation parameters.radiation.velocity perturbation",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.radiationAbsorption,
            defaultParameters.baseValues.radiationAbsorption,
            "simulation parameters.radiation.absorption",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.radiationAbsorptionVelocityPenalty,
            defaultParameters.radiationAbsorptionVelocityPenalty,
            "simulation parameters.radiation.absorption velocity penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.highRadiationMinCellEnergy,
            defaultParameters.highRadiationMinCellEnergy,
            "simulation parameters.high radiation.min cell energy",
            parserTask);
        encodeDecodeProperty(
            tree, parameters.highRadiationFactor, defaultParameters.highRadiationFactor, "simulation parameters.high radiation.factor", parserTask);
        encodeDecodeProperty(
            tree, parameters.radiationMinCellAge, defaultParameters.radiationMinCellAge, "simulation parameters.radiation.min cell age", parserTask);

        encodeDecodeProperty(tree, parameters.clusterDecay, defaultParameters.clusterDecay, "simulation parameters.cluster.decay", parserTask);
        encodeDecodeProperty(
            tree, parameters.clusterDecayProb, defaultParameters.clusterDecayProb, "simulation parameters.cluster.decay probability", parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorPumpEnergyFactor,
            defaultParameters.cellFunctionConstructorPumpEnergyFactor,
            "simulation parameters.cell.function.constructor.pump energy factor",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorOffspringDistance,
            defaultParameters.cellFunctionConstructorOffspringDistance,
            "simulation parameters.cell.function.constructor.offspring distance",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorConnectingCellMaxDistance,
            defaultParameters.cellFunctionConstructorConnectingCellMaxDistance,
            "simulation parameters.cell.function.constructor.connecting cell max distance",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorActivityThreshold,
            defaultParameters.cellFunctionConstructorActivityThreshold,
            "simulation parameters.cell.function.constructor.activity threshold",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability,
            "simulation parameters.cell.function.constructor.mutation probability.neuron data",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationPropertiesProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationPropertiesProbability,
            "simulation parameters.cell.function.constructor.mutation probability.data",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationGeometryProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationGeometryProbability,
            "simulation parameters.cell.function.constructor.mutation probability.geometry",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationCustomGeometryProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationCustomGeometryProbability,
            "simulation parameters.cell.function.constructor.mutation probability.custom geometry",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability,
            "simulation parameters.cell.function.constructor.mutation probability.cell function",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationInsertionProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationInsertionProbability,
            "simulation parameters.cell.function.constructor.mutation probability.insertion",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationDeletionProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationDeletionProbability,
            "simulation parameters.cell.function.constructor.mutation probability.deletion",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationTranslationProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationTranslationProbability,
            "simulation parameters.cell.function.constructor.mutation probability.translation",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationDuplicationProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationDuplicationProbability,
            "simulation parameters.cell.function.constructor.mutation probability.duplication",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationColorProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationColorProbability,
            "simulation parameters.cell.function.constructor.mutation probability.color",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionConstructorMutationUniformColorProbability,
            defaultParameters.baseValues.cellFunctionConstructorMutationUniformColorProbability,
            "simulation parameters.cell.function.constructor.mutation probability.uniform color",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorMutationColorTransitions,
            defaultParameters.cellFunctionConstructorMutationColorTransitions,
            "simulation parameters.cell.function.constructor.mutation color transition",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorMutationSelfReplication,
            defaultParameters.cellFunctionConstructorMutationSelfReplication,
            "simulation parameters.cell.function.constructor.mutation self replication",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorMutationPreventDepthIncrease,
            defaultParameters.cellFunctionConstructorMutationPreventDepthIncrease,
            "simulation parameters.cell.function.constructor.mutation prevent depth increase",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructorCheckCompletenessForSelfReplication,
            defaultParameters.cellFunctionConstructorCheckCompletenessForSelfReplication,
            "simulation parameters.cell.function.constructor.completeness check for self-replication",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionConstructionUnlimitedEnergy,
            defaultParameters.cellFunctionConstructionUnlimitedEnergy,
            "simulation parameters.cell.function.constructor.unlimited energy",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionInjectorRadius,
            defaultParameters.cellFunctionInjectorRadius,
            "simulation parameters.cell.function.injector.radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionInjectorDurationColorMatrix,
            defaultParameters.cellFunctionInjectorDurationColorMatrix,
            "simulation parameters.cell.function.injector.duration",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerRadius,
            defaultParameters.cellFunctionAttackerRadius,
            "simulation parameters.cell.function.attacker.radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerStrength,
            defaultParameters.cellFunctionAttackerStrength,
            "simulation parameters.cell.function.attacker.strength",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerEnergyDistributionRadius,
            defaultParameters.cellFunctionAttackerEnergyDistributionRadius,
            "simulation parameters.cell.function.attacker.energy distribution radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerEnergyDistributionValue,
            defaultParameters.cellFunctionAttackerEnergyDistributionValue,
            "simulation parameters.cell.function.attacker.energy distribution value",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerColorInhomogeneityFactor,
            defaultParameters.cellFunctionAttackerColorInhomogeneityFactor,
            "simulation parameters.cell.function.attacker.color inhomogeneity factor",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerActivityThreshold,
            defaultParameters.cellFunctionAttackerActivityThreshold,
            "simulation parameters.cell.function.attacker.activity threshold",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerEnergyCost,
            defaultParameters.baseValues.cellFunctionAttackerEnergyCost,
            "simulation parameters.cell.function.attacker.energy cost",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent,
            defaultParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent,
            "simulation parameters.cell.function.attacker.geometry deviation exponent",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix,
            defaultParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix,
            "simulation parameters.cell.function.attacker.food chain color matrix",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty,
            defaultParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty,
            "simulation parameters.cell.function.attacker.connections mismatch penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerGenomeSizeBonus,
            defaultParameters.cellFunctionAttackerGenomeSizeBonus,
            "simulation parameters.cell.function.attacker.genome size bonus",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerSameMutantPenalty,
            defaultParameters.cellFunctionAttackerSameMutantPenalty,
            "simulation parameters.cell.function.attacker.same mutant penalty",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerSensorDetectionFactor,
            defaultParameters.cellFunctionAttackerSensorDetectionFactor,
            "simulation parameters.cell.function.attacker.sensor detection factor",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionAttackerDestroyCells,
            defaultParameters.cellFunctionAttackerDestroyCells,
            "simulation parameters.cell.function.attacker.destroy cells",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionDefenderAgainstAttackerStrength,
            defaultParameters.cellFunctionDefenderAgainstAttackerStrength,
            "simulation parameters.cell.function.defender.against attacker strength",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionDefenderAgainstInjectorStrength,
            defaultParameters.cellFunctionDefenderAgainstInjectorStrength,
            "simulation parameters.cell.function.defender.against injector strength",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionTransmitterEnergyDistributionSameCreature,
            defaultParameters.cellFunctionTransmitterEnergyDistributionSameCreature,
            "simulation parameters.cell.function.transmitter.energy distribution same creature",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionTransmitterEnergyDistributionRadius,
            defaultParameters.cellFunctionTransmitterEnergyDistributionRadius,
            "simulation parameters.cell.function.transmitter.energy distribution radius",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionTransmitterEnergyDistributionValue,
            defaultParameters.cellFunctionTransmitterEnergyDistributionValue,
            "simulation parameters.cell.function.transmitter.energy distribution value",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleContractionExpansionDelta,
            defaultParameters.cellFunctionMuscleContractionExpansionDelta,
            "simulation parameters.cell.function.muscle.contraction expansion delta",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleMovementAcceleration,
            defaultParameters.cellFunctionMuscleMovementAcceleration,
            "simulation parameters.cell.function.muscle.movement acceleration",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleBendingAngle,
            defaultParameters.cellFunctionMuscleBendingAngle,
            "simulation parameters.cell.function.muscle.bending angle",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleBendingAcceleration,
            defaultParameters.cellFunctionMuscleBendingAcceleration,
            "simulation parameters.cell.function.muscle.bending acceleration",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionMuscleBendingAccelerationThreshold,
            defaultParameters.cellFunctionMuscleBendingAccelerationThreshold,
            "simulation parameters.cell.function.muscle.bending acceleration threshold",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.particleTransformationAllowed,
            defaultParameters.particleTransformationAllowed,
            "simulation parameters.particle.transformation allowed",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.particleTransformationRandomCellFunction,
            defaultParameters.particleTransformationRandomCellFunction,
            "simulation parameters.particle.transformation.random cell function",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.particleTransformationMaxGenomeSize,
            defaultParameters.particleTransformationMaxGenomeSize,
            "simulation parameters.particle.transformation.max genome size",
            parserTask);

        encodeDecodeProperty(
            tree,
            parameters.cellFunctionSensorRange,
            defaultParameters.cellFunctionSensorRange,
            "simulation parameters.cell.function.sensor.range",
            parserTask);
        encodeDecodeProperty(
            tree,
            parameters.cellFunctionSensorActivityThreshold,
            defaultParameters.cellFunctionSensorActivityThreshold,
            "simulation parameters.cell.function.sensor.activity threshold",
            parserTask);

        //particle sources
        encodeDecodeProperty(
            tree, parameters.numParticleSources, defaultParameters.numParticleSources, "simulation parameters.particle sources.num sources", parserTask);
        for (int index = 0; index < parameters.numParticleSources; ++index) {
            std::string base = "simulation parameters.particle sources." + std::to_string(index) + ".";
            auto& source = parameters.particleSources[index];
            auto& defaultSource = defaultParameters.particleSources[index];
            encodeDecodeProperty(tree, source.posX, defaultSource.posX, base + "pos.x", parserTask);
            encodeDecodeProperty(tree, source.posY, defaultSource.posY, base + "pos.y", parserTask);
            encodeDecodeProperty(tree, source.velX, defaultSource.velX, base + "vel.x", parserTask);
            encodeDecodeProperty(tree, source.velY, defaultSource.velY, base + "vel.y", parserTask);
            encodeDecodeProperty(tree, source.useAngle, defaultSource.useAngle, base + "use angle", parserTask);
            encodeDecodeProperty(tree, source.angle, defaultSource.angle, base + "angle", parserTask);
            encodeDecodeProperty(tree, source.shapeType, defaultSource.shapeType, base + "shape.type", parserTask);
            if (source.shapeType == SpotShapeType_Circular) {
                encodeDecodeProperty(
                    tree,
                    source.shapeData.circularRadiationSource.radius,
                    defaultSource.shapeData.circularRadiationSource.radius,
                    base + "shape.circular.radius",
                    parserTask);
            }
            if (source.shapeType == SpotShapeType_Rectangular) {
                encodeDecodeProperty(
                    tree,
                    source.shapeData.rectangularRadiationSource.width,
                    defaultSource.shapeData.rectangularRadiationSource.width,
                    base + "shape.rectangular.width",
                    parserTask);
                encodeDecodeProperty(
                    tree,
                    source.shapeData.rectangularRadiationSource.height,
                    defaultSource.shapeData.rectangularRadiationSource.height,
                    base + "shape.rectangular.height",
                    parserTask);
            }
        }

        //spots
        encodeDecodeProperty(tree, parameters.numSpots, defaultParameters.numSpots, "simulation parameters.spots.num spots", parserTask);
        for (int index = 0; index < parameters.numSpots; ++index) {
            std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
            auto& spot = parameters.spots[index];
            auto& defaultSpot = defaultParameters.spots[index];
            encodeDecodeProperty(tree, spot.color, defaultSpot.color, base + "color", parserTask);
            encodeDecodeProperty(tree, spot.posX, defaultSpot.posX, base + "pos.x", parserTask);
            encodeDecodeProperty(tree, spot.posY, defaultSpot.posY, base + "pos.y", parserTask);

            encodeDecodeProperty(tree, spot.shapeType, defaultSpot.shapeType, base + "shape.type", parserTask);
            if (spot.shapeType == SpotShapeType_Circular) {
                encodeDecodeProperty(
                    tree,
                    spot.shapeData.circularSpot.coreRadius,
                    defaultSpot.shapeData.circularSpot.coreRadius,
                    base + "shape.circular.core radius",
                    parserTask);
            }
            if (spot.shapeType == SpotShapeType_Rectangular) {
                encodeDecodeProperty(
                    tree, spot.shapeData.rectangularSpot.width, defaultSpot.shapeData.rectangularSpot.width, base + "shape.rectangular.core width", parserTask);
                encodeDecodeProperty(
                    tree,
                    spot.shapeData.rectangularSpot.height,
                    defaultSpot.shapeData.rectangularSpot.height,
                    base + "shape.rectangular.core height",
                    parserTask);
            }
            encodeDecodeProperty(tree, spot.flowType, defaultSpot.flowType, base + "flow.type", parserTask);
            if (spot.flowType == FlowType_Radial) {
                encodeDecodeProperty(
                    tree, spot.flowData.radialFlow.orientation, defaultSpot.flowData.radialFlow.orientation, base + "flow.radial.orientation", parserTask);
                encodeDecodeProperty(
                    tree, spot.flowData.radialFlow.strength, defaultSpot.flowData.radialFlow.strength, base + "flow.radial.strength", parserTask);
                encodeDecodeProperty(
                    tree, spot.flowData.radialFlow.driftAngle, defaultSpot.flowData.radialFlow.driftAngle, base + "flow.radial.drift angle", parserTask);
            }
            if (spot.flowType == FlowType_Central) {
                encodeDecodeProperty(
                    tree, spot.flowData.centralFlow.strength, defaultSpot.flowData.centralFlow.strength, base + "flow.central.strength", parserTask);
            }
            if (spot.flowType == FlowType_Linear) {
                encodeDecodeProperty(tree, spot.flowData.linearFlow.angle, defaultSpot.flowData.linearFlow.angle, base + "flow.linear.angle", parserTask);
                encodeDecodeProperty(
                    tree, spot.flowData.linearFlow.strength, defaultSpot.flowData.linearFlow.strength, base + "flow.linear.strength", parserTask);
            }
            encodeDecodeProperty(tree, spot.fadeoutRadius, defaultSpot.fadeoutRadius, base + "fadeout radius", parserTask);

            encodeDecodeSpotProperty(tree, spot.values.friction, spot.activatedValues.friction, defaultSpot.values.friction, base + "friction", parserTask);
            encodeDecodeSpotProperty(tree, spot.values.rigidity, spot.activatedValues.rigidity, defaultSpot.values.rigidity, base + "rigidity", parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.radiationAbsorption,
                spot.activatedValues.radiationAbsorption,
                defaultSpot.values.radiationAbsorption,
                base + "radiation.absorption",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.radiationCellAgeStrength,
                spot.activatedValues.radiationCellAgeStrength,
                defaultSpot.values.radiationCellAgeStrength,
                base + "radiation.factor",
                parserTask);
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

            encodeDecodeProperty(tree, spot.activatedValues.cellColorTransition, false, base + "cell.color transition rules.activated", parserTask);
            encodeDecodeProperty(
                tree,
                spot.values.cellColorTransitionDuration,
                defaultSpot.values.cellColorTransitionDuration,
                base + "cell.color transition rules.duration",
                parserTask);
            encodeDecodeProperty(
                tree,
                spot.values.cellColorTransitionTargetColor,
                defaultSpot.values.cellColorTransitionTargetColor,
                base + "cell.color transition rules.target color",
                parserTask);

            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionAttackerEnergyCost,
                spot.activatedValues.cellFunctionAttackerEnergyCost,
                defaultSpot.values.cellFunctionAttackerEnergyCost,
                base + "cell.function.attacker.energy cost",
                parserTask);
            encodeDecodeProperty(
                tree,
                spot.activatedValues.cellFunctionAttackerFoodChainColorMatrix,
                false,
                base + "cell.function.attacker.food chain color matrix.activated",
                parserTask);
            encodeDecodeProperty(
                tree,
                spot.values.cellFunctionAttackerFoodChainColorMatrix,
                defaultSpot.values.cellFunctionAttackerFoodChainColorMatrix,
                base + "cell.function.attacker.food chain color matrix",
                parserTask);
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
                spot.values.cellFunctionConstructorMutationPropertiesProbability,
                spot.activatedValues.cellFunctionConstructorMutationPropertiesProbability,
                defaultSpot.values.cellFunctionConstructorMutationPropertiesProbability,
                base + "cell.function.constructor.mutation probability.data ",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionConstructorMutationGeometryProbability,
                spot.activatedValues.cellFunctionConstructorMutationGeometryProbability,
                defaultSpot.values.cellFunctionConstructorMutationGeometryProbability,
                base + "cell.function.constructor.mutation probability.geometry",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionConstructorMutationCustomGeometryProbability,
                spot.activatedValues.cellFunctionConstructorMutationCustomGeometryProbability,
                defaultSpot.values.cellFunctionConstructorMutationCustomGeometryProbability,
                base + "cell.function.constructor.mutation probability.custom geometry",
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
                base + "cell.function.constructor.mutation probability.translation",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionConstructorMutationDuplicationProbability,
                spot.activatedValues.cellFunctionConstructorMutationDuplicationProbability,
                defaultSpot.values.cellFunctionConstructorMutationDuplicationProbability,
                base + "cell.function.constructor.mutation probability.duplication",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionConstructorMutationColorProbability,
                spot.activatedValues.cellFunctionConstructorMutationColorProbability,
                defaultSpot.values.cellFunctionConstructorMutationColorProbability,
                base + "cell.function.constructor.mutation probability.color",
                parserTask);
            encodeDecodeSpotProperty(
                tree,
                spot.values.cellFunctionConstructorMutationUniformColorProbability,
                spot.activatedValues.cellFunctionConstructorMutationUniformColorProbability,
                defaultSpot.values.cellFunctionConstructorMutationUniformColorProbability,
                base + "cell.function.constructor.mutation probability.uniform color",
                parserTask);
        }
    }

    void encodeDecode(boost::property_tree::ptree& tree, AuxiliaryData& data, ParserTask parserTask)
    {
        AuxiliaryData defaultSettings;

        //general settings
        encodeDecodeProperty(tree, data.timestep, uint64_t(0), "general.time step", parserTask);
        encodeDecodeProperty(tree, data.zoom, 4.0f, "general.zoom", parserTask);
        encodeDecodeProperty(tree, data.center.x, 0.0f, "general.center.x", parserTask);
        encodeDecodeProperty(tree, data.center.y, 0.0f, "general.center.y", parserTask);
        encodeDecodeProperty(tree, data.generalSettings.worldSizeX, defaultSettings.generalSettings.worldSizeX, "general.world size.x", parserTask);
        encodeDecodeProperty(tree, data.generalSettings.worldSizeY, defaultSettings.generalSettings.worldSizeY, "general.world size.y", parserTask);

        encodeDecode(tree, data.simulationParameters, parserTask);
    }
}

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
