#include "AuxiliaryDataParserService.h"

#include "GeneralSettings.h"
#include "LegacyAuxiliaryDataParserService.h"
#include "Settings.h"

#include "PropertyParser.h"

namespace
{
    void encodeDecode(
        boost::property_tree::ptree& tree,
        SimulationParameters& parameters,
        MissingParameters& missingParameters,
        MissingFeatures& missingFeatures,
        ParserTask parserTask)
    {
        SimulationParameters defaultParameters;

        PropertyParser::encodeDecode(tree, parameters.backgroundColor, defaultParameters.backgroundColor, "simulation parameters.background color", parserTask);
        PropertyParser::encodeDecode(tree, parameters.cellColoring, defaultParameters.cellColoring, "simulation parameters.cell colorization", parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.cellGlowColoring, defaultParameters.cellGlowColoring, "simulation parameters.cell glow.coloring", parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellGlowRadius,
            defaultParameters.cellGlowRadius,
            "simulation parameters.cell glow.radius",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellGlowStrength,
            defaultParameters.cellGlowStrength,
            "simulation parameters.cell glow.strength",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.highlightedCellFunction,
            defaultParameters.highlightedCellFunction,
            "simulation parameters.highlighted cell function",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.zoomLevelNeuronalActivity,
            defaultParameters.zoomLevelNeuronalActivity,
            "simulation parameters.zoom level.neural activity",
            parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.borderlessRendering, defaultParameters.borderlessRendering, "simulation parameters.borderless rendering", parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.markReferenceDomain, defaultParameters.markReferenceDomain, "simulation parameters.mark reference domain", parserTask);
        PropertyParser::encodeDecode(tree, parameters.gridLines, defaultParameters.gridLines, "simulation parameters.grid lines", parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.attackVisualization, defaultParameters.attackVisualization, "simulation parameters.attack visualization", parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.muscleMovementVisualization,
            defaultParameters.muscleMovementVisualization,
            "simulation parameters.muscle movement visualization",
            parserTask);
        PropertyParser::encodeDecode(tree, parameters.cellRadius, defaultParameters.cellRadius, "simulation parameters.cek", parserTask);

        PropertyParser::encodeDecode(tree, parameters.timestepSize, defaultParameters.timestepSize, "simulation parameters.time step size", parserTask);

        PropertyParser::encodeDecode(tree, parameters.motionType, defaultParameters.motionType, "simulation parameters.motion.type", parserTask);
        if (parameters.motionType == MotionType_Fluid) {
            PropertyParser::encodeDecode(
                tree,
                parameters.motionData.fluidMotion.smoothingLength,
                defaultParameters.motionData.fluidMotion.smoothingLength,
                "simulation parameters.fluid.smoothing length",
                parserTask);
            PropertyParser::encodeDecode(
                tree,
                parameters.motionData.fluidMotion.pressureStrength,
                defaultParameters.motionData.fluidMotion.pressureStrength,
                "simulation parameters.fluid.pressure strength",
                parserTask);
            PropertyParser::encodeDecode(
                tree,
                parameters.motionData.fluidMotion.viscosityStrength,
                defaultParameters.motionData.fluidMotion.viscosityStrength,
                "simulation parameters.fluid.viscosity strength",
                parserTask);
        } else {
            PropertyParser::encodeDecode(
                tree,
                parameters.motionData.collisionMotion.cellMaxCollisionDistance,
                defaultParameters.motionData.collisionMotion.cellMaxCollisionDistance,
                "simulation parameters.motion.collision.max distance",
                parserTask);
            PropertyParser::encodeDecode(
                tree,
                parameters.motionData.collisionMotion.cellRepulsionStrength,
                defaultParameters.motionData.collisionMotion.cellRepulsionStrength,
                "simulation parameters.motion.collision.repulsion strength",
                parserTask);
        }

        PropertyParser::encodeDecode(tree, parameters.baseValues.friction, defaultParameters.baseValues.friction, "simulation parameters.friction", parserTask);
        PropertyParser::encodeDecode(tree, parameters.baseValues.rigidity, defaultParameters.baseValues.rigidity, "simulation parameters.rigidity", parserTask);
        PropertyParser::encodeDecode(tree, parameters.cellMaxVelocity, defaultParameters.cellMaxVelocity, "simulation parameters.cell.max velocity", parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.cellMaxBindingDistance, defaultParameters.cellMaxBindingDistance, "simulation parameters.cell.max binding distance", parserTask);
        PropertyParser::encodeDecode(tree, parameters.cellNormalEnergy, defaultParameters.cellNormalEnergy, "simulation parameters.cell.normal energy", parserTask);

        PropertyParser::encodeDecode(tree, parameters.cellMinDistance, defaultParameters.cellMinDistance, "simulation parameters.cell.min distance", parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.baseValues.cellMaxForce, defaultParameters.baseValues.cellMaxForce, "simulation parameters.cell.max force", parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellMaxForceDecayProb,
            defaultParameters.cellMaxForceDecayProb,
            "simulation parameters.cell.max force decay probability",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellNumExecutionOrderNumbers,
            defaultParameters.cellNumExecutionOrderNumbers,
            "simulation parameters.cell.max execution order number",
            parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.baseValues.cellMinEnergy, defaultParameters.baseValues.cellMinEnergy, "simulation parameters.cell.min energy", parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellFusionVelocity,
            defaultParameters.baseValues.cellFusionVelocity,
            "simulation parameters.cell.fusion velocity",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellMaxBindingEnergy,
            parameters.baseValues.cellMaxBindingEnergy,
            "simulation parameters.cell.max binding energy",
            parserTask);
        PropertyParser::encodeDecode(tree, parameters.cellMaxAge, defaultParameters.cellMaxAge, "simulation parameters.cell.max age", parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.cellMaxAgeBalancer, defaultParameters.cellMaxAgeBalancer, "simulation parameters.cell.max age.balance.enabled", parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellMaxAgeBalancerInterval,
            defaultParameters.cellMaxAgeBalancerInterval,
            "simulation parameters.cell.max age.balance.interval",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellInactiveMaxAgeActivated,
            defaultParameters.cellInactiveMaxAgeActivated,
            "simulation parameters.cell.inactive max age activated",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellInactiveMaxAge,
            defaultParameters.baseValues.cellInactiveMaxAge,
            "simulation parameters.cell.inactive max age",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellEmergentMaxAgeActivated,
            defaultParameters.cellEmergentMaxAgeActivated,
            "simulation parameters.cell.nutrient max age activated",
            parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.cellEmergentMaxAge, defaultParameters.cellEmergentMaxAge, "simulation parameters.cell.nutrient max age", parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellResetAgeAfterActivation,
            defaultParameters.cellResetAgeAfterActivation,
            "simulation parameters.cell.reset age after activation",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellColorTransitionDuration,
            defaultParameters.baseValues.cellColorTransitionDuration,
            "simulation parameters.cell.color transition rules.duration",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellColorTransitionTargetColor,
            defaultParameters.baseValues.cellColorTransitionTargetColor,
            "simulation parameters.cell.color transition rules.target color",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.genomeComplexityRamificationFactor,
            defaultParameters.genomeComplexityRamificationFactor,
            "simulation parameters.genome complexity.genome complexity ramification factor",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.genomeComplexitySizeFactor,
            defaultParameters.genomeComplexitySizeFactor,
            "simulation parameters.genome complexity.genome complexity size factor",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.genomeComplexityNeuronFactor,
            defaultParameters.genomeComplexityNeuronFactor,
            "simulation parameters.genome complexity.genome complexity neuron factor",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.radiationCellAgeStrength,
            defaultParameters.baseValues.radiationCellAgeStrength,
            "simulation parameters.radiation.factor",
            parserTask);
        PropertyParser::encodeDecode(tree, parameters.radiationProb, defaultParameters.radiationProb, "simulation parameters.radiation.probability", parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.radiationVelocityMultiplier,
            defaultParameters.radiationVelocityMultiplier,
            "simulation parameters.radiation.velocity multiplier",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.radiationVelocityPerturbation,
            defaultParameters.radiationVelocityPerturbation,
            "simulation parameters.radiation.velocity perturbation",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.radiationDisableSources,
            defaultParameters.baseValues.radiationDisableSources,
            "simulation parameters.radiation.disable sources",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.radiationAbsorption,
            defaultParameters.baseValues.radiationAbsorption,
            "simulation parameters.radiation.absorption",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.radiationAbsorptionHighVelocityPenalty,
            defaultParameters.radiationAbsorptionHighVelocityPenalty,
            "simulation parameters.radiation.absorption velocity penalty",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.radiationAbsorptionLowVelocityPenalty,
            defaultParameters.baseValues.radiationAbsorptionLowVelocityPenalty,
            "simulation parameters.radiation.absorption low velocity penalty",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.radiationAbsorptionLowConnectionPenalty,
            defaultParameters.radiationAbsorptionLowConnectionPenalty,
            "simulation parameters.radiation.absorption low connection penalty",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty,
            defaultParameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty,
            "simulation parameters.radiation.absorption low genome complexity penalty",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.highRadiationMinCellEnergy,
            defaultParameters.highRadiationMinCellEnergy,
            "simulation parameters.high radiation.min cell energy",
            parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.highRadiationFactor, defaultParameters.highRadiationFactor, "simulation parameters.high radiation.factor", parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.radiationMinCellAge, defaultParameters.radiationMinCellAge, "simulation parameters.radiation.min cell age", parserTask);

        PropertyParser::encodeDecode(
            tree, parameters.externalEnergy, defaultParameters.externalEnergy, "simulation parameters.cell.function.constructor.external energy", parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.externalEnergyInflowFactor,
            defaultParameters.externalEnergyInflowFactor,
            "simulation parameters.cell.function.constructor.external energy supply rate",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.externalEnergyConditionalInflowFactor,
            defaultParameters.externalEnergyConditionalInflowFactor,
            "simulation parameters.cell.function.constructor.pump energy factor",
            parserTask);
        missingParameters.externalEnergyBackflowFactor = PropertyParser::encodeDecode(
            tree,
            parameters.externalEnergyBackflowFactor,
            defaultParameters.externalEnergyBackflowFactor,
            "simulation parameters.cell.function.constructor.external energy backflow",
            parserTask);

        missingParameters.cellDeathConsequences = PropertyParser::encodeDecode(
            tree, parameters.cellDeathConsequences, defaultParameters.cellDeathConsequences, "simulation parameters.cell.death consequences", parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.cellDeathProbability, defaultParameters.cellDeathProbability, "simulation parameters.cell.death probability", parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionConstructorOffspringDistance,
            defaultParameters.cellFunctionConstructorOffspringDistance,
            "simulation parameters.cell.function.constructor.offspring distance",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionConstructorConnectingCellMaxDistance,
            defaultParameters.cellFunctionConstructorConnectingCellMaxDistance,
            "simulation parameters.cell.function.constructor.connecting cell max distance",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionConstructorActivityThreshold,
            defaultParameters.cellFunctionConstructorActivityThreshold,
            "simulation parameters.cell.function.constructor.activity threshold",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionConstructorCheckCompletenessForSelfReplication,
            defaultParameters.cellFunctionConstructorCheckCompletenessForSelfReplication,
            "simulation parameters.cell.function.constructor.completeness check for self-replication",
            parserTask);

        missingParameters.copyMutations = PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationNeuronData,
            defaultParameters.baseValues.cellCopyMutationNeuronData,
            "simulation parameters.cell.copy mutation.neuron data",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationCellProperties,
            defaultParameters.baseValues.cellCopyMutationCellProperties,
            "simulation parameters.cell.copy mutation.cell properties",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationGeometry,
            defaultParameters.baseValues.cellCopyMutationGeometry,
            "simulation parameters.cell.copy mutation.geometry",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationCustomGeometry,
            defaultParameters.baseValues.cellCopyMutationCustomGeometry,
            "simulation parameters.cell.copy mutation.custom geometry",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationCellFunction,
            defaultParameters.baseValues.cellCopyMutationCellFunction,
            "simulation parameters.cell.copy mutation.cell function",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationInsertion,
            defaultParameters.baseValues.cellCopyMutationInsertion,
            "simulation parameters.cell.copy mutation.insertion",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationDeletion,
            defaultParameters.baseValues.cellCopyMutationDeletion,
            "simulation parameters.cell.copy mutation.deletion",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationTranslation,
            defaultParameters.baseValues.cellCopyMutationTranslation,
            "simulation parameters.cell.copy mutation.translation",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationDuplication,
            defaultParameters.baseValues.cellCopyMutationDuplication,
            "simulation parameters.cell.copy mutation.duplication",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationCellColor,
            defaultParameters.baseValues.cellCopyMutationCellColor,
            "simulation parameters.cell.copy mutation.cell color",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationSubgenomeColor,
            defaultParameters.baseValues.cellCopyMutationSubgenomeColor,
            "simulation parameters.cell.copy mutation.subgenome color",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationGenomeColor,
            defaultParameters.baseValues.cellCopyMutationGenomeColor,
            "simulation parameters.cell.copy mutation.genome color",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionConstructorMutationColorTransitions,
            defaultParameters.cellFunctionConstructorMutationColorTransitions,
            "simulation parameters.cell.copy mutation.color transition",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionConstructorMutationSelfReplication,
            defaultParameters.cellFunctionConstructorMutationSelfReplication,
            "simulation parameters.cell.copy mutation.self replication flag",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionConstructorMutationPreventDepthIncrease,
            defaultParameters.cellFunctionConstructorMutationPreventDepthIncrease,
            "simulation parameters.cell.copy mutation.prevent depth increase",
            parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionInjectorRadius,
            defaultParameters.cellFunctionInjectorRadius,
            "simulation parameters.cell.function.injector.radius",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionInjectorDurationColorMatrix,
            defaultParameters.cellFunctionInjectorDurationColorMatrix,
            "simulation parameters.cell.function.injector.duration",
            parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionAttackerRadius,
            defaultParameters.cellFunctionAttackerRadius,
            "simulation parameters.cell.function.attacker.radius",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionAttackerStrength,
            defaultParameters.cellFunctionAttackerStrength,
            "simulation parameters.cell.function.attacker.strength",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionAttackerEnergyDistributionRadius,
            defaultParameters.cellFunctionAttackerEnergyDistributionRadius,
            "simulation parameters.cell.function.attacker.energy distribution radius",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionAttackerEnergyDistributionValue,
            defaultParameters.cellFunctionAttackerEnergyDistributionValue,
            "simulation parameters.cell.function.attacker.energy distribution value",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionAttackerColorInhomogeneityFactor,
            defaultParameters.cellFunctionAttackerColorInhomogeneityFactor,
            "simulation parameters.cell.function.attacker.color inhomogeneity factor",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionAttackerActivityThreshold,
            defaultParameters.cellFunctionAttackerActivityThreshold,
            "simulation parameters.cell.function.attacker.activity threshold",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellFunctionAttackerEnergyCost,
            defaultParameters.baseValues.cellFunctionAttackerEnergyCost,
            "simulation parameters.cell.function.attacker.energy cost",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent,
            defaultParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent,
            "simulation parameters.cell.function.attacker.geometry deviation exponent",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix,
            defaultParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix,
            "simulation parameters.cell.function.attacker.food chain color matrix",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty,
            defaultParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty,
            "simulation parameters.cell.function.attacker.connections mismatch penalty",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus,
            defaultParameters.baseValues.cellFunctionAttackerGenomeComplexityBonus,
            "simulation parameters.cell.function.attacker.genome size bonus",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionAttackerSameMutantPenalty,
            defaultParameters.cellFunctionAttackerSameMutantPenalty,
            "simulation parameters.cell.function.attacker.same mutant penalty",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty,
            defaultParameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty,
            "simulation parameters.cell.function.attacker.new complex mutant penalty",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionAttackerSensorDetectionFactor,
            defaultParameters.cellFunctionAttackerSensorDetectionFactor,
            "simulation parameters.cell.function.attacker.sensor detection factor",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionAttackerDestroyCells,
            defaultParameters.cellFunctionAttackerDestroyCells,
            "simulation parameters.cell.function.attacker.destroy cells",
            parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionDefenderAgainstAttackerStrength,
            defaultParameters.cellFunctionDefenderAgainstAttackerStrength,
            "simulation parameters.cell.function.defender.against attacker strength",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionDefenderAgainstInjectorStrength,
            defaultParameters.cellFunctionDefenderAgainstInjectorStrength,
            "simulation parameters.cell.function.defender.against injector strength",
            parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionTransmitterEnergyDistributionSameCreature,
            defaultParameters.cellFunctionTransmitterEnergyDistributionSameCreature,
            "simulation parameters.cell.function.transmitter.energy distribution same creature",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionTransmitterEnergyDistributionRadius,
            defaultParameters.cellFunctionTransmitterEnergyDistributionRadius,
            "simulation parameters.cell.function.transmitter.energy distribution radius",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionTransmitterEnergyDistributionValue,
            defaultParameters.cellFunctionTransmitterEnergyDistributionValue,
            "simulation parameters.cell.function.transmitter.energy distribution value",
            parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionMuscleContractionExpansionDelta,
            defaultParameters.cellFunctionMuscleContractionExpansionDelta,
            "simulation parameters.cell.function.muscle.contraction expansion delta",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionMuscleMovementAcceleration,
            defaultParameters.cellFunctionMuscleMovementAcceleration,
            "simulation parameters.cell.function.muscle.movement acceleration",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionMuscleBendingAngle,
            defaultParameters.cellFunctionMuscleBendingAngle,
            "simulation parameters.cell.function.muscle.bending angle",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionMuscleBendingAcceleration,
            defaultParameters.cellFunctionMuscleBendingAcceleration,
            "simulation parameters.cell.function.muscle.bending acceleration",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionMuscleBendingAccelerationThreshold,
            defaultParameters.cellFunctionMuscleBendingAccelerationThreshold,
            "simulation parameters.cell.function.muscle.bending acceleration threshold",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionMuscleMovementTowardTargetedObject,
            defaultParameters.cellFunctionMuscleMovementTowardTargetedObject,
            "simulation parameters.cell.function.muscle.movement toward targeted object",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.legacyCellFunctionMuscleMovementAngleFromSensor,
            defaultParameters.legacyCellFunctionMuscleMovementAngleFromSensor,
            "simulation parameters.legacy.cell.function.muscle.movement angle from sensor",
            parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.particleTransformationAllowed,
            defaultParameters.particleTransformationAllowed,
            "simulation parameters.particle.transformation allowed",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.particleTransformationRandomCellFunction,
            defaultParameters.particleTransformationRandomCellFunction,
            "simulation parameters.particle.transformation.random cell function",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.particleTransformationMaxGenomeSize,
            defaultParameters.particleTransformationMaxGenomeSize,
            "simulation parameters.particle.transformation.max genome size",
            parserTask);
        PropertyParser::encodeDecode(
            tree, parameters.particleSplitEnergy, defaultParameters.particleSplitEnergy, "simulation parameters.particle.split energy", parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionSensorRange,
            defaultParameters.cellFunctionSensorRange,
            "simulation parameters.cell.function.sensor.range",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionSensorActivityThreshold,
            defaultParameters.cellFunctionSensorActivityThreshold,
            "simulation parameters.cell.function.sensor.activity threshold",
            parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionReconnectorRadius,
            defaultParameters.cellFunctionReconnectorRadius,
            "simulation parameters.cell.function.reconnector.radius",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionReconnectorActivityThreshold,
            defaultParameters.cellFunctionReconnectorActivityThreshold,
            "simulation parameters.cell.function.reconnector.activity threshold",
            parserTask);

        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionDetonatorRadius,
            defaultParameters.cellFunctionDetonatorRadius,
            "simulation parameters.cell.function.detonator.radius",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionDetonatorChainExplosionProbability,
            defaultParameters.cellFunctionDetonatorChainExplosionProbability,
            "simulation parameters.cell.function.detonator.chain explosion probability",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.cellFunctionDetonatorActivityThreshold,
            defaultParameters.cellFunctionDetonatorActivityThreshold,
            "simulation parameters.cell.function.detonator.activity threshold",
            parserTask);

        //particle sources
        PropertyParser::encodeDecode(
            tree, parameters.numRadiationSources, defaultParameters.numRadiationSources, "simulation parameters.particle sources.num sources", parserTask);
        for (int index = 0; index < parameters.numRadiationSources; ++index) {
            std::string base = "simulation parameters.particle sources." + std::to_string(index) + ".";
            auto& source = parameters.radiationSources[index];
            auto& defaultSource = defaultParameters.radiationSources[index];
            PropertyParser::encodeDecode(tree, source.posX, defaultSource.posX, base + "pos.x", parserTask);
            PropertyParser::encodeDecode(tree, source.posY, defaultSource.posY, base + "pos.y", parserTask);
            PropertyParser::encodeDecode(tree, source.velX, defaultSource.velX, base + "vel.x", parserTask);
            PropertyParser::encodeDecode(tree, source.velY, defaultSource.velY, base + "vel.y", parserTask);
            PropertyParser::encodeDecode(tree, source.useAngle, defaultSource.useAngle, base + "use angle", parserTask);
            PropertyParser::encodeDecode(tree, source.angle, defaultSource.angle, base + "angle", parserTask);
            PropertyParser::encodeDecode(tree, source.shapeType, defaultSource.shapeType, base + "shape.type", parserTask);
            if (source.shapeType == SpotShapeType_Circular) {
                PropertyParser::encodeDecode(
                    tree,
                    source.shapeData.circularRadiationSource.radius,
                    defaultSource.shapeData.circularRadiationSource.radius,
                    base + "shape.circular.radius",
                    parserTask);
            }
            if (source.shapeType == SpotShapeType_Rectangular) {
                PropertyParser::encodeDecode(
                    tree,
                    source.shapeData.rectangularRadiationSource.width,
                    defaultSource.shapeData.rectangularRadiationSource.width,
                    base + "shape.rectangular.width",
                    parserTask);
                PropertyParser::encodeDecode(
                    tree,
                    source.shapeData.rectangularRadiationSource.height,
                    defaultSource.shapeData.rectangularRadiationSource.height,
                    base + "shape.rectangular.height",
                    parserTask);
            }
        }

        //spots
        PropertyParser::encodeDecode(tree, parameters.numSpots, defaultParameters.numSpots, "simulation parameters.spots.num spots", parserTask);
        for (int index = 0; index < parameters.numSpots; ++index) {
            std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
            auto& spot = parameters.spots[index];
            auto& defaultSpot = defaultParameters.spots[index];
            PropertyParser::encodeDecode(tree, spot.color, defaultSpot.color, base + "color", parserTask);
            PropertyParser::encodeDecode(tree, spot.posX, defaultSpot.posX, base + "pos.x", parserTask);
            PropertyParser::encodeDecode(tree, spot.posY, defaultSpot.posY, base + "pos.y", parserTask);
            PropertyParser::encodeDecode(tree, spot.velX, defaultSpot.velX, base + "vel.x", parserTask);
            PropertyParser::encodeDecode(tree, spot.velY, defaultSpot.velY, base + "vel.y", parserTask);

            PropertyParser::encodeDecode(tree, spot.shapeType, defaultSpot.shapeType, base + "shape.type", parserTask);
            if (spot.shapeType == SpotShapeType_Circular) {
                PropertyParser::encodeDecode(
                    tree,
                    spot.shapeData.circularSpot.coreRadius,
                    defaultSpot.shapeData.circularSpot.coreRadius,
                    base + "shape.circular.core radius",
                    parserTask);
            }
            if (spot.shapeType == SpotShapeType_Rectangular) {
                PropertyParser::encodeDecode(
                    tree, spot.shapeData.rectangularSpot.width, defaultSpot.shapeData.rectangularSpot.width, base + "shape.rectangular.core width", parserTask);
                PropertyParser::encodeDecode(
                    tree,
                    spot.shapeData.rectangularSpot.height,
                    defaultSpot.shapeData.rectangularSpot.height,
                    base + "shape.rectangular.core height",
                    parserTask);
            }
            PropertyParser::encodeDecode(tree, spot.flowType, defaultSpot.flowType, base + "flow.type", parserTask);
            if (spot.flowType == FlowType_Radial) {
                PropertyParser::encodeDecode(
                    tree, spot.flowData.radialFlow.orientation, defaultSpot.flowData.radialFlow.orientation, base + "flow.radial.orientation", parserTask);
                PropertyParser::encodeDecode(
                    tree, spot.flowData.radialFlow.strength, defaultSpot.flowData.radialFlow.strength, base + "flow.radial.strength", parserTask);
                PropertyParser::encodeDecode(
                    tree, spot.flowData.radialFlow.driftAngle, defaultSpot.flowData.radialFlow.driftAngle, base + "flow.radial.drift angle", parserTask);
            }
            if (spot.flowType == FlowType_Central) {
                PropertyParser::encodeDecode(
                    tree, spot.flowData.centralFlow.strength, defaultSpot.flowData.centralFlow.strength, base + "flow.central.strength", parserTask);
            }
            if (spot.flowType == FlowType_Linear) {
                PropertyParser::encodeDecode(tree, spot.flowData.linearFlow.angle, defaultSpot.flowData.linearFlow.angle, base + "flow.linear.angle", parserTask);
                PropertyParser::encodeDecode(
                    tree, spot.flowData.linearFlow.strength, defaultSpot.flowData.linearFlow.strength, base + "flow.linear.strength", parserTask);
            }
            PropertyParser::encodeDecode(tree, spot.fadeoutRadius, defaultSpot.fadeoutRadius, base + "fadeout radius", parserTask);

            PropertyParser::encodeDecodeWithEnabled(tree, spot.values.friction, spot.activatedValues.friction, defaultSpot.values.friction, base + "friction", parserTask);
            PropertyParser::encodeDecodeWithEnabled(tree, spot.values.rigidity, spot.activatedValues.rigidity, defaultSpot.values.rigidity, base + "rigidity", parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationDisableSources,
                spot.activatedValues.radiationDisableSources,
                defaultSpot.values.radiationDisableSources,
                base + "radiation.disable sources",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationAbsorption,
                spot.activatedValues.radiationAbsorption,
                defaultSpot.values.radiationAbsorption,
                base + "radiation.absorption",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationAbsorptionLowVelocityPenalty,
                spot.activatedValues.radiationAbsorptionLowVelocityPenalty,
                defaultSpot.values.radiationAbsorptionLowVelocityPenalty,
                base + "radiation.absorption low velocity penalty",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationAbsorptionLowGenomeComplexityPenalty,
                spot.activatedValues.radiationAbsorptionLowGenomeComplexityPenalty,
                defaultSpot.values.radiationAbsorptionLowGenomeComplexityPenalty,
                base +"radiation.absorption low genome complexity penalty",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationCellAgeStrength,
                spot.activatedValues.radiationCellAgeStrength,
                defaultSpot.values.radiationCellAgeStrength,
                base + "radiation.factor",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree, spot.values.cellMaxForce, spot.activatedValues.cellMaxForce, defaultSpot.values.cellMaxForce, base + "cell.max force", parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree, spot.values.cellMinEnergy, spot.activatedValues.cellMinEnergy, defaultSpot.values.cellMinEnergy, base + "cell.min energy", parserTask);

            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellFusionVelocity,
                spot.activatedValues.cellFusionVelocity,
                defaultSpot.values.cellFusionVelocity,
                base + "cell.fusion velocity",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellMaxBindingEnergy,
                spot.activatedValues.cellMaxBindingEnergy,
                defaultSpot.values.cellMaxBindingEnergy,
                base + "cell.max binding energy",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellInactiveMaxAge,
                spot.activatedValues.cellInactiveMaxAge,
                defaultSpot.values.cellInactiveMaxAge,
                base + "cell.inactive max age",
                parserTask);

            PropertyParser::encodeDecode(tree, spot.activatedValues.cellColorTransition, false, base + "cell.color transition rules.activated", parserTask);
            PropertyParser::encodeDecode(
                tree,
                spot.values.cellColorTransitionDuration,
                defaultSpot.values.cellColorTransitionDuration,
                base + "cell.color transition rules.duration",
                parserTask);
            PropertyParser::encodeDecode(
                tree,
                spot.values.cellColorTransitionTargetColor,
                defaultSpot.values.cellColorTransitionTargetColor,
                base + "cell.color transition rules.target color",
                parserTask);

            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellFunctionAttackerEnergyCost,
                spot.activatedValues.cellFunctionAttackerEnergyCost,
                defaultSpot.values.cellFunctionAttackerEnergyCost,
                base + "cell.function.attacker.energy cost",
                parserTask);

            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellFunctionAttackerFoodChainColorMatrix,
                spot.activatedValues.cellFunctionAttackerFoodChainColorMatrix,
                defaultSpot.values.cellFunctionAttackerFoodChainColorMatrix,
                base + "cell.function.attacker.food chain color matrix",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellFunctionAttackerGenomeComplexityBonus,
                spot.activatedValues.cellFunctionAttackerGenomeComplexityBonus,
                defaultSpot.values.cellFunctionAttackerGenomeComplexityBonus,
                base + "cell.function.attacker.genome size bonus",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellFunctionAttackerNewComplexMutantPenalty,
                spot.activatedValues.cellFunctionAttackerNewComplexMutantPenalty,
                defaultSpot.values.cellFunctionAttackerNewComplexMutantPenalty,
                base + "cell.function.attacker.new complex mutant penalty",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellFunctionAttackerGeometryDeviationExponent,
                spot.activatedValues.cellFunctionAttackerGeometryDeviationExponent,
                defaultSpot.values.cellFunctionAttackerGeometryDeviationExponent,
                base + "cell.function.attacker.geometry deviation exponent",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellFunctionAttackerConnectionsMismatchPenalty,
                spot.activatedValues.cellFunctionAttackerConnectionsMismatchPenalty,
                defaultSpot.values.cellFunctionAttackerConnectionsMismatchPenalty,
                base + "cell.function.attacker.connections mismatch penalty",
                parserTask);

            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationNeuronData,
                spot.activatedValues.cellCopyMutationNeuronData,
                defaultSpot.values.cellCopyMutationNeuronData,
                base + "cell.copy mutation.neuron data",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationCellProperties,
                spot.activatedValues.cellCopyMutationCellProperties,
                defaultSpot.values.cellCopyMutationCellProperties,
                base + "cell.copy mutation.cell properties",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationGeometry,
                spot.activatedValues.cellCopyMutationGeometry,
                defaultSpot.values.cellCopyMutationGeometry,
                base + "cell.copy mutation.geometry",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationCustomGeometry,
                spot.activatedValues.cellCopyMutationCustomGeometry,
                defaultSpot.values.cellCopyMutationCustomGeometry,
                base + "cell.copy mutation.custom geometry",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationCellFunction,
                spot.activatedValues.cellCopyMutationCellFunction,
                defaultSpot.values.cellCopyMutationCellFunction,
                base + "cell.copy mutation.cell function",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationInsertion,
                spot.activatedValues.cellCopyMutationInsertion,
                defaultSpot.values.cellCopyMutationInsertion,
                base + "cell.copy mutation.insertion",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationDeletion,
                spot.activatedValues.cellCopyMutationDeletion,
                defaultSpot.values.cellCopyMutationDeletion,
                base + "cell.copy mutation.deletion",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationTranslation,
                spot.activatedValues.cellCopyMutationTranslation,
                defaultSpot.values.cellCopyMutationTranslation,
                base + "cell.copy mutation.translation",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationDuplication,
                spot.activatedValues.cellCopyMutationDuplication,
                defaultSpot.values.cellCopyMutationDuplication,
                base + "cell.copy mutation.duplication",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationCellColor,
                spot.activatedValues.cellCopyMutationCellColor,
                defaultSpot.values.cellCopyMutationCellColor,
                base + "cell.copy mutation.cell color",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationSubgenomeColor,
                spot.activatedValues.cellCopyMutationSubgenomeColor,
                defaultSpot.values.cellCopyMutationSubgenomeColor,
                base + "cell.copy mutation.subgenome color",
                parserTask);
            PropertyParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationGenomeColor,
                spot.activatedValues.cellCopyMutationGenomeColor,
                defaultSpot.values.cellCopyMutationGenomeColor,
                base + "cell.copy mutation.genome color",
                parserTask);
        }

        //features
        PropertyParser::encodeDecode(
            tree,
            parameters.features.genomeComplexityMeasurement,
            defaultParameters.features.genomeComplexityMeasurement,
            "simulation parameters.features.genome complexity measurement",
            parserTask);
        missingFeatures.advancedAbsorptionControl = PropertyParser::encodeDecode(
            tree,
            parameters.features.advancedAbsorptionControl,
            defaultParameters.features.advancedAbsorptionControl,
            "simulation parameters.features.additional absorption control",
            parserTask);
        missingFeatures.advancedAttackerControl = PropertyParser::encodeDecode(
            tree,
            parameters.features.advancedAttackerControl,
            defaultParameters.features.advancedAttackerControl,
            "simulation parameters.features.additional attacker control",
            parserTask);
        missingFeatures.externalEnergyControl = PropertyParser::encodeDecode(
            tree, parameters.features.externalEnergyControl, defaultParameters.features.externalEnergyControl, "simulation parameters.features.external energy", parserTask);
        missingFeatures.cellColorTransitionRules = PropertyParser::encodeDecode(
            tree,
            parameters.features.cellColorTransitionRules,
            defaultParameters.features.cellColorTransitionRules,
            "simulation parameters.features.cell color transition rules",
            parserTask);
        missingFeatures.cellAgeLimiter = PropertyParser::encodeDecode(
            tree,
            parameters.features.cellAgeLimiter,
            defaultParameters.features.cellAgeLimiter,
            "simulation parameters.features.cell age limiter",
            parserTask);
        PropertyParser::encodeDecode(
            tree,
            parameters.features.cellGlow,
            defaultParameters.features.cellGlow,
            "simulation parameters.features.cell glow",
            parserTask);
        missingFeatures.legacyMode = PropertyParser::encodeDecode(
            tree, parameters.features.legacyModes, defaultParameters.features.legacyModes, "simulation parameters.features.legacy modes", parserTask);
    }

    void encodeDecodeSimulationParameters(boost::property_tree::ptree& tree, SimulationParameters& parameters, ParserTask parserTask)
    {
        MissingParameters missingParameters;
        MissingFeatures missingFeatures;
        encodeDecode(tree, parameters, missingParameters, missingFeatures, parserTask);

        // Compatibility with legacy parameters
        if (parserTask == ParserTask::Decode) {
            LegacyAuxiliaryDataParserService::searchAndApplyLegacyParameters(tree, missingFeatures, missingParameters, parameters);
        }
    }

    void encodeDecode(boost::property_tree::ptree& tree, AuxiliaryData& data, ParserTask parserTask)
    {
        AuxiliaryData defaultSettings;

        //general settings
        PropertyParser::encodeDecode(tree, data.timestep, uint64_t(0), "general.time step", parserTask);
        PropertyParser::encodeDecode(tree, data.realTime, std::chrono::milliseconds(0), "general.real time", parserTask);
        PropertyParser::encodeDecode(tree, data.zoom, 4.0f, "general.zoom", parserTask);
        PropertyParser::encodeDecode(tree, data.center.x, 0.0f, "general.center.x", parserTask);
        PropertyParser::encodeDecode(tree, data.center.y, 0.0f, "general.center.y", parserTask);
        PropertyParser::encodeDecode(tree, data.generalSettings.worldSizeX, defaultSettings.generalSettings.worldSizeX, "general.world size.x", parserTask);
        PropertyParser::encodeDecode(tree, data.generalSettings.worldSizeY, defaultSettings.generalSettings.worldSizeY, "general.world size.y", parserTask);

        encodeDecodeSimulationParameters(tree, data.simulationParameters, parserTask);
    }
}

boost::property_tree::ptree AuxiliaryDataParserService::encodeAuxiliaryData(AuxiliaryData const& data)
{
    boost::property_tree::ptree tree;
    encodeDecode(tree, const_cast<AuxiliaryData&>(data), ParserTask::Encode);
    return tree;
}

AuxiliaryData AuxiliaryDataParserService::decodeAuxiliaryData(boost::property_tree::ptree tree)
{
    AuxiliaryData result;
    encodeDecode(tree, result, ParserTask::Decode);
    return result;
}

boost::property_tree::ptree AuxiliaryDataParserService::encodeSimulationParameters(SimulationParameters const& data)
{
    boost::property_tree::ptree tree;
    encodeDecodeSimulationParameters(tree, const_cast<SimulationParameters&>(data), ParserTask::Encode);
    return tree;
}

SimulationParameters AuxiliaryDataParserService::decodeSimulationParameters(boost::property_tree::ptree tree)
{
    SimulationParameters result;
    encodeDecodeSimulationParameters(tree, result, ParserTask::Decode);
    return result;
}
