#include "SettingsParserService.h"

#include "Base/Resources.h"
#include "EngineInterface/SettingsForSimulation.h"

#include "LegacySettingsParserService.h"
#include "ParameterParser.h"

namespace
{
    void encodeDecodeLatestSimulationParameters(
        boost::property_tree::ptree& tree,
        SimulationParameters& parameters,
        ParserTask parserTask)
    {
        SimulationParameters defaultParameters;
        ParameterParser::encodeDecode(tree, parameters.projectName, defaultParameters.projectName, "simulation parameters.project name", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.backgroundColor, defaultParameters.backgroundColor, "simulation parameters.background color", parserTask);
        ParameterParser::encodeDecode(tree, parameters.cellColoring, defaultParameters.cellColoring, "simulation parameters.cell colorization", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellGlowColoring, defaultParameters.cellGlowColoring, "simulation parameters.cell glow.coloring", parserTask);
        ParameterParser::encodeDecode(tree, parameters.cellGlowRadius, defaultParameters.cellGlowRadius, "simulation parameters.cell glow.radius", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellGlowStrength, defaultParameters.cellGlowStrength, "simulation parameters.cell glow.strength", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.highlightedCellType, defaultParameters.highlightedCellType, "simulation parameters.highlighted cell function", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.zoomLevelNeuronalActivity,
            defaultParameters.zoomLevelNeuronalActivity,
            "simulation parameters.zoom level.neural activity",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.borderlessRendering, defaultParameters.borderlessRendering, "simulation parameters.borderless rendering", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.markReferenceDomain, defaultParameters.markReferenceDomain, "simulation parameters.mark reference domain", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.showRadiationSources, defaultParameters.showRadiationSources, "simulation parameters.show radiation sources", parserTask);
        ParameterParser::encodeDecode(tree, parameters.gridLines, defaultParameters.gridLines, "simulation parameters.grid lines", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.attackVisualization, defaultParameters.attackVisualization, "simulation parameters.attack visualization", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.muscleMovementVisualization,
            defaultParameters.muscleMovementVisualization,
            "simulation parameters.muscle movement visualization",
            parserTask);
        ParameterParser::encodeDecode(tree, parameters.cellRadius, defaultParameters.cellRadius, "simulation parameters.cek", parserTask);

        ParameterParser::encodeDecode(tree, parameters.timestepSize, defaultParameters.timestepSize, "simulation parameters.time step size", parserTask);

        ParameterParser::encodeDecode(tree, parameters.motionData.type, defaultParameters.motionData.type, "simulation parameters.motion.type", parserTask);
        if (parameters.motionData.type == MotionType_Fluid) {
            ParameterParser::encodeDecode(
                tree,
                parameters.motionData.alternatives.fluidMotion.smoothingLength,
                defaultParameters.motionData.alternatives.fluidMotion.smoothingLength,
                "simulation parameters.fluid.smoothing length",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.motionData.alternatives.fluidMotion.pressureStrength,
                defaultParameters.motionData.alternatives.fluidMotion.pressureStrength,
                "simulation parameters.fluid.pressure strength",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.motionData.alternatives.fluidMotion.viscosityStrength,
                defaultParameters.motionData.alternatives.fluidMotion.viscosityStrength,
                "simulation parameters.fluid.viscosity strength",
                parserTask);
        } else {
            ParameterParser::encodeDecode(
                tree,
                parameters.motionData.alternatives.collisionMotion.cellMaxCollisionDistance,
                defaultParameters.motionData.alternatives.collisionMotion.cellMaxCollisionDistance,
                "simulation parameters.motion.collision.max distance",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.motionData.alternatives.collisionMotion.cellRepulsionStrength,
                defaultParameters.motionData.alternatives.collisionMotion.cellRepulsionStrength,
                "simulation parameters.motion.collision.repulsion strength",
                parserTask);
        }

        ParameterParser::encodeDecode(
            tree, parameters.baseValues.friction, defaultParameters.baseValues.friction, "simulation parameters.friction", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.baseValues.rigidity, defaultParameters.baseValues.rigidity, "simulation parameters.rigidity", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellMaxVelocity, defaultParameters.cellMaxVelocity, "simulation parameters.cell.max velocity", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellMaxBindingDistance, defaultParameters.cellMaxBindingDistance, "simulation parameters.cell.max binding distance", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellNormalEnergy, defaultParameters.cellNormalEnergy, "simulation parameters.cell.normal energy", parserTask);

        ParameterParser::encodeDecode(
            tree, parameters.cellMinDistance, defaultParameters.cellMinDistance, "simulation parameters.cell.min distance", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.baseValues.cellMaxForce, defaultParameters.baseValues.cellMaxForce, "simulation parameters.cell.max force", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellMaxForceDecayProb,
            defaultParameters.cellMaxForceDecayProb,
            "simulation parameters.cell.max force decay probability",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.baseValues.cellMinEnergy, defaultParameters.baseValues.cellMinEnergy, "simulation parameters.cell.min energy", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellFusionVelocity,
            defaultParameters.baseValues.cellFusionVelocity,
            "simulation parameters.cell.fusion velocity",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellMaxBindingEnergy,
            parameters.baseValues.cellMaxBindingEnergy,
            "simulation parameters.cell.max binding energy",
            parserTask);
        ParameterParser::encodeDecode(tree, parameters.cellMaxAge, defaultParameters.cellMaxAge, "simulation parameters.cell.max age", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellMaxAgeBalancer, defaultParameters.cellMaxAgeBalancer, "simulation parameters.cell.max age.balance.enabled", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellMaxAgeBalancerInterval,
            defaultParameters.cellMaxAgeBalancerInterval,
            "simulation parameters.cell.max age.balance.interval",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellInactiveMaxAgeActivated,
            defaultParameters.cellInactiveMaxAgeActivated,
            "simulation parameters.cell.inactive max age activated",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellInactiveMaxAge,
            defaultParameters.baseValues.cellInactiveMaxAge,
            "simulation parameters.cell.inactive max age",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellEmergentMaxAgeActivated,
            defaultParameters.cellEmergentMaxAgeActivated,
            "simulation parameters.cell.nutrient max age activated",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellEmergentMaxAge, defaultParameters.cellEmergentMaxAge, "simulation parameters.cell.nutrient max age", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellResetAgeAfterActivation,
            defaultParameters.cellResetAgeAfterActivation,
            "simulation parameters.cell.reset age after activation",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellColorTransitionDuration,
            defaultParameters.baseValues.cellColorTransitionDuration,
            "simulation parameters.cell.color transition rules.duration",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellColorTransitionTargetColor,
            defaultParameters.baseValues.cellColorTransitionTargetColor,
            "simulation parameters.cell.color transition rules.target color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.genomeComplexityRamificationFactor,
            defaultParameters.genomeComplexityRamificationFactor,
            "simulation parameters.genome complexity.genome complexity ramification factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.genomeComplexitySizeFactor,
            defaultParameters.genomeComplexitySizeFactor,
            "simulation parameters.genome complexity.genome complexity size factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.genomeComplexityNeuronFactor,
            defaultParameters.genomeComplexityNeuronFactor,
            "simulation parameters.genome complexity.genome complexity neuron factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.genomeComplexityDepthLevel,
            defaultParameters.genomeComplexityDepthLevel,
            "simulation parameters.genome complexity.genome complexity depth level",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.radiationCellAgeStrength,
            defaultParameters.baseValues.radiationCellAgeStrength,
            "simulation parameters.radiation.factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.radiationProb, defaultParameters.radiationProb, "simulation parameters.radiation.probability", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationVelocityMultiplier,
            defaultParameters.radiationVelocityMultiplier,
            "simulation parameters.radiation.velocity multiplier",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationVelocityPerturbation,
            defaultParameters.radiationVelocityPerturbation,
            "simulation parameters.radiation.velocity perturbation",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.radiationDisableSources,
            defaultParameters.baseValues.radiationDisableSources,
            "simulation parameters.radiation.disable sources",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.radiationAbsorption,
            defaultParameters.baseValues.radiationAbsorption,
            "simulation parameters.radiation.absorption",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationAbsorptionHighVelocityPenalty,
            defaultParameters.radiationAbsorptionHighVelocityPenalty,
            "simulation parameters.radiation.absorption velocity penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.radiationAbsorptionLowVelocityPenalty,
            defaultParameters.baseValues.radiationAbsorptionLowVelocityPenalty,
            "simulation parameters.radiation.absorption low velocity penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationAbsorptionLowConnectionPenalty,
            defaultParameters.radiationAbsorptionLowConnectionPenalty,
            "simulation parameters.radiation.absorption low connection penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty,
            defaultParameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty,
            "simulation parameters.radiation.absorption low genome complexity penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.highRadiationMinCellEnergy,
            defaultParameters.highRadiationMinCellEnergy,
            "simulation parameters.high radiation.min cell energy",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.highRadiationFactor, defaultParameters.highRadiationFactor, "simulation parameters.high radiation.factor", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.radiationMinCellAge, defaultParameters.radiationMinCellAge, "simulation parameters.radiation.min cell age", parserTask);

        ParameterParser::encodeDecode(
            tree, parameters.externalEnergy, defaultParameters.externalEnergy, "simulation parameters.cell.function.constructor.external energy", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyInflowFactor,
            defaultParameters.externalEnergyInflowFactor,
            "simulation parameters.cell.function.constructor.external energy supply rate",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyConditionalInflowFactor,
            defaultParameters.externalEnergyConditionalInflowFactor,
            "simulation parameters.cell.function.constructor.pump energy factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyBackflowFactor,
            defaultParameters.externalEnergyBackflowFactor,
            "simulation parameters.cell.function.constructor.external energy backflow",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyInflowOnlyForNonSelfReplicators,
            defaultParameters.externalEnergyInflowOnlyForNonSelfReplicators,
            "simulation parameters.cell.function.constructor.external energy inflow only for non-self-replicators",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyBackflowLimit,
            defaultParameters.externalEnergyBackflowLimit,
            "simulation parameters.cell.function.constructor.external energy backflow limit",
            parserTask);

        ParameterParser::encodeDecode(
            tree, parameters.cellDeathConsequences, defaultParameters.cellDeathConsequences, "simulation parameters.cell.death consequences", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellDeathProbability,
            defaultParameters.baseValues.cellDeathProbability,
            "simulation parameters.cell.death probability",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeConstructorConnectingCellMaxDistance,
            defaultParameters.cellTypeConstructorConnectingCellMaxDistance,
            "simulation parameters.cell.function.constructor.connecting cell max distance",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeConstructorAdditionalOffspringDistance,
            defaultParameters.cellTypeConstructorAdditionalOffspringDistance,
            "simulation parameters.cell.function.constructor.additional offspring distance",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeConstructorCheckCompletenessForSelfReplication,
            defaultParameters.cellTypeConstructorCheckCompletenessForSelfReplication,
            "simulation parameters.cell.function.constructor.completeness check for self-replication",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationNeuronData,
            defaultParameters.baseValues.cellCopyMutationNeuronData,
            "simulation parameters.cell.copy mutation.neuron data",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataWeight,
            defaultParameters.cellCopyMutationNeuronDataWeight,
            "simulation parameters.cell.copy mutation.neuron data.weights",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataBias,
            defaultParameters.cellCopyMutationNeuronDataBias,
            "simulation parameters.cell.copy mutation.neuron data.biases",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataActivationFunction,
            defaultParameters.cellCopyMutationNeuronDataActivationFunction,
            "simulation parameters.cell.copy mutation.neuron data.activation functions",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataReinforcement,
            defaultParameters.cellCopyMutationNeuronDataReinforcement,
            "simulation parameters.cell.copy mutation.neuron data.reinforcement",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataDamping,
            defaultParameters.cellCopyMutationNeuronDataDamping,
            "simulation parameters.cell.copy mutation.neuron data.damping",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataOffset,
            defaultParameters.cellCopyMutationNeuronDataOffset,
            "simulation parameters.cell.copy mutation.neuron data.offset",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationCellProperties,
            defaultParameters.baseValues.cellCopyMutationCellProperties,
            "simulation parameters.cell.copy mutation.cell properties",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationGeometry,
            defaultParameters.baseValues.cellCopyMutationGeometry,
            "simulation parameters.cell.copy mutation.geometry",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationCustomGeometry,
            defaultParameters.baseValues.cellCopyMutationCustomGeometry,
            "simulation parameters.cell.copy mutation.custom geometry",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationCellType,
            defaultParameters.baseValues.cellCopyMutationCellType,
            "simulation parameters.cell.copy mutation.cell function",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationInsertion,
            defaultParameters.baseValues.cellCopyMutationInsertion,
            "simulation parameters.cell.copy mutation.insertion",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationDeletion,
            defaultParameters.baseValues.cellCopyMutationDeletion,
            "simulation parameters.cell.copy mutation.deletion",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationDeletionMinSize,
            defaultParameters.cellCopyMutationDeletionMinSize,
            "simulation parameters.cell.copy mutation.deletion.min size",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationTranslation,
            defaultParameters.baseValues.cellCopyMutationTranslation,
            "simulation parameters.cell.copy mutation.translation",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationDuplication,
            defaultParameters.baseValues.cellCopyMutationDuplication,
            "simulation parameters.cell.copy mutation.duplication",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationCellColor,
            defaultParameters.baseValues.cellCopyMutationCellColor,
            "simulation parameters.cell.copy mutation.cell color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationSubgenomeColor,
            defaultParameters.baseValues.cellCopyMutationSubgenomeColor,
            "simulation parameters.cell.copy mutation.subgenome color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellCopyMutationGenomeColor,
            defaultParameters.baseValues.cellCopyMutationGenomeColor,
            "simulation parameters.cell.copy mutation.genome color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationColorTransitions,
            defaultParameters.cellCopyMutationColorTransitions,
            "simulation parameters.cell.copy mutation.color transition",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationSelfReplication,
            defaultParameters.cellCopyMutationSelfReplication,
            "simulation parameters.cell.copy mutation.self replication flag",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationPreventDepthIncrease,
            defaultParameters.cellCopyMutationPreventDepthIncrease,
            "simulation parameters.cell.copy mutation.prevent depth increase",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeInjectorRadius,
            defaultParameters.cellTypeInjectorRadius,
            "simulation parameters.cell.function.injector.radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeInjectorDurationColorMatrix,
            defaultParameters.cellTypeInjectorDurationColorMatrix,
            "simulation parameters.cell.function.injector.duration",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeAttackerRadius,
            defaultParameters.cellTypeAttackerRadius,
            "simulation parameters.cell.function.attacker.radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeAttackerStrength,
            defaultParameters.cellTypeAttackerStrength,
            "simulation parameters.cell.function.attacker.strength",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeAttackerEnergyDistributionRadius,
            defaultParameters.cellTypeAttackerEnergyDistributionRadius,
            "simulation parameters.cell.function.attacker.energy distribution radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeAttackerEnergyDistributionValue,
            defaultParameters.cellTypeAttackerEnergyDistributionValue,
            "simulation parameters.cell.function.attacker.energy distribution value",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeAttackerColorInhomogeneityFactor,
            defaultParameters.cellTypeAttackerColorInhomogeneityFactor,
            "simulation parameters.cell.function.attacker.color inhomogeneity factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellTypeAttackerEnergyCost,
            defaultParameters.baseValues.cellTypeAttackerEnergyCost,
            "simulation parameters.cell.function.attacker.energy cost",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellTypeAttackerGeometryDeviationExponent,
            defaultParameters.baseValues.cellTypeAttackerGeometryDeviationExponent,
            "simulation parameters.cell.function.attacker.geometry deviation exponent",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellTypeAttackerFoodChainColorMatrix,
            defaultParameters.baseValues.cellTypeAttackerFoodChainColorMatrix,
            "simulation parameters.cell.function.attacker.food chain color matrix",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellTypeAttackerConnectionsMismatchPenalty,
            defaultParameters.baseValues.cellTypeAttackerConnectionsMismatchPenalty,
            "simulation parameters.cell.function.attacker.connections mismatch penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellTypeAttackerGenomeComplexityBonus,
            defaultParameters.baseValues.cellTypeAttackerGenomeComplexityBonus,
            "simulation parameters.cell.function.attacker.genome size bonus",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeAttackerSameMutantPenalty,
            defaultParameters.cellTypeAttackerSameMutantPenalty,
            "simulation parameters.cell.function.attacker.same mutant penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.cellTypeAttackerNewComplexMutantPenalty,
            defaultParameters.baseValues.cellTypeAttackerNewComplexMutantPenalty,
            "simulation parameters.cell.function.attacker.new complex mutant penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeAttackerSensorDetectionFactor,
            defaultParameters.cellTypeAttackerSensorDetectionFactor,
            "simulation parameters.cell.function.attacker.sensor detection factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeAttackerDestroyCells,
            defaultParameters.cellTypeAttackerDestroyCells,
            "simulation parameters.cell.function.attacker.destroy cells",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeDefenderAgainstAttackerStrength,
            defaultParameters.cellTypeDefenderAgainstAttackerStrength,
            "simulation parameters.cell.function.defender.against attacker strength",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeDefenderAgainstInjectorStrength,
            defaultParameters.cellTypeDefenderAgainstInjectorStrength,
            "simulation parameters.cell.function.defender.against injector strength",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeTransmitterEnergyDistributionSameCreature,
            defaultParameters.cellTypeTransmitterEnergyDistributionSameCreature,
            "simulation parameters.cell.function.transmitter.energy distribution same creature",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeTransmitterEnergyDistributionRadius,
            defaultParameters.cellTypeTransmitterEnergyDistributionRadius,
            "simulation parameters.cell.function.transmitter.energy distribution radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeTransmitterEnergyDistributionValue,
            defaultParameters.cellTypeTransmitterEnergyDistributionValue,
            "simulation parameters.cell.function.transmitter.energy distribution value",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeMuscleContractionExpansionDelta,
            defaultParameters.cellTypeMuscleContractionExpansionDelta,
            "simulation parameters.cell.function.muscle.contraction expansion delta",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeMuscleMovementAcceleration,
            defaultParameters.cellTypeMuscleMovementAcceleration,
            "simulation parameters.cell.function.muscle.movement acceleration",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeMuscleBendingAngle,
            defaultParameters.cellTypeMuscleBendingAngle,
            "simulation parameters.cell.function.muscle.bending angle",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeMuscleBendingAcceleration,
            defaultParameters.cellTypeMuscleBendingAcceleration,
            "simulation parameters.cell.function.muscle.bending acceleration",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeMuscleMovementTowardTargetedObject,
            defaultParameters.cellTypeMuscleMovementTowardTargetedObject,
            "simulation parameters.cell.function.muscle.movement toward targeted object",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeMuscleEnergyCost,
            defaultParameters.cellTypeMuscleEnergyCost,
            "simulation parameters.cell.function.muscle.energy cost",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.particleTransformationAllowed,
            defaultParameters.particleTransformationAllowed,
            "simulation parameters.particle.transformation allowed",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.particleSplitEnergy, defaultParameters.particleSplitEnergy, "simulation parameters.particle.split energy", parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeSensorRange,
            defaultParameters.cellTypeSensorRange,
            "simulation parameters.cell.function.sensor.range",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeReconnectorRadius,
            defaultParameters.cellTypeReconnectorRadius,
            "simulation parameters.cell.function.reconnector.radius",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeDetonatorRadius,
            defaultParameters.cellTypeDetonatorRadius,
            "simulation parameters.cell.function.detonator.radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellTypeDetonatorChainExplosionProbability,
            defaultParameters.cellTypeDetonatorChainExplosionProbability,
            "simulation parameters.cell.function.detonator.chain explosion probability",
            parserTask);

        //particle sources
        ParameterParser::encodeDecode(
            tree, parameters.numRadiationSources, defaultParameters.numRadiationSources, "simulation parameters.particle sources.num sources", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseStrengthRatioPinned,
            defaultParameters.baseStrengthRatioPinned,
            "simulation parameters.particle sources.base strength pinned",
            parserTask);
        for (int index = 0; index < parameters.numRadiationSources; ++index) {
            std::string base = "simulation parameters.particle sources." + std::to_string(index) + ".";
            auto& source = parameters.radiationSource[index];
            auto& defaultSource = defaultParameters.radiationSource[index];
            ParameterParser::encodeDecode(tree, source.name, defaultSource.name, base + "name", parserTask);
            ParameterParser::encodeDecode(tree, source.locationIndex, defaultSource.locationIndex, base + "location index", parserTask);
            ParameterParser::encodeDecode(tree, source.posX, defaultSource.posX, base + "pos.x", parserTask);
            ParameterParser::encodeDecode(tree, source.posY, defaultSource.posY, base + "pos.y", parserTask);
            ParameterParser::encodeDecode(tree, source.velX, defaultSource.velX, base + "vel.x", parserTask);
            ParameterParser::encodeDecode(tree, source.velY, defaultSource.velY, base + "vel.y", parserTask);
            ParameterParser::encodeDecode(tree, source.useAngle, defaultSource.useAngle, base + "use angle", parserTask);
            ParameterParser::encodeDecode(tree, source.strength, defaultSource.strength, base + "strength", parserTask);
            ParameterParser::encodeDecode(tree, source.strengthPinned, defaultSource.strengthPinned, base + "strength pinned", parserTask);
            ParameterParser::encodeDecode(tree, source.angle, defaultSource.angle, base + "angle", parserTask);
            ParameterParser::encodeDecode(tree, source.shape.type, defaultSource.shape.type, base + "shape.type", parserTask);
            if (source.shape.type == ZoneShapeType_Circular) {
                ParameterParser::encodeDecode(
                    tree,
                    source.shape.alternatives.circularRadiationSource.radius,
                    defaultSource.shape.alternatives.circularRadiationSource.radius,
                    base + "shape.circular.radius",
                    parserTask);
            }
            if (source.shape.type == ZoneShapeType_Rectangular) {
                ParameterParser::encodeDecode(
                    tree,
                    source.shape.alternatives.rectangularRadiationSource.width,
                    defaultSource.shape.alternatives.rectangularRadiationSource.width,
                    base + "shape.rectangular.width",
                    parserTask);
                ParameterParser::encodeDecode(
                    tree,
                    source.shape.alternatives.rectangularRadiationSource.height,
                    defaultSource.shape.alternatives.rectangularRadiationSource.height,
                    base + "shape.rectangular.height",
                    parserTask);
            }
        }

        //spots
        ParameterParser::encodeDecode(tree, parameters.numZones, defaultParameters.numZones, "simulation parameters.spots.num spots", parserTask);
        for (int index = 0; index < parameters.numZones; ++index) {
            std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
            auto& spot = parameters.zone[index];
            auto& defaultSpot = defaultParameters.zone[index];
            ParameterParser::encodeDecode(tree, spot.name, defaultSpot.name, base + "name", parserTask);
            ParameterParser::encodeDecode(tree, spot.locationIndex, defaultSpot.locationIndex, base + "location index", parserTask);
            ParameterParser::encodeDecode(tree, spot.color, defaultSpot.color, base + "color", parserTask);
            ParameterParser::encodeDecode(tree, spot.posX, defaultSpot.posX, base + "pos.x", parserTask);
            ParameterParser::encodeDecode(tree, spot.posY, defaultSpot.posY, base + "pos.y", parserTask);
            ParameterParser::encodeDecode(tree, spot.velX, defaultSpot.velX, base + "vel.x", parserTask);
            ParameterParser::encodeDecode(tree, spot.velY, defaultSpot.velY, base + "vel.y", parserTask);

            ParameterParser::encodeDecode(tree, spot.shape.type, defaultSpot.shape.type, base + "shape.type", parserTask);
            if (spot.shape.type == ZoneShapeType_Circular) {
                ParameterParser::encodeDecode(
                    tree,
                    spot.shape.alternatives.circularSpot.coreRadius,
                    defaultSpot.shape.alternatives.circularSpot.coreRadius,
                    base + "shape.circular.core radius",
                    parserTask);
            }
            if (spot.shape.type == ZoneShapeType_Rectangular) {
                ParameterParser::encodeDecode(
                    tree,
                    spot.shape.alternatives.rectangularSpot.width,
                    defaultSpot.shape.alternatives.rectangularSpot.width,
                    base + "shape.rectangular.core width",
                    parserTask);
                ParameterParser::encodeDecode(
                    tree,
                    spot.shape.alternatives.rectangularSpot.height,
                    defaultSpot.shape.alternatives.rectangularSpot.height,
                    base + "shape.rectangular.core height",
                    parserTask);
            }
            ParameterParser::encodeDecode(tree, spot.flow.type, defaultSpot.flow.type, base + "flow.type", parserTask);
            if (spot.flow.type == FlowType_Radial) {
                ParameterParser::encodeDecode(
                    tree, spot.flow.alternatives.radialFlow.orientation, defaultSpot.flow.alternatives.radialFlow.orientation, base + "flow.radial.orientation", parserTask);
                ParameterParser::encodeDecode(
                    tree, spot.flow.alternatives.radialFlow.strength, defaultSpot.flow.alternatives.radialFlow.strength, base + "flow.radial.strength", parserTask);
                ParameterParser::encodeDecode(
                    tree, spot.flow.alternatives.radialFlow.driftAngle, defaultSpot.flow.alternatives.radialFlow.driftAngle, base + "flow.radial.drift angle", parserTask);
            }
            if (spot.flow.type == FlowType_Central) {
                ParameterParser::encodeDecode(
                    tree, spot.flow.alternatives.centralFlow.strength, defaultSpot.flow.alternatives.centralFlow.strength, base + "flow.central.strength", parserTask);
            }
            if (spot.flow.type == FlowType_Linear) {
                ParameterParser::encodeDecode(
                    tree, spot.flow.alternatives.linearFlow.angle, defaultSpot.flow.alternatives.linearFlow.angle, base + "flow.linear.angle", parserTask);
                ParameterParser::encodeDecode(
                    tree, spot.flow.alternatives.linearFlow.strength, defaultSpot.flow.alternatives.linearFlow.strength, base + "flow.linear.strength", parserTask);
            }
            ParameterParser::encodeDecode(tree, spot.fadeoutRadius, defaultSpot.fadeoutRadius, base + "fadeout radius", parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree, spot.values.friction, spot.activatedValues.friction, defaultSpot.values.friction, base + "friction", parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree, spot.values.rigidity, spot.activatedValues.rigidity, defaultSpot.values.rigidity, base + "rigidity", parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationDisableSources,
                spot.activatedValues.radiationDisableSources,
                defaultSpot.values.radiationDisableSources,
                base + "radiation.disable sources",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationAbsorption,
                spot.activatedValues.radiationAbsorption,
                defaultSpot.values.radiationAbsorption,
                base + "radiation.absorption",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationAbsorptionLowVelocityPenalty,
                spot.activatedValues.radiationAbsorptionLowVelocityPenalty,
                defaultSpot.values.radiationAbsorptionLowVelocityPenalty,
                base + "radiation.absorption low velocity penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationAbsorptionLowGenomeComplexityPenalty,
                spot.activatedValues.radiationAbsorptionLowGenomeComplexityPenalty,
                defaultSpot.values.radiationAbsorptionLowGenomeComplexityPenalty,
                base + "radiation.absorption low genome complexity penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.radiationCellAgeStrength,
                spot.activatedValues.radiationCellAgeStrength,
                defaultSpot.values.radiationCellAgeStrength,
                base + "radiation.factor",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree, spot.values.cellMaxForce, spot.activatedValues.cellMaxForce, defaultSpot.values.cellMaxForce, base + "cell.max force", parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree, spot.values.cellMinEnergy, spot.activatedValues.cellMinEnergy, defaultSpot.values.cellMinEnergy, base + "cell.min energy", parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellDeathProbability,
                spot.activatedValues.cellDeathProbability,
                defaultSpot.values.cellDeathProbability,
                base + "cell.death probability",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellFusionVelocity,
                spot.activatedValues.cellFusionVelocity,
                defaultSpot.values.cellFusionVelocity,
                base + "cell.fusion velocity",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellMaxBindingEnergy,
                spot.activatedValues.cellMaxBindingEnergy,
                defaultSpot.values.cellMaxBindingEnergy,
                base + "cell.max binding energy",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellInactiveMaxAge,
                spot.activatedValues.cellInactiveMaxAge,
                defaultSpot.values.cellInactiveMaxAge,
                base + "cell.inactive max age",
                parserTask);

            ParameterParser::encodeDecode(tree, spot.activatedValues.cellColorTransition, false, base + "cell.color transition rules.activated", parserTask);
            ParameterParser::encodeDecode(
                tree,
                spot.values.cellColorTransitionDuration,
                defaultSpot.values.cellColorTransitionDuration,
                base + "cell.color transition rules.duration",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                spot.values.cellColorTransitionTargetColor,
                defaultSpot.values.cellColorTransitionTargetColor,
                base + "cell.color transition rules.target color",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellTypeAttackerEnergyCost,
                spot.activatedValues.cellTypeAttackerEnergyCost,
                defaultSpot.values.cellTypeAttackerEnergyCost,
                base + "cell.function.attacker.energy cost",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellTypeAttackerFoodChainColorMatrix,
                spot.activatedValues.cellTypeAttackerFoodChainColorMatrix,
                defaultSpot.values.cellTypeAttackerFoodChainColorMatrix,
                base + "cell.function.attacker.food chain color matrix",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellTypeAttackerGenomeComplexityBonus,
                spot.activatedValues.cellTypeAttackerGenomeComplexityBonus,
                defaultSpot.values.cellTypeAttackerGenomeComplexityBonus,
                base + "cell.function.attacker.genome size bonus",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellTypeAttackerNewComplexMutantPenalty,
                spot.activatedValues.cellTypeAttackerNewComplexMutantPenalty,
                defaultSpot.values.cellTypeAttackerNewComplexMutantPenalty,
                base + "cell.function.attacker.new complex mutant penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellTypeAttackerGeometryDeviationExponent,
                spot.activatedValues.cellTypeAttackerGeometryDeviationExponent,
                defaultSpot.values.cellTypeAttackerGeometryDeviationExponent,
                base + "cell.function.attacker.geometry deviation exponent",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellTypeAttackerConnectionsMismatchPenalty,
                spot.activatedValues.cellTypeAttackerConnectionsMismatchPenalty,
                defaultSpot.values.cellTypeAttackerConnectionsMismatchPenalty,
                base + "cell.function.attacker.connections mismatch penalty",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationNeuronData,
                spot.activatedValues.cellCopyMutationNeuronData,
                defaultSpot.values.cellCopyMutationNeuronData,
                base + "cell.copy mutation.neuron data",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationCellProperties,
                spot.activatedValues.cellCopyMutationCellProperties,
                defaultSpot.values.cellCopyMutationCellProperties,
                base + "cell.copy mutation.cell properties",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationGeometry,
                spot.activatedValues.cellCopyMutationGeometry,
                defaultSpot.values.cellCopyMutationGeometry,
                base + "cell.copy mutation.geometry",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationCustomGeometry,
                spot.activatedValues.cellCopyMutationCustomGeometry,
                defaultSpot.values.cellCopyMutationCustomGeometry,
                base + "cell.copy mutation.custom geometry",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationCellType,
                spot.activatedValues.cellCopyMutationCellType,
                defaultSpot.values.cellCopyMutationCellType,
                base + "cell.copy mutation.cell function",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationInsertion,
                spot.activatedValues.cellCopyMutationInsertion,
                defaultSpot.values.cellCopyMutationInsertion,
                base + "cell.copy mutation.insertion",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationDeletion,
                spot.activatedValues.cellCopyMutationDeletion,
                defaultSpot.values.cellCopyMutationDeletion,
                base + "cell.copy mutation.deletion",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationTranslation,
                spot.activatedValues.cellCopyMutationTranslation,
                defaultSpot.values.cellCopyMutationTranslation,
                base + "cell.copy mutation.translation",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationDuplication,
                spot.activatedValues.cellCopyMutationDuplication,
                defaultSpot.values.cellCopyMutationDuplication,
                base + "cell.copy mutation.duplication",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationCellColor,
                spot.activatedValues.cellCopyMutationCellColor,
                defaultSpot.values.cellCopyMutationCellColor,
                base + "cell.copy mutation.cell color",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationSubgenomeColor,
                spot.activatedValues.cellCopyMutationSubgenomeColor,
                defaultSpot.values.cellCopyMutationSubgenomeColor,
                base + "cell.copy mutation.subgenome color",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.cellCopyMutationGenomeColor,
                spot.activatedValues.cellCopyMutationGenomeColor,
                defaultSpot.values.cellCopyMutationGenomeColor,
                base + "cell.copy mutation.genome color",
                parserTask);
        }

        //features
        ParameterParser::encodeDecode(
            tree,
            parameters.features.genomeComplexityMeasurement,
            defaultParameters.features.genomeComplexityMeasurement,
            "simulation parameters.features.genome complexity measurement",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.features.advancedAbsorptionControl,
            defaultParameters.features.advancedAbsorptionControl,
            "simulation parameters.features.additional absorption control",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.features.advancedAttackerControl,
            defaultParameters.features.advancedAttackerControl,
            "simulation parameters.features.additional attacker control",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.features.externalEnergyControl,
            defaultParameters.features.externalEnergyControl,
            "simulation parameters.features.external energy",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.features.cellColorTransitionRules,
            defaultParameters.features.cellColorTransitionRules,
            "simulation parameters.features.cell color transition rules",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.features.cellAgeLimiter, defaultParameters.features.cellAgeLimiter, "simulation parameters.features.cell age limiter", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.features.cellGlow, defaultParameters.features.cellGlow, "simulation parameters.features.cell glow", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.features.customizeNeuronMutations,
            defaultParameters.features.customizeNeuronMutations,
            "simulation parameters.features.customize neuron mutations",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.features.customizeDeletionMutations,
            defaultParameters.features.customizeDeletionMutations,
            "simulation parameters.features.customize deletion mutations",
            parserTask);
    }

    void encodeDecodeSimulationParameters(boost::property_tree::ptree& tree, SimulationParameters& parameters, ParserTask parserTask)
    {
        auto programVersion = Const::ProgramVersion;
        ParameterParser::encodeDecode(tree, programVersion, std::string(), "simulation parameters.version", parserTask);

        encodeDecodeLatestSimulationParameters(tree, parameters, parserTask);

        // Compatibility with legacy parameters
        if (parserTask == ParserTask::Decode) {
            LegacySettingsParserService::get().searchAndApplyLegacyParameters(programVersion, tree, parameters);
        }
    }

    void encodeDecode(boost::property_tree::ptree& tree, SettingsForSerialization& data, ParserTask parserTask)
    {
        SettingsForSerialization defaultSettings;

        //general settings
        ParameterParser::encodeDecode(tree, data.timestep, uint64_t(0), "general.time step", parserTask);
        ParameterParser::encodeDecode(tree, data.realTime, std::chrono::milliseconds(0), "general.real time", parserTask);
        ParameterParser::encodeDecode(tree, data.zoom, 4.0f, "general.zoom", parserTask);
        ParameterParser::encodeDecode(tree, data.center.x, 0.0f, "general.center.x", parserTask);
        ParameterParser::encodeDecode(tree, data.center.y, 0.0f, "general.center.y", parserTask);
        ParameterParser::encodeDecode(tree, data.worldSize.x, defaultSettings.worldSize.x, "general.world size.x", parserTask);
        ParameterParser::encodeDecode(tree, data.worldSize.y, defaultSettings.worldSize.y, "general.world size.y", parserTask);

        encodeDecodeSimulationParameters(tree, data.simulationParameters, parserTask);
    }
}

boost::property_tree::ptree SettingsParserService::encodeAuxiliaryData(SettingsForSerialization const& data)
{
    boost::property_tree::ptree tree;
    encodeDecode(tree, const_cast<SettingsForSerialization&>(data), ParserTask::Encode);
    return tree;
}

SettingsForSerialization SettingsParserService::decodeAuxiliaryData(boost::property_tree::ptree tree)
{
    SettingsForSerialization result;
    encodeDecode(tree, result, ParserTask::Decode);
    return result;
}

boost::property_tree::ptree SettingsParserService::encodeSimulationParameters(SimulationParameters const& data)
{
    boost::property_tree::ptree tree;
    encodeDecodeSimulationParameters(tree, const_cast<SimulationParameters&>(data), ParserTask::Encode);
    return tree;
}

SimulationParameters SettingsParserService::decodeSimulationParameters(boost::property_tree::ptree tree)
{
    SimulationParameters result;
    encodeDecodeSimulationParameters(tree, result, ParserTask::Decode);
    return result;
}
