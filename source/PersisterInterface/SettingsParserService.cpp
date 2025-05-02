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
        ParameterParser::encodeDecode(
            tree, parameters.projectName.value, defaultParameters.projectName.value, "simulation parameters.project name", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.backgroundColor.baseValue.r,
            defaultParameters.backgroundColor.baseValue.r,
            "simulation parameters.background color.r",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.backgroundColor.baseValue.g,
            defaultParameters.backgroundColor.baseValue.g,
            "simulation parameters.background color.g",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.backgroundColor.baseValue.b,
            defaultParameters.backgroundColor.baseValue.b,
            "simulation parameters.background color.b",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.primaryCellColoring.value, defaultParameters.primaryCellColoring.value, "simulation parameters.cell colorization", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellGlowColoring.value, defaultParameters.cellGlowColoring.value, "simulation parameters.cell glow.coloring", parserTask);
        ParameterParser::encodeDecode(tree, parameters.cellGlowRadius.value, defaultParameters.cellGlowRadius.value, "simulation parameters.cell glow.radius", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellGlowStrength.value, defaultParameters.cellGlowStrength.value, "simulation parameters.cell glow.strength", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.highlightedCellType.value,
            defaultParameters.highlightedCellType.value,
            "simulation parameters.highlighted cell function",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.zoomLevelForNeuronVisualization.value,
            defaultParameters.zoomLevelForNeuronVisualization.value,
            "simulation parameters.zoom level.neural activity",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.borderlessRendering.value, defaultParameters.borderlessRendering.value, "simulation parameters.borderless rendering", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.markReferenceDomain.value, defaultParameters.markReferenceDomain.value, "simulation parameters.mark reference domain", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.showRadiationSources.value,
            defaultParameters.showRadiationSources.value,
            "simulation parameters.show radiation sources",
            parserTask);
        ParameterParser::encodeDecode(tree, parameters.gridLines.value, defaultParameters.gridLines.value, "simulation parameters.grid lines", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.attackVisualization.value, defaultParameters.attackVisualization.value, "simulation parameters.attack visualization", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.muscleMovementVisualization.value,
            defaultParameters.muscleMovementVisualization.value,
            "simulation parameters.muscle movement visualization",
            parserTask);
        ParameterParser::encodeDecode(tree, parameters.cellRadius.value, defaultParameters.cellRadius.value, "simulation parameters.cek", parserTask);

        ParameterParser::encodeDecode(
            tree, parameters.timestepSize.value, defaultParameters.timestepSize.value, "simulation parameters.time step size", parserTask);

        ParameterParser::encodeDecode(tree, parameters.motionType.value, defaultParameters.motionType.value, "simulation parameters.motion.type", parserTask);
        if (parameters.motionType.value == MotionType_Fluid) {
            ParameterParser::encodeDecode(
                tree,
                parameters.smoothingLength.value,
                defaultParameters.smoothingLength.value,
                "simulation parameters.fluid.smoothing length",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.pressureStrength.value,
                defaultParameters.pressureStrength.value,
                "simulation parameters.fluid.pressure strength",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.viscosityStrength.value,
                defaultParameters.viscosityStrength.value,
                "simulation parameters.fluid.viscosity strength",
                parserTask);
        } else {
            ParameterParser::encodeDecode(
                tree,
                parameters.maxCollisionDistance.value,
                defaultParameters.maxCollisionDistance.value,
                "simulation parameters.motion.collision.max distance",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.repulsionStrength.value,
                defaultParameters.repulsionStrength.value,
                "simulation parameters.motion.collision.repulsion strength",
                parserTask);
        }

        ParameterParser::encodeDecode(
            tree, parameters.friction.baseValue, defaultParameters.friction.baseValue, "simulation parameters.friction", parserTask);
        ParameterParser::encodeDecode(tree, parameters.rigidity.baseValue, defaultParameters.rigidity.baseValue, "simulation parameters.rigidity", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.maxVelocity.value, defaultParameters.maxVelocity.value, "simulation parameters.cell.max velocity", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.maxBindingDistance.value,
            defaultParameters.maxBindingDistance.value,
            "simulation parameters.cell.max binding distance",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.normalCellEnergy.value, defaultParameters.normalCellEnergy.value, "simulation parameters.cell.normal energy", parserTask);

        ParameterParser::encodeDecode(
            tree, parameters.minCellDistance.value, defaultParameters.minCellDistance.value, "simulation parameters.cell.min distance", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.maxForce.baseValue, defaultParameters.maxForce.baseValue, "simulation parameters.cell.max force", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.minCellEnergy.baseValue, defaultParameters.minCellEnergy.baseValue, "simulation parameters.cell.min energy", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellFusionVelocity.baseValue,
            parameters.cellFusionVelocity.baseValue,
            "simulation parameters.cell.fusion velocity",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellMaxBindingEnergy.baseValue,
            parameters.cellMaxBindingEnergy.baseValue,
            "simulation parameters.cell.max binding energy",
            parserTask);
        ParameterParser::encodeDecode(tree, parameters.maxCellAge.value, defaultParameters.maxCellAge.value, "simulation parameters.cell.max age", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.maxCellAgeBalancerInterval.enabled,
            defaultParameters.maxCellAgeBalancerInterval.enabled,
            "simulation parameters.cell.max age.balance.enabled",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.maxCellAgeBalancerInterval.value,
            defaultParameters.maxCellAgeBalancerInterval.value,
            "simulation parameters.cell.max age.balance.interval",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.maxAgeForInactiveCells.baseValue,
            defaultParameters.maxAgeForInactiveCells.baseValue,
            "simulation parameters.cell.inactive max age",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.freeCellMaxAge.value, defaultParameters.freeCellMaxAge.value, "simulation parameters.cell.nutrient max age", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.resetCellAgeAfterActivation.value,
            defaultParameters.resetCellAgeAfterActivation.value,
            "simulation parameters.cell.reset age after activation",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.colorTransitionRules.baseValue.cellColorTransitionDuration,
            defaultParameters.colorTransitionRules.baseValue.cellColorTransitionDuration,
            "simulation parameters.cell.color transition rules.duration",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.colorTransitionRules.baseValue.cellColorTransitionTargetColor,
            defaultParameters.colorTransitionRules.baseValue.cellColorTransitionTargetColor,
            "simulation parameters.cell.color transition rules.target color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.genomeComplexitySizeFactor.value,
            defaultParameters.genomeComplexitySizeFactor.value,
            "simulation parameters.genome complexity.genome complexity size factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.genomeComplexityRamificationFactor.value,
            defaultParameters.genomeComplexityRamificationFactor.value,
            "simulation parameters.genome complexity.genome complexity ramification factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.genomeComplexityDepthLevel.value,
            defaultParameters.genomeComplexityDepthLevel.value,
            "simulation parameters.genome complexity.genome complexity depth level",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationType1_strength.baseValue,
            defaultParameters.radiationType1_strength.baseValue,
            "simulation parameters.radiation.factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationAbsorption.baseValue,
            defaultParameters.radiationAbsorption.baseValue,
            "simulation parameters.radiation.absorption",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationAbsorptionHighVelocityPenalty.value,
            defaultParameters.radiationAbsorptionHighVelocityPenalty.value,
            "simulation parameters.radiation.absorption velocity penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationAbsorptionLowVelocityPenalty.baseValue,
            defaultParameters.radiationAbsorptionLowVelocityPenalty.baseValue,
            "simulation parameters.radiation.absorption low velocity penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationAbsorptionLowConnectionPenalty.value,
            defaultParameters.radiationAbsorptionLowConnectionPenalty.value,
            "simulation parameters.radiation.absorption low connection penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationAbsorptionLowGenomeComplexityPenalty.baseValue,
            defaultParameters.radiationAbsorptionLowGenomeComplexityPenalty.baseValue,
            "simulation parameters.radiation.absorption low genome complexity penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationType2_energyThreshold.value,
            defaultParameters.radiationType2_energyThreshold.value,
            "simulation parameters.high radiation.min cell energy",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationType2_strength.value,
            defaultParameters.radiationType2_strength.value,
            "simulation parameters.high radiation.factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.radiationType1_minimumAge.value,
            defaultParameters.radiationType1_minimumAge.value,
            "simulation parameters.radiation.min cell age",
            parserTask);

        ParameterParser::encodeDecode(
            tree, parameters.externalEnergy.value, defaultParameters.externalEnergy.value, "simulation parameters.cell.function.constructor.external energy", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyInflowFactor.value,
            defaultParameters.externalEnergyInflowFactor.value,
            "simulation parameters.cell.function.constructor.external energy supply rate",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyConditionalInflowFactor.value,
            defaultParameters.externalEnergyConditionalInflowFactor.value,
            "simulation parameters.cell.function.constructor.pump energy factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyInflowOnlyForNonSelfReplicators.value,
            defaultParameters.externalEnergyInflowOnlyForNonSelfReplicators.value,
            "simulation parameters.cell.function.constructor.externalEnergyInflowOnlyForNonSelfReplicators",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyBackflowFactor.value,
            defaultParameters.externalEnergyBackflowFactor.value,
            "simulation parameters.cell.function.constructor.external energy backflow",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyInflowOnlyForNonSelfReplicators.value,
            defaultParameters.externalEnergyInflowOnlyForNonSelfReplicators.value,
            "simulation parameters.cell.function.constructor.external energy inflow only for non-self-replicators",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyBackflowLimit.value,
            defaultParameters.externalEnergyBackflowLimit.value,
            "simulation parameters.cell.function.constructor.external energy backflow limit",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.cellDeathConsequences.value,
            defaultParameters.cellDeathConsequences.value,
            "simulation parameters.cell.death consequences",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellDeathProbability.baseValue,
            defaultParameters.cellDeathProbability.baseValue,
            "simulation parameters.cell.death probability",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.constructorConnectingCellDistance.value,
            defaultParameters.constructorConnectingCellDistance.value,
            "simulation parameters.cell.function.constructor.connecting cell max distance",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.constructorCompletenessCheck.value,
            defaultParameters.constructorCompletenessCheck.value,
            "simulation parameters.cell.function.constructor.completeness check for self-replication",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationNeuronData.baseValue,
            defaultParameters.copyMutationNeuronData.baseValue,
            "simulation parameters.cell.copy mutation.neuron data",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataWeight.value,
            defaultParameters.cellCopyMutationNeuronDataWeight.value,
            "simulation parameters.cell.copy mutation.neuron data.weights",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataBias.value,
            defaultParameters.cellCopyMutationNeuronDataBias.value,
            "simulation parameters.cell.copy mutation.neuron data.biases",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataActivationFunction.value,
            defaultParameters.cellCopyMutationNeuronDataActivationFunction.value,
            "simulation parameters.cell.copy mutation.neuron data.activation functions",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataReinforcement.value,
            defaultParameters.cellCopyMutationNeuronDataReinforcement.value,
            "simulation parameters.cell.copy mutation.neuron data.reinforcement",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataDamping.value,
            defaultParameters.cellCopyMutationNeuronDataDamping.value,
            "simulation parameters.cell.copy mutation.neuron data.damping",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationNeuronDataOffset.value,
            defaultParameters.cellCopyMutationNeuronDataOffset.value,
            "simulation parameters.cell.copy mutation.neuron data.offset",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationCellProperties.baseValue,
            defaultParameters.copyMutationCellProperties.baseValue,
            "simulation parameters.cell.copy mutation.cell properties",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationGeometry.baseValue,
            defaultParameters.copyMutationGeometry.baseValue,
            "simulation parameters.cell.copy mutation.geometry",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationCustomGeometry.baseValue,
            defaultParameters.copyMutationCustomGeometry.baseValue,
            "simulation parameters.cell.copy mutation.custom geometry",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationCellType.baseValue,
            defaultParameters.copyMutationCellType.baseValue,
            "simulation parameters.cell.copy mutation.cell function",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationInsertion.baseValue,
            defaultParameters.copyMutationInsertion.baseValue,
            "simulation parameters.cell.copy mutation.insertion",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationDeletion.baseValue,
            defaultParameters.copyMutationDeletion.baseValue,
            "simulation parameters.cell.copy mutation.deletion",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.cellCopyMutationDeletionMinSize.value,
            defaultParameters.cellCopyMutationDeletionMinSize.value,
            "simulation parameters.cell.copy mutation.deletion.min size",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationTranslation.baseValue,
            defaultParameters.copyMutationTranslation.baseValue,
            "simulation parameters.cell.copy mutation.translation",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationDuplication.baseValue,
            defaultParameters.copyMutationDuplication.baseValue,
            "simulation parameters.cell.copy mutation.duplication",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationCellColor.baseValue,
            defaultParameters.copyMutationCellColor.baseValue,
            "simulation parameters.cell.copy mutation.cell color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationSubgenomeColor.baseValue,
            defaultParameters.copyMutationSubgenomeColor.baseValue,
            "simulation parameters.cell.copy mutation.subgenome color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationGenomeColor.baseValue,
            defaultParameters.copyMutationGenomeColor.baseValue,
            "simulation parameters.cell.copy mutation.genome color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationColorTransitions.value,
            defaultParameters.copyMutationColorTransitions.value,
            "simulation parameters.cell.copy mutation.color transition",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationSelfReplication.value,
            defaultParameters.copyMutationSelfReplication.value,
            "simulation parameters.cell.copy mutation.self replication flag",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationPreventDepthIncrease.value,
            defaultParameters.copyMutationPreventDepthIncrease.value,
            "simulation parameters.cell.copy mutation.prevent depth increase",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.injectorInjectionRadius.value,
            defaultParameters.injectorInjectionRadius.value,
            "simulation parameters.cell.function.injector.radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.injectorInjectionTime.value,
            defaultParameters.injectorInjectionTime.value,
            "simulation parameters.cell.function.injector.duration",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.attackerRadius.value,
            defaultParameters.attackerRadius.value,
            "simulation parameters.cell.function.attacker.radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerStrength.value,
            defaultParameters.attackerStrength.value,
            "simulation parameters.cell.function.attacker.strength",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerEnergyCost.baseValue,
            defaultParameters.attackerEnergyCost.baseValue,
            "simulation parameters.cell.function.attacker.energy cost",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerGeometryDeviationProtection.baseValue,
            defaultParameters.attackerGeometryDeviationProtection.baseValue,
            "simulation parameters.cell.function.attacker.geometry deviation exponent",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerFoodChainColorMatrix.baseValue,
            defaultParameters.attackerFoodChainColorMatrix.baseValue,
            "simulation parameters.cell.function.attacker.food chain color matrix",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerConnectionsMismatchProtection.baseValue,
            defaultParameters.attackerConnectionsMismatchProtection.baseValue,
            "simulation parameters.cell.function.attacker.connections mismatch penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerComplexCreatureProtection.baseValue,
            defaultParameters.attackerComplexCreatureProtection.baseValue,
            "simulation parameters.cell.function.attacker.genome size bonus",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerSameMutantProtection.value,
            defaultParameters.attackerSameMutantProtection.value,
            "simulation parameters.cell.function.attacker.same mutant penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerNewComplexMutantProtection.baseValue,
            defaultParameters.attackerNewComplexMutantProtection.baseValue,
            "simulation parameters.cell.function.attacker.new complex mutant penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerSensorDetectionFactor.value,
            defaultParameters.attackerSensorDetectionFactor.value,
            "simulation parameters.cell.function.attacker.sensor detection factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerDestroyCells.value,
            defaultParameters.attackerDestroyCells.value,
            "simulation parameters.cell.function.attacker.destroy cells",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.defenderAntiAttackerStrength.value,
            defaultParameters.defenderAntiAttackerStrength.value,
            "simulation parameters.cell.function.defender.against attacker strength",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.defenderAntiInjectorStrength.value,
            defaultParameters.defenderAntiInjectorStrength.value,
            "simulation parameters.cell.function.defender.against injector strength",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.transmitterEnergyDistributionSameCreature.value,
            defaultParameters.transmitterEnergyDistributionSameCreature.value,
            "simulation parameters.cell.function.transmitter.energy distribution same creature",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.transmitterEnergyDistributionRadius.value,
            defaultParameters.transmitterEnergyDistributionRadius.value,
            "simulation parameters.cell.function.transmitter.energy distribution radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.transmitterEnergyDistributionValue.value,
            defaultParameters.transmitterEnergyDistributionValue.value,
            "simulation parameters.cell.function.transmitter.energy distribution value",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.muscleCrawlingAcceleration.value,
            defaultParameters.muscleCrawlingAcceleration.value,
            "simulation parameters.cell.function.muscle.crawling acceleration",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.muscleMovementAcceleration.value,
            defaultParameters.muscleMovementAcceleration.value,
            "simulation parameters.cell.function.muscle.movement acceleration",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.muscleBendingAcceleration.value,
            defaultParameters.muscleBendingAcceleration.value,
            "simulation parameters.cell.function.muscle.bending acceleration",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.muscleEnergyCost.value,
            defaultParameters.muscleEnergyCost.value,
            "simulation parameters.cell.function.muscle.energy cost",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.particleTransformationAllowed.value,
            defaultParameters.particleTransformationAllowed.value,
            "simulation parameters.particle.transformation allowed",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.particleSplitEnergy.value, defaultParameters.particleSplitEnergy.value, "simulation parameters.particle.split energy", parserTask);

        ParameterParser::encodeDecode(
            tree, parameters.sensorRadius.value, defaultParameters.sensorRadius.value, "simulation parameters.cell.function.sensor.range", parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.reconnectorRadius.value,
            defaultParameters.reconnectorRadius.value,
            "simulation parameters.cell.function.reconnector.radius",
            parserTask);

        ParameterParser::encodeDecode(
            tree, parameters.detonatorRadius.value,
            defaultParameters.detonatorRadius.value,
            "simulation parameters.cell.function.detonator.radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.detonatorChainExplosionProbability.value,
            defaultParameters.detonatorChainExplosionProbability.value,
            "simulation parameters.cell.function.detonator.chain explosion probability",
            parserTask);

        //particle sources
        ParameterParser::encodeDecode(
            tree,
            parameters.numSources,
            defaultParameters.numSources,
            "simulation parameters.particle sources.num sources",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.relativeStrengthBasePin.pinned,
            defaultParameters.relativeStrengthBasePin.pinned,
            "simulation parameters.particle sources.base strength pinned",
            parserTask);
        for (int index = 0; index < parameters.numSources; ++index) {
            std::string base = "simulation parameters.particle sources." + std::to_string(index) + ".";
            ParameterParser::encodeDecode(
                tree, parameters.sourceName.sourceValues[index], defaultParameters.sourceName.sourceValues[index], base + "name", parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.sourceLocationIndex[index],
                defaultParameters.sourceLocationIndex[index],
                base + "location index",
                parserTask);
            ParameterParser::encodeDecode(
                tree, parameters.sourcePosition.sourceValues[index].x, defaultParameters.sourcePosition.sourceValues[index].x, base + "pos.x", parserTask);
            ParameterParser::encodeDecode(
                tree, parameters.sourcePosition.sourceValues[index].y, defaultParameters.sourcePosition.sourceValues[index].y, base + "pos.y", parserTask);
            ParameterParser::encodeDecode(
                tree, parameters.sourceVelocity.sourceValues[index].x, defaultParameters.sourceVelocity.sourceValues[index].x, base + "vel.x", parserTask);
            ParameterParser::encodeDecode(
                tree, parameters.sourceVelocity.sourceValues[index].y, defaultParameters.sourceVelocity.sourceValues[index].y, base + "vel.y", parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.sourceRadiationAngle.sourceValues[index].enabled,
                defaultParameters.sourceRadiationAngle.sourceValues[index].enabled,
                base + "use angle",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.sourceRelativeStrength.sourceValues[index].value,
                defaultParameters.sourceRelativeStrength.sourceValues[index].value,
                base + "strength",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.sourceRelativeStrength.sourceValues[index].pinned,
                defaultParameters.sourceRelativeStrength.sourceValues[index].pinned,
                base + "strength pinned",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.sourceRadiationAngle.sourceValues[index].value,
                defaultParameters.sourceRadiationAngle.sourceValues[index].value,
                base + "angle",
                parserTask);
            ParameterParser::encodeDecode(
                tree, parameters.sourceShapeType.sourceValues[index], defaultParameters.sourceShapeType.sourceValues[index], base + "shape.type", parserTask);
            if (parameters.sourceShapeType.sourceValues[index] == LayerShapeType_Circular) {
                ParameterParser::encodeDecode(
                    tree,
                    parameters.sourceCircularRadius.sourceValues[index],
                    defaultParameters.sourceCircularRadius.sourceValues[index],
                    base + "shape.circular.radius",
                    parserTask);
            }
            if (parameters.sourceShapeType.sourceValues[index] == LayerShapeType_Rectangular) {
                ParameterParser::encodeDecode(
                    tree,
                    parameters.sourceRectangularRect.sourceValues[index].x,
                    defaultParameters.sourceRectangularRect.sourceValues[index].x,
                    base + "shape.rectangular.width",
                    parserTask);
                ParameterParser::encodeDecode(
                    tree,
                    parameters.sourceRectangularRect.sourceValues[index].y,
                    defaultParameters.sourceRectangularRect.sourceValues[index].y,
                    base + "shape.rectangular.height",
                    parserTask);
            }
        }

        // Layers
        ParameterParser::encodeDecode(tree, parameters.numLayers, defaultParameters.numLayers, "simulation parameters.spots.num spots", parserTask);
        for (int index = 0; index < parameters.numLayers; ++index) {
            std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
            ParameterParser::encodeDecode(tree, parameters.layerName.layerValues[index], defaultParameters.layerName.layerValues[0], base + "name", parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerLocationIndex[index],
                defaultParameters.layerLocationIndex[index],
                base + "location index",
                parserTask);
            ParameterParser::encodeDecode(tree, parameters.layerPosition.layerValues[index].x, defaultParameters.layerPosition.layerValues[index].x, base + "pos.x", parserTask);
            ParameterParser::encodeDecode(tree, parameters.layerPosition.layerValues[index].y, defaultParameters.layerPosition.layerValues[index].y, base + "pos.y", parserTask);
            ParameterParser::encodeDecode(tree, parameters.layerVelocity.layerValues[index].x, defaultParameters.layerVelocity.layerValues[index].x, base + "vel.x", parserTask);
            ParameterParser::encodeDecode(tree, parameters.layerVelocity.layerValues[index].y, defaultParameters.layerVelocity.layerValues[index].y, base + "vel.y", parserTask);

            ParameterParser::encodeDecode(
                tree, parameters.layerShape.layerValues[index], defaultParameters.layerShape.layerValues[index], base + "shape.type", parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerCoreRadius.layerValues[index],
                defaultParameters.layerCoreRadius.layerValues[index],
                base + "shape.circular.core radius",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerCoreRect.layerValues[index].x,
                defaultParameters.layerCoreRect.layerValues[index].x,
                base + "shape.rectangular.core width",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerCoreRect.layerValues[index].y,
                defaultParameters.layerCoreRect.layerValues[index].y,
                base + "shape.rectangular.core height",
                parserTask);

            ParameterParser::encodeDecode(
                tree, parameters.layerForceFieldType.layerValues[index], defaultParameters.layerForceFieldType.layerValues[index], base + "flow.type", parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerRadialForceFieldOrientation.layerValues[index],
                defaultParameters.layerRadialForceFieldOrientation.layerValues[index],
                base + "flow.radial.orientation",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerRadialForceFieldStrength.layerValues[index],
                defaultParameters.layerRadialForceFieldStrength.layerValues[index],
                base + "flow.radial.strength",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerRadialForceFieldDriftAngle.layerValues[index],
                defaultParameters.layerRadialForceFieldDriftAngle.layerValues[index],
                base + "flow.radial.drift angle",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerCentralForceFieldStrength.layerValues[index],
                defaultParameters.layerCentralForceFieldStrength.layerValues[index],
                base + "flow.central.strength",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerLinearForceFieldAngle.layerValues[index],
                defaultParameters.layerLinearForceFieldAngle.layerValues[index],
                base + "flow.linear.angle",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerLinearForceFieldStrength.layerValues[index],
                defaultParameters.layerLinearForceFieldStrength.layerValues[index],
                base + "flow.linear.strength",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.layerFadeoutRadius.layerValues[index],
                defaultParameters.layerFadeoutRadius.layerValues[index],
                base + "fadeout radius",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.backgroundColor.layerValues[index].value.r,
                parameters.backgroundColor.layerValues[index].enabled,
                parameters.backgroundColor.baseValue.r,
                base + "color.r",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.backgroundColor.layerValues[index].value.g,
                parameters.backgroundColor.layerValues[index].enabled,
                parameters.backgroundColor.baseValue.g,
                base + "color.g",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.backgroundColor.layerValues[index].value.b,
                parameters.backgroundColor.layerValues[index].enabled,
                parameters.backgroundColor.baseValue.b,
                base + "color.b",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.friction.layerValues[index].value,
                parameters.friction.layerValues[index].enabled,
                defaultParameters.friction.baseValue,
                base + "friction",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.rigidity.layerValues[index].value,
                parameters.rigidity.layerValues[index].enabled,
                defaultParameters.rigidity.baseValue,
                base + "rigidity",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.disableRadiationSources.layerValues[index],
                defaultParameters.disableRadiationSources.layerValues[0],
                base + "radiation.disable sources",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.radiationAbsorption.layerValues[index].value,
                parameters.radiationAbsorption.layerValues[index].enabled,
                parameters.radiationAbsorption.baseValue,
                base + "radiation.absorption",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.radiationAbsorptionLowVelocityPenalty.layerValues[index].value,
                parameters.radiationAbsorptionLowVelocityPenalty.layerValues[index].enabled,
                parameters.radiationAbsorptionLowVelocityPenalty.baseValue,
                base + "radiation.absorption low velocity penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.radiationAbsorptionLowGenomeComplexityPenalty.layerValues[index].value,
                parameters.radiationAbsorptionLowGenomeComplexityPenalty.layerValues[index].enabled,
                parameters.radiationAbsorptionLowGenomeComplexityPenalty.baseValue,
                base + "radiation.absorption low genome complexity penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.radiationType1_strength.layerValues[index].value,
                parameters.radiationType1_strength.layerValues[index].enabled,
                parameters.radiationType1_strength.baseValue,
                base + "radiation.factor",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.maxForce.layerValues[index].value,
                parameters.maxForce.layerValues[index].enabled,
                defaultParameters.maxForce.baseValue,
                base + "cell.max force",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.minCellEnergy.layerValues[index].value,
                parameters.minCellEnergy.layerValues[index].enabled,
                defaultParameters.minCellEnergy.baseValue,
                base + "cell.min energy",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.cellDeathProbability.layerValues[index].value,
                parameters.cellDeathProbability.layerValues[index].enabled,
                defaultParameters.cellDeathProbability.baseValue,
                base + "cell.death probability",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.cellFusionVelocity.layerValues[index].value,
                parameters.cellFusionVelocity.layerValues[index].enabled,
                defaultParameters.cellFusionVelocity.baseValue,
                base + "cell.fusion velocity",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.cellMaxBindingEnergy.layerValues[index].value,
                parameters.cellMaxBindingEnergy.layerValues[index].enabled,
                defaultParameters.cellMaxBindingEnergy.baseValue,
                base + "cell.max binding energy",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.maxAgeForInactiveCells.layerValues[index].value,
                parameters.maxAgeForInactiveCells.layerValues[index].enabled,
                defaultParameters.maxAgeForInactiveCells.baseValue,
                base + "cell.inactive max age",
                parserTask);

            ParameterParser::encodeDecode(
                tree, parameters.colorTransitionRules.layerValues[index].enabled, false, base + "cell.color transition rules.activated", parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.colorTransitionRules.layerValues[index].value.cellColorTransitionDuration,
                defaultParameters.colorTransitionRules.layerValues[index].value.cellColorTransitionDuration,
                base + "cell.color transition rules.duration",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.colorTransitionRules.layerValues[index].value.cellColorTransitionTargetColor,
                defaultParameters.colorTransitionRules.layerValues[index].value.cellColorTransitionTargetColor,
                base + "cell.color transition rules.target color",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerEnergyCost.layerValues[index].value,
                parameters.attackerEnergyCost.layerValues[index].enabled,
                defaultParameters.attackerEnergyCost.baseValue,
                base + "cell.function.attacker.energy cost",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerFoodChainColorMatrix.layerValues[index].value,
                parameters.attackerFoodChainColorMatrix.layerValues[index].enabled,
                defaultParameters.attackerFoodChainColorMatrix.baseValue,
                base + "cell.function.attacker.food chain color matrix",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerComplexCreatureProtection.layerValues[index].value,
                parameters.attackerComplexCreatureProtection.layerValues[index].enabled,
                defaultParameters.attackerComplexCreatureProtection.baseValue,
                base + "cell.function.attacker.genome size bonus",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerNewComplexMutantProtection.layerValues[index].value,
                parameters.attackerNewComplexMutantProtection.layerValues[index].enabled,
                defaultParameters.attackerNewComplexMutantProtection.baseValue,
                base + "cell.function.attacker.new complex mutant penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerGeometryDeviationProtection.layerValues[index].value,
                parameters.attackerGeometryDeviationProtection.layerValues[index].enabled,
                defaultParameters.attackerGeometryDeviationProtection.baseValue,
                base + "cell.function.attacker.geometry deviation exponent",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerConnectionsMismatchProtection.layerValues[index].value,
                parameters.attackerConnectionsMismatchProtection.layerValues[index].enabled,
                defaultParameters.attackerConnectionsMismatchProtection.baseValue,
                base + "cell.function.attacker.connections mismatch penalty",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationNeuronData.layerValues[index].value,
                parameters.copyMutationNeuronData.layerValues[index].enabled,
                defaultParameters.copyMutationNeuronData.baseValue,
                base + "cell.copy mutation.neuron data",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationCellProperties.layerValues[index].value,
                parameters.copyMutationCellProperties.layerValues[index].enabled,
                defaultParameters.copyMutationCellProperties.baseValue,
                base + "cell.copy mutation.cell properties",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationGeometry.layerValues[index].value,
                parameters.copyMutationGeometry.layerValues[index].enabled,
                defaultParameters.copyMutationGeometry.baseValue,
                base + "cell.copy mutation.geometry",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationCustomGeometry.layerValues[index].value,
                parameters.copyMutationCustomGeometry.layerValues[index].enabled,
                defaultParameters.copyMutationCustomGeometry.baseValue,
                base + "cell.copy mutation.custom geometry",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationCellType.layerValues[index].value,
                parameters.copyMutationCellType.layerValues[index].enabled,
                defaultParameters.copyMutationCellType.baseValue,
                base + "cell.copy mutation.cell function",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationInsertion.layerValues[index].value,
                parameters.copyMutationInsertion.layerValues[index].enabled,
                defaultParameters.copyMutationInsertion.baseValue,
                base + "cell.copy mutation.insertion",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationDeletion.layerValues[index].value,
                parameters.copyMutationDeletion.layerValues[index].enabled,
                defaultParameters.copyMutationDeletion.baseValue,
                base + "cell.copy mutation.deletion",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationTranslation.layerValues[index].value,
                parameters.copyMutationTranslation.layerValues[index].enabled,
                defaultParameters.copyMutationTranslation.baseValue,
                base + "cell.copy mutation.translation",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationDuplication.layerValues[index].value,
                parameters.copyMutationDuplication.layerValues[index].enabled,
                defaultParameters.copyMutationDuplication.baseValue,
                base + "cell.copy mutation.duplication",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationCellColor.layerValues[index].value,
                parameters.copyMutationCellColor.layerValues[index].enabled,
                defaultParameters.copyMutationCellColor.baseValue,
                base + "cell.copy mutation.cell color",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationSubgenomeColor.layerValues[index].value,
                parameters.copyMutationSubgenomeColor.layerValues[index].enabled,
                defaultParameters.copyMutationSubgenomeColor.baseValue,
                base + "cell.copy mutation.subgenome color",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationGenomeColor.layerValues[index].value,
                parameters.copyMutationGenomeColor.layerValues[index].enabled,
                defaultParameters.copyMutationGenomeColor.baseValue,
                base + "cell.copy mutation.genome color",
                parserTask);
        }

        //features
        ParameterParser::encodeDecode(
            tree,
            parameters.genomeComplexityMeasurementToggle.value,
            defaultParameters.genomeComplexityMeasurementToggle.value,
            "simulation parameters.features.genome complexity measurement",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.advancedAbsorptionControlToggle.value,
            defaultParameters.advancedAbsorptionControlToggle.value,
            "simulation parameters.features.additional absorption control",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.advancedAttackerControlToggle.value,
            defaultParameters.advancedAttackerControlToggle.value,
            "simulation parameters.features.additional attacker control",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.externalEnergyControlToggle.value,
            defaultParameters.externalEnergyControlToggle.value,
            "simulation parameters.features.external energy",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.colorTransitionRulesToggle.value,
            defaultParameters.colorTransitionRulesToggle.value,
            "simulation parameters.features.cell color transition rules",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellAgeLimiterToggle.value, defaultParameters.cellAgeLimiterToggle.value, "simulation parameters.features.cell age limiter", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellGlowToggle.value, defaultParameters.cellGlowToggle.value, "simulation parameters.features.cell glow", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.customizeNeuronMutationsToggle.value,
            defaultParameters.customizeNeuronMutationsToggle.value,
            "simulation parameters.features.customize neuron mutations",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.customizeDeletionMutationsToggle.value,
            defaultParameters.customizeDeletionMutationsToggle.value,
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
