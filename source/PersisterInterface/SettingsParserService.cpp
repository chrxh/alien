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
            parameters.numRadiationSources.value,
            defaultParameters.numRadiationSources.value,
            "simulation parameters.particle sources.num sources",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.relativeStrengthPinned.pinned,
            defaultParameters.relativeStrengthPinned.pinned,
            "simulation parameters.particle sources.base strength pinned",
            parserTask);
        for (int index = 0; index < parameters.numRadiationSources.value; ++index) {
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

        // Zones
        ParameterParser::encodeDecode(tree, parameters.numZones.value, defaultParameters.numZones.value, "simulation parameters.spots.num spots", parserTask);
        for (int index = 0; index < parameters.numZones.value; ++index) {
            std::string base = "simulation parameters.spots." + std::to_string(index) + ".";
            auto& spot = parameters.zone[index];
            auto& defaultSpot = defaultParameters.zone[index];
            ParameterParser::encodeDecode(tree, parameters.zoneNames.zoneValues[index], defaultParameters.zoneNames.zoneValues[0], base + "name", parserTask);
            ParameterParser::encodeDecode(tree, spot.locationIndex, defaultSpot.locationIndex, base + "location index", parserTask);
            ParameterParser::encodeDecode(tree, parameters.zonePosition.zoneValues[index].x, defaultParameters.zonePosition.zoneValues[index].x, base + "pos.x", parserTask);
            ParameterParser::encodeDecode(tree, parameters.zonePosition.zoneValues[index].y, defaultParameters.zonePosition.zoneValues[index].y, base + "pos.y", parserTask);
            ParameterParser::encodeDecode(tree, spot.velX, defaultSpot.velX, base + "vel.x", parserTask);
            ParameterParser::encodeDecode(tree, spot.velY, defaultSpot.velY, base + "vel.y", parserTask);

            ParameterParser::encodeDecode(tree, spot.shape.type, defaultSpot.shape.type, base + "shape.type", parserTask);
            if (spot.shape.type == ZoneShapeType_Circular) {
                ParameterParser::encodeDecode(
                    tree,
                    spot.shape.alternatives.circularZone.coreRadius,
                    defaultSpot.shape.alternatives.circularZone.coreRadius,
                    base + "shape.circular.core radius",
                    parserTask);
            }
            if (spot.shape.type == ZoneShapeType_Rectangular) {
                ParameterParser::encodeDecode(
                    tree,
                    spot.shape.alternatives.rectangularZone.width,
                    defaultSpot.shape.alternatives.rectangularZone.width,
                    base + "shape.rectangular.core width",
                    parserTask);
                ParameterParser::encodeDecode(
                    tree,
                    spot.shape.alternatives.rectangularZone.height,
                    defaultSpot.shape.alternatives.rectangularZone.height,
                    base + "shape.rectangular.core height",
                    parserTask);
            }
            ParameterParser::encodeDecode(tree, spot.flow.type, defaultSpot.flow.type, base + "flow.type", parserTask);
            if (spot.flow.type == FlowType_Radial) {
                ParameterParser::encodeDecode(
                    tree,
                    spot.flow.alternatives.radialFlow.orientation,
                    defaultSpot.flow.alternatives.radialFlow.orientation,
                    base + "flow.radial.orientation",
                    parserTask);
                ParameterParser::encodeDecode(
                    tree,
                    spot.flow.alternatives.radialFlow.strength,
                    defaultSpot.flow.alternatives.radialFlow.strength,
                    base + "flow.radial.strength",
                    parserTask);
                ParameterParser::encodeDecode(
                    tree,
                    spot.flow.alternatives.radialFlow.driftAngle,
                    defaultSpot.flow.alternatives.radialFlow.driftAngle,
                    base + "flow.radial.drift angle",
                    parserTask);
            }
            if (spot.flow.type == FlowType_Central) {
                ParameterParser::encodeDecode(
                    tree,
                    spot.flow.alternatives.centralFlow.strength,
                    defaultSpot.flow.alternatives.centralFlow.strength,
                    base + "flow.central.strength",
                    parserTask);
            }
            if (spot.flow.type == FlowType_Linear) {
                ParameterParser::encodeDecode(
                    tree, spot.flow.alternatives.linearFlow.angle, defaultSpot.flow.alternatives.linearFlow.angle, base + "flow.linear.angle", parserTask);
                ParameterParser::encodeDecode(
                    tree,
                    spot.flow.alternatives.linearFlow.strength,
                    defaultSpot.flow.alternatives.linearFlow.strength,
                    base + "flow.linear.strength",
                    parserTask);
            }
            ParameterParser::encodeDecode(tree, spot.fadeoutRadius, defaultSpot.fadeoutRadius, base + "fadeout radius", parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.backgroundColor.zoneValues[index].value.r,
                parameters.backgroundColor.zoneValues[index].enabled,
                parameters.backgroundColor.baseValue.r,
                base + "color.r",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.backgroundColor.zoneValues[index].value.g,
                parameters.backgroundColor.zoneValues[index].enabled,
                parameters.backgroundColor.baseValue.g,
                base + "color.g",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.backgroundColor.zoneValues[index].value.b,
                parameters.backgroundColor.zoneValues[index].enabled,
                parameters.backgroundColor.baseValue.b,
                base + "color.b",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.friction.zoneValues[index].value,
                parameters.friction.zoneValues[index].enabled,
                defaultParameters.friction.baseValue,
                base + "friction",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.rigidity.zoneValues[index].value,
                parameters.rigidity.zoneValues[index].enabled,
                defaultParameters.rigidity.baseValue,
                base + "rigidity",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.radiationDisableSources.zoneValues[index],
                defaultParameters.radiationDisableSources.zoneValues[0],
                base + "radiation.disable sources",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.radiationAbsorption.zoneValues[index].value,
                parameters.radiationAbsorption.zoneValues[index].enabled,
                parameters.radiationAbsorption.baseValue,
                base + "radiation.absorption",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.radiationAbsorptionLowVelocityPenalty.zoneValues[index].value,
                parameters.radiationAbsorptionLowVelocityPenalty.zoneValues[index].enabled,
                parameters.radiationAbsorptionLowVelocityPenalty.baseValue,
                base + "radiation.absorption low velocity penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.radiationAbsorptionLowGenomeComplexityPenalty.zoneValues[index].value,
                parameters.radiationAbsorptionLowGenomeComplexityPenalty.zoneValues[index].enabled,
                parameters.radiationAbsorptionLowGenomeComplexityPenalty.baseValue,
                base + "radiation.absorption low genome complexity penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.radiationType1_strength.zoneValues[index].value,
                parameters.radiationType1_strength.zoneValues[index].enabled,
                parameters.radiationType1_strength.baseValue,
                base + "radiation.factor",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.maxForce.zoneValues[index].value,
                parameters.maxForce.zoneValues[index].enabled,
                defaultParameters.maxForce.baseValue,
                base + "cell.max force",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.minCellEnergy.zoneValues[index].value,
                parameters.minCellEnergy.zoneValues[index].enabled,
                defaultParameters.minCellEnergy.baseValue,
                base + "cell.min energy",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.cellDeathProbability.zoneValues[index].value,
                parameters.cellDeathProbability.zoneValues[index].enabled,
                defaultParameters.cellDeathProbability.baseValue,
                base + "cell.death probability",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.cellFusionVelocity.zoneValues[index].value,
                parameters.cellFusionVelocity.zoneValues[index].enabled,
                defaultParameters.cellFusionVelocity.baseValue,
                base + "cell.fusion velocity",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.cellMaxBindingEnergy.zoneValues[index].value,
                parameters.cellMaxBindingEnergy.zoneValues[index].enabled,
                defaultParameters.cellMaxBindingEnergy.baseValue,
                base + "cell.max binding energy",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.maxAgeForInactiveCells.zoneValues[index].value,
                parameters.maxAgeForInactiveCells.zoneValues[index].enabled,
                defaultParameters.maxAgeForInactiveCells.baseValue,
                base + "cell.inactive max age",
                parserTask);

            ParameterParser::encodeDecode(
                tree, parameters.colorTransitionRules.zoneValues[index].enabled, false, base + "cell.color transition rules.activated", parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.colorTransitionRules.zoneValues[index].value.cellColorTransitionDuration,
                defaultParameters.colorTransitionRules.zoneValues[index].value.cellColorTransitionDuration,
                base + "cell.color transition rules.duration",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.colorTransitionRules.zoneValues[index].value.cellColorTransitionTargetColor,
                defaultParameters.colorTransitionRules.zoneValues[index].value.cellColorTransitionTargetColor,
                base + "cell.color transition rules.target color",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerEnergyCost.zoneValues[index].value,
                parameters.attackerEnergyCost.zoneValues[index].enabled,
                defaultParameters.attackerEnergyCost.baseValue,
                base + "cell.function.attacker.energy cost",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerFoodChainColorMatrix.zoneValues[index].value,
                parameters.attackerFoodChainColorMatrix.zoneValues[index].enabled,
                defaultParameters.attackerFoodChainColorMatrix.baseValue,
                base + "cell.function.attacker.food chain color matrix",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerComplexCreatureProtection.zoneValues[index].value,
                parameters.attackerComplexCreatureProtection.zoneValues[index].enabled,
                defaultParameters.attackerComplexCreatureProtection.baseValue,
                base + "cell.function.attacker.genome size bonus",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerNewComplexMutantProtection.zoneValues[index].value,
                parameters.attackerNewComplexMutantProtection.zoneValues[index].enabled,
                defaultParameters.attackerNewComplexMutantProtection.baseValue,
                base + "cell.function.attacker.new complex mutant penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerGeometryDeviationProtection.zoneValues[index].value,
                parameters.attackerGeometryDeviationProtection.zoneValues[index].enabled,
                defaultParameters.attackerGeometryDeviationProtection.baseValue,
                base + "cell.function.attacker.geometry deviation exponent",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.attackerConnectionsMismatchProtection.zoneValues[index].value,
                parameters.attackerConnectionsMismatchProtection.zoneValues[index].enabled,
                defaultParameters.attackerConnectionsMismatchProtection.baseValue,
                base + "cell.function.attacker.connections mismatch penalty",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationNeuronData.zoneValues[index].value,
                parameters.copyMutationNeuronData.zoneValues[index].enabled,
                defaultParameters.copyMutationNeuronData.baseValue,
                base + "cell.copy mutation.neuron data",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationCellProperties.zoneValues[index].value,
                parameters.copyMutationCellProperties.zoneValues[index].enabled,
                defaultParameters.copyMutationCellProperties.baseValue,
                base + "cell.copy mutation.cell properties",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationGeometry.zoneValues[index].value,
                parameters.copyMutationGeometry.zoneValues[index].enabled,
                defaultParameters.copyMutationGeometry.baseValue,
                base + "cell.copy mutation.geometry",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationCustomGeometry.zoneValues[index].value,
                parameters.copyMutationCustomGeometry.zoneValues[index].enabled,
                defaultParameters.copyMutationCustomGeometry.baseValue,
                base + "cell.copy mutation.custom geometry",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationCellType.zoneValues[index].value,
                parameters.copyMutationCellType.zoneValues[index].enabled,
                defaultParameters.copyMutationCellType.baseValue,
                base + "cell.copy mutation.cell function",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationInsertion.zoneValues[index].value,
                parameters.copyMutationInsertion.zoneValues[index].enabled,
                defaultParameters.copyMutationInsertion.baseValue,
                base + "cell.copy mutation.insertion",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationDeletion.zoneValues[index].value,
                parameters.copyMutationDeletion.zoneValues[index].enabled,
                defaultParameters.copyMutationDeletion.baseValue,
                base + "cell.copy mutation.deletion",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationTranslation.zoneValues[index].value,
                parameters.copyMutationTranslation.zoneValues[index].enabled,
                defaultParameters.copyMutationTranslation.baseValue,
                base + "cell.copy mutation.translation",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationDuplication.zoneValues[index].value,
                parameters.copyMutationDuplication.zoneValues[index].enabled,
                defaultParameters.copyMutationDuplication.baseValue,
                base + "cell.copy mutation.duplication",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationCellColor.zoneValues[index].value,
                parameters.copyMutationCellColor.zoneValues[index].enabled,
                defaultParameters.copyMutationCellColor.baseValue,
                base + "cell.copy mutation.cell color",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationSubgenomeColor.zoneValues[index].value,
                parameters.copyMutationSubgenomeColor.zoneValues[index].enabled,
                defaultParameters.copyMutationSubgenomeColor.baseValue,
                base + "cell.copy mutation.subgenome color",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                parameters.copyMutationGenomeColor.zoneValues[index].value,
                parameters.copyMutationGenomeColor.zoneValues[index].enabled,
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
