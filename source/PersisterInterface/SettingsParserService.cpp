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
            tree, parameters.baseValues.backgroundColor, defaultParameters.baseValues.backgroundColor, "simulation parameters.background color", parserTask);
        ParameterParser::encodeDecode(tree, parameters.primaryCellColoring, defaultParameters.primaryCellColoring, "simulation parameters.cell colorization", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellGlowColoring, defaultParameters.cellGlowColoring, "simulation parameters.cell glow.coloring", parserTask);
        ParameterParser::encodeDecode(tree, parameters.cellGlowRadius, defaultParameters.cellGlowRadius, "simulation parameters.cell glow.radius", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.cellGlowStrength, defaultParameters.cellGlowStrength, "simulation parameters.cell glow.strength", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.highlightedCellType, defaultParameters.highlightedCellType, "simulation parameters.highlighted cell function", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.zoomLevelForNeuronVisualization,
            defaultParameters.zoomLevelForNeuronVisualization,
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

        ParameterParser::encodeDecode(tree, parameters.motionType, defaultParameters.motionType, "simulation parameters.motion.type", parserTask);
        if (parameters.motionType == MotionType_Fluid) {
            ParameterParser::encodeDecode(
                tree,
                parameters.smoothingLength,
                defaultParameters.smoothingLength,
                "simulation parameters.fluid.smoothing length",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.pressureStrength,
                defaultParameters.pressureStrength,
                "simulation parameters.fluid.pressure strength",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.viscosityStrength,
                defaultParameters.viscosityStrength,
                "simulation parameters.fluid.viscosity strength",
                parserTask);
        } else {
            ParameterParser::encodeDecode(
                tree,
                parameters.maxCollisionDistance,
                defaultParameters.maxCollisionDistance,
                "simulation parameters.motion.collision.max distance",
                parserTask);
            ParameterParser::encodeDecode(
                tree,
                parameters.repulsionStrength,
                defaultParameters.repulsionStrength,
                "simulation parameters.motion.collision.repulsion strength",
                parserTask);
        }

        ParameterParser::encodeDecode(
            tree, parameters.baseValues.friction, defaultParameters.baseValues.friction, "simulation parameters.friction", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.baseValues.rigidity, defaultParameters.baseValues.rigidity, "simulation parameters.rigidity", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.maxVelocity, defaultParameters.maxVelocity, "simulation parameters.cell.max velocity", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.maxBindingDistance, defaultParameters.maxBindingDistance, "simulation parameters.cell.max binding distance", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.normalCellEnergy, defaultParameters.normalCellEnergy, "simulation parameters.cell.normal energy", parserTask);

        ParameterParser::encodeDecode(
            tree, parameters.minCellDistance, defaultParameters.minCellDistance, "simulation parameters.cell.min distance", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.baseValues.cellMaxForce, defaultParameters.baseValues.cellMaxForce, "simulation parameters.cell.max force", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.baseValues.minCellEnergy, defaultParameters.baseValues.minCellEnergy, "simulation parameters.cell.min energy", parserTask);
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
        ParameterParser::encodeDecode(tree, parameters.maxCellAge, defaultParameters.maxCellAge, "simulation parameters.cell.max age", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.maxCellAgeBalancerActivated, defaultParameters.maxCellAgeBalancerActivated, "simulation parameters.cell.max age.balance.enabled", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.maxCellAgeBalancerInterval,
            defaultParameters.maxCellAgeBalancerInterval,
            "simulation parameters.cell.max age.balance.interval",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.maxAgeForInactiveCellsActivated,
            defaultParameters.maxAgeForInactiveCellsActivated,
            "simulation parameters.cell.inactive max age activated",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.maxAgeForInactiveCells,
            defaultParameters.baseValues.maxAgeForInactiveCells,
            "simulation parameters.cell.inactive max age",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.freeCellMaxAgeActivated,
            defaultParameters.freeCellMaxAgeActivated,
            "simulation parameters.cell.nutrient max age activated",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.freeCellMaxAge, defaultParameters.freeCellMaxAge, "simulation parameters.cell.nutrient max age", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.resetCellAgeAfterActivation,
            defaultParameters.resetCellAgeAfterActivation,
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
            parameters.genomeComplexityDepthLevel,
            defaultParameters.genomeComplexityDepthLevel,
            "simulation parameters.genome complexity.genome complexity depth level",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.radiationType1_strength,
            defaultParameters.baseValues.radiationType1_strength,
            "simulation parameters.radiation.factor",
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
            parameters.radiationType2_energyThreshold,
            defaultParameters.radiationType2_energyThreshold,
            "simulation parameters.high radiation.min cell energy",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.radiationType2_strength, defaultParameters.radiationType2_strength, "simulation parameters.high radiation.factor", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.radiationType1_minimumAge, defaultParameters.radiationType1_minimumAge, "simulation parameters.radiation.min cell age", parserTask);

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
            parameters.externalEnergyInflowOnlyForNonSelfReplicators,
            defaultParameters.externalEnergyInflowOnlyForNonSelfReplicators,
            "simulation parameters.cell.function.constructor.externalEnergyInflowOnlyForNonSelfReplicators",
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
            parameters.constructorConnectingCellDistance,
            defaultParameters.constructorConnectingCellDistance,
            "simulation parameters.cell.function.constructor.connecting cell max distance",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.constructorCompletenessCheck,
            defaultParameters.constructorCompletenessCheck,
            "simulation parameters.cell.function.constructor.completeness check for self-replication",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationNeuronData,
            defaultParameters.baseValues.copyMutationNeuronData,
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
            parameters.baseValues.copyMutationCellProperties,
            defaultParameters.baseValues.copyMutationCellProperties,
            "simulation parameters.cell.copy mutation.cell properties",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationGeometry,
            defaultParameters.baseValues.copyMutationGeometry,
            "simulation parameters.cell.copy mutation.geometry",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationCustomGeometry,
            defaultParameters.baseValues.copyMutationCustomGeometry,
            "simulation parameters.cell.copy mutation.custom geometry",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationCellType,
            defaultParameters.baseValues.copyMutationCellType,
            "simulation parameters.cell.copy mutation.cell function",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationInsertion,
            defaultParameters.baseValues.copyMutationInsertion,
            "simulation parameters.cell.copy mutation.insertion",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationDeletion,
            defaultParameters.baseValues.copyMutationDeletion,
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
            parameters.baseValues.copyMutationTranslation,
            defaultParameters.baseValues.copyMutationTranslation,
            "simulation parameters.cell.copy mutation.translation",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationDuplication,
            defaultParameters.baseValues.copyMutationDuplication,
            "simulation parameters.cell.copy mutation.duplication",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationCellColor,
            defaultParameters.baseValues.copyMutationCellColor,
            "simulation parameters.cell.copy mutation.cell color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationSubgenomeColor,
            defaultParameters.baseValues.copyMutationSubgenomeColor,
            "simulation parameters.cell.copy mutation.subgenome color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.copyMutationGenomeColor,
            defaultParameters.baseValues.copyMutationGenomeColor,
            "simulation parameters.cell.copy mutation.genome color",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationColorTransitions,
            defaultParameters.copyMutationColorTransitions,
            "simulation parameters.cell.copy mutation.color transition",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationSelfReplication,
            defaultParameters.copyMutationSelfReplication,
            "simulation parameters.cell.copy mutation.self replication flag",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.copyMutationPreventDepthIncrease,
            defaultParameters.copyMutationPreventDepthIncrease,
            "simulation parameters.cell.copy mutation.prevent depth increase",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.injectorInjectionRadius,
            defaultParameters.injectorInjectionRadius,
            "simulation parameters.cell.function.injector.radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.injectorInjectionTime,
            defaultParameters.injectorInjectionTime,
            "simulation parameters.cell.function.injector.duration",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.attackerRadius,
            defaultParameters.attackerRadius,
            "simulation parameters.cell.function.attacker.radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerStrength,
            defaultParameters.attackerStrength,
            "simulation parameters.cell.function.attacker.strength",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.attackerEnergyCost,
            defaultParameters.baseValues.attackerEnergyCost,
            "simulation parameters.cell.function.attacker.energy cost",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.attackerGeometryDeviationProtection,
            defaultParameters.baseValues.attackerGeometryDeviationProtection,
            "simulation parameters.cell.function.attacker.geometry deviation exponent",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.attackerFoodChainColorMatrix,
            defaultParameters.baseValues.attackerFoodChainColorMatrix,
            "simulation parameters.cell.function.attacker.food chain color matrix",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.attackerConnectionsMismatchProtection,
            defaultParameters.baseValues.attackerConnectionsMismatchProtection,
            "simulation parameters.cell.function.attacker.connections mismatch penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.attackerCreatureProtection,
            defaultParameters.baseValues.attackerCreatureProtection,
            "simulation parameters.cell.function.attacker.genome size bonus",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerSameMutantProtection,
            defaultParameters.attackerSameMutantProtection,
            "simulation parameters.cell.function.attacker.same mutant penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.baseValues.attackerNewComplexMutantProtection,
            defaultParameters.baseValues.attackerNewComplexMutantProtection,
            "simulation parameters.cell.function.attacker.new complex mutant penalty",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerSensorDetectionFactor,
            defaultParameters.attackerSensorDetectionFactor,
            "simulation parameters.cell.function.attacker.sensor detection factor",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.attackerDestroyCells,
            defaultParameters.attackerDestroyCells,
            "simulation parameters.cell.function.attacker.destroy cells",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.defenderAntiAttackerStrength,
            defaultParameters.defenderAntiAttackerStrength,
            "simulation parameters.cell.function.defender.against attacker strength",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.defenderAntiInjectorStrength,
            defaultParameters.defenderAntiInjectorStrength,
            "simulation parameters.cell.function.defender.against injector strength",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.transmitterEnergyDistributionSameCreature,
            defaultParameters.transmitterEnergyDistributionSameCreature,
            "simulation parameters.cell.function.transmitter.energy distribution same creature",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.transmitterEnergyDistributionRadius,
            defaultParameters.transmitterEnergyDistributionRadius,
            "simulation parameters.cell.function.transmitter.energy distribution radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.transmitterEnergyDistributionValue,
            defaultParameters.transmitterEnergyDistributionValue,
            "simulation parameters.cell.function.transmitter.energy distribution value",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.muscleCrawlingAcceleration,
            defaultParameters.muscleCrawlingAcceleration,
            "simulation parameters.cell.function.muscle.crawling acceleration",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.muscleMovementAcceleration,
            defaultParameters.muscleMovementAcceleration,
            "simulation parameters.cell.function.muscle.movement acceleration",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.muscleBendingAcceleration,
            defaultParameters.muscleBendingAcceleration,
            "simulation parameters.cell.function.muscle.bending acceleration",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.muscleEnergyCost,
            defaultParameters.muscleEnergyCost,
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
            parameters.sensorRadius,
            defaultParameters.sensorRadius,
            "simulation parameters.cell.function.sensor.range",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.reconnectorRadius,
            defaultParameters.reconnectorRadius,
            "simulation parameters.cell.function.reconnector.radius",
            parserTask);

        ParameterParser::encodeDecode(
            tree,
            parameters.detonatorRadius,
            defaultParameters.detonatorRadius,
            "simulation parameters.cell.function.detonator.radius",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.detonatorChainExplosionProbability,
            defaultParameters.detonatorChainExplosionProbability,
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
            ParameterParser::encodeDecode(tree, spot.values.backgroundColor, defaultSpot.values.backgroundColor, base + "color", parserTask);
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
                spot.values.radiationType1_strength,
                spot.activatedValues.radiationCellAgeStrength,
                defaultSpot.values.radiationType1_strength,
                base + "radiation.factor",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree, spot.values.cellMaxForce, spot.activatedValues.cellMaxForce, defaultSpot.values.cellMaxForce, base + "cell.max force", parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree, spot.values.minCellEnergy, spot.activatedValues.cellMinEnergy, defaultSpot.values.minCellEnergy, base + "cell.min energy", parserTask);
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
                spot.values.maxAgeForInactiveCells,
                spot.activatedValues.cellInactiveMaxAge,
                defaultSpot.values.maxAgeForInactiveCells,
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
                spot.values.attackerEnergyCost,
                spot.activatedValues.cellTypeAttackerEnergyCost,
                defaultSpot.values.attackerEnergyCost,
                base + "cell.function.attacker.energy cost",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.attackerFoodChainColorMatrix,
                spot.activatedValues.cellTypeAttackerFoodChainColorMatrix,
                defaultSpot.values.attackerFoodChainColorMatrix,
                base + "cell.function.attacker.food chain color matrix",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.attackerCreatureProtection,
                spot.activatedValues.cellTypeAttackerGenomeComplexityBonus,
                defaultSpot.values.attackerCreatureProtection,
                base + "cell.function.attacker.genome size bonus",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.attackerNewComplexMutantProtection,
                spot.activatedValues.cellTypeAttackerNewComplexMutantPenalty,
                defaultSpot.values.attackerNewComplexMutantProtection,
                base + "cell.function.attacker.new complex mutant penalty",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.attackerGeometryDeviationProtection,
                spot.activatedValues.cellTypeAttackerGeometryDeviationExponent,
                defaultSpot.values.attackerGeometryDeviationProtection,
                base + "cell.function.attacker.geometry deviation exponent",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.attackerConnectionsMismatchProtection,
                spot.activatedValues.cellTypeAttackerConnectionsMismatchPenalty,
                defaultSpot.values.attackerConnectionsMismatchProtection,
                base + "cell.function.attacker.connections mismatch penalty",
                parserTask);

            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationNeuronData,
                spot.activatedValues.cellCopyMutationNeuronData,
                defaultSpot.values.copyMutationNeuronData,
                base + "cell.copy mutation.neuron data",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationCellProperties,
                spot.activatedValues.cellCopyMutationCellProperties,
                defaultSpot.values.copyMutationCellProperties,
                base + "cell.copy mutation.cell properties",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationGeometry,
                spot.activatedValues.cellCopyMutationGeometry,
                defaultSpot.values.copyMutationGeometry,
                base + "cell.copy mutation.geometry",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationCustomGeometry,
                spot.activatedValues.cellCopyMutationCustomGeometry,
                defaultSpot.values.copyMutationCustomGeometry,
                base + "cell.copy mutation.custom geometry",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationCellType,
                spot.activatedValues.cellCopyMutationCellType,
                defaultSpot.values.copyMutationCellType,
                base + "cell.copy mutation.cell function",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationInsertion,
                spot.activatedValues.cellCopyMutationInsertion,
                defaultSpot.values.copyMutationInsertion,
                base + "cell.copy mutation.insertion",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationDeletion,
                spot.activatedValues.cellCopyMutationDeletion,
                defaultSpot.values.copyMutationDeletion,
                base + "cell.copy mutation.deletion",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationTranslation,
                spot.activatedValues.cellCopyMutationTranslation,
                defaultSpot.values.copyMutationTranslation,
                base + "cell.copy mutation.translation",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationDuplication,
                spot.activatedValues.cellCopyMutationDuplication,
                defaultSpot.values.copyMutationDuplication,
                base + "cell.copy mutation.duplication",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationCellColor,
                spot.activatedValues.cellCopyMutationCellColor,
                defaultSpot.values.copyMutationCellColor,
                base + "cell.copy mutation.cell color",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationSubgenomeColor,
                spot.activatedValues.cellCopyMutationSubgenomeColor,
                defaultSpot.values.copyMutationSubgenomeColor,
                base + "cell.copy mutation.subgenome color",
                parserTask);
            ParameterParser::encodeDecodeWithEnabled(
                tree,
                spot.values.copyMutationGenomeColor,
                spot.activatedValues.cellCopyMutationGenomeColor,
                defaultSpot.values.copyMutationGenomeColor,
                base + "cell.copy mutation.genome color",
                parserTask);
        }

        //features
        ParameterParser::encodeDecode(
            tree,
            parameters.expertSettingsToggles.genomeComplexityMeasurement,
            defaultParameters.expertSettingsToggles.genomeComplexityMeasurement,
            "simulation parameters.features.genome complexity measurement",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.expertSettingsToggles.advancedAbsorptionControl,
            defaultParameters.expertSettingsToggles.advancedAbsorptionControl,
            "simulation parameters.features.additional absorption control",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.expertSettingsToggles.advancedAttackerControl,
            defaultParameters.expertSettingsToggles.advancedAttackerControl,
            "simulation parameters.features.additional attacker control",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.expertSettingsToggles.externalEnergyControl,
            defaultParameters.expertSettingsToggles.externalEnergyControl,
            "simulation parameters.features.external energy",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.expertSettingsToggles.cellColorTransitionRules,
            defaultParameters.expertSettingsToggles.cellColorTransitionRules,
            "simulation parameters.features.cell color transition rules",
            parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.expertSettingsToggles.cellAgeLimiter, defaultParameters.expertSettingsToggles.cellAgeLimiter, "simulation parameters.features.cell age limiter", parserTask);
        ParameterParser::encodeDecode(
            tree, parameters.expertSettingsToggles.cellGlow, defaultParameters.expertSettingsToggles.cellGlow, "simulation parameters.features.cell glow", parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.expertSettingsToggles.customizeNeuronMutations,
            defaultParameters.expertSettingsToggles.customizeNeuronMutations,
            "simulation parameters.features.customize neuron mutations",
            parserTask);
        ParameterParser::encodeDecode(
            tree,
            parameters.expertSettingsToggles.customizeDeletionMutations,
            defaultParameters.expertSettingsToggles.customizeDeletionMutations,
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
