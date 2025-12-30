#include "GenomeDescriptionValidationService.h"

#include <algorithm>
#include <cmath>

#include <Base/Math.h>

#include "EngineConstants.h"

void GenomeDescriptionValidationService::validateAndCorrect(GenomeDescription& genome)
{
    // Validate genome-level attributes
    // frontAngle is unbounded, so no validation needed
    genome._frontAngle = Math::modulo(genome._frontAngle, 360.0f);

    // Validate each gene
    for (auto& gene : genome._genes) {
        // Validate gene-level attributes
        gene._numBranches = std::max(gene._numBranches, 1);
        gene._numConcatenations = std::max(gene._numConcatenations, 1);
        gene._angleAlignment = std::clamp(gene._angleAlignment, 0, ConstructorAngleAlignment_Count - 1);
        gene._stiffness = std::max(gene._stiffness, 0.0f);
        gene._connectionDistance = std::max(gene._connectionDistance, 0.0f);
        gene._shape = std::clamp(gene._shape, 0, ConstructorShape_Count - 1);

        // Validate each node in the gene
        for (auto& node : gene._nodes) {
            // Validate node-level attributes
            node._color = std::clamp(node._color, 0, MAX_COLORS - 1);
            node._numAdditionalConnections = std::clamp(node._numAdditionalConnections, 0, MAX_CELL_BONDS - 1);

            // Validate neural network
            for (auto& activationFunction : node._neuralNetwork._activationFunctions) {
                activationFunction =
                    std::clamp(activationFunction, static_cast<ActivationFunction>(0), static_cast<ActivationFunction>(ActivationFunction_Count - 1));
            }

            // Validate signal restriction
            node._signalRestriction._baseAngle = Math::getNormalizedAngle(node._signalRestriction._baseAngle, -180.0f);
            node._signalRestriction._openingAngle = Math::modulo(node._signalRestriction._openingAngle, 360.0f);

            // Validate cell-specific attributes based on type
            auto nodeType = node.getCellType();

            if (nodeType == CellTypeGenome_Depot) {
                auto& depot = std::get<DepotGenomeDescription>(node._cellType);
                depot._storageLimit = std::clamp(depot._storageLimit, 0.0f, 1000.0f);
                depot._initialStoredUsableEnergy = std::clamp(depot._initialStoredUsableEnergy, 0.0f, depot._storageLimit);

            } else if (nodeType == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(node._cellType);
                if (constructor._autoTriggerInterval.has_value()) {
                    auto& value = constructor._autoTriggerInterval.value();
                    value = std::max(value, 1);
                }
                constructor._geneIndex = std::max(constructor._geneIndex, 0);
                constructor._constructionActivationTime = std::clamp(constructor._constructionActivationTime, 0, MAX_ACTIVATION_TIME);
                constructor._provideEnergy =
                    std::clamp(constructor._provideEnergy, static_cast<ProvideEnergy>(0), static_cast<ProvideEnergy>(ProvideEnergy_FreeGeneration));

            } else if (nodeType == CellTypeGenome_Sensor) {
                auto& sensor = std::get<SensorGenomeDescription>(node._cellType);
                if (sensor._autoTriggerInterval.has_value()) {
                    auto& value = sensor._autoTriggerInterval.value();
                    value = std::max(value, 1);
                }
                sensor._minRange = std::max(0, std::min(255, sensor._minRange));
                sensor._maxRange = std::max(0, std::min(255, sensor._maxRange));
                
                // Validate mode-specific data
                auto mode = sensor.getMode();
                if (mode == SensorMode_DetectEnergy) {
                    auto& detectEnergy = std::get<DetectEnergyGenomeDescription>(sensor._mode);
                    detectEnergy._minDensity = std::max(detectEnergy._minDensity, 0.0f);
                } else if (mode == SensorMode_DetectFreeCell) {
                    auto& detectFreeCell = std::get<DetectFreeCellGenomeDescription>(sensor._mode);
                    detectFreeCell._minDensity = std::clamp(detectFreeCell._minDensity, 0.0f, 1.0f);
                    if (detectFreeCell._restrictToColor.has_value()) {
                        auto& value = detectFreeCell._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                } else if (mode == SensorMode_DetectCreature) {
                    auto& detectCreature = std::get<DetectCreatureGenomeDescription>(sensor._mode);
                    if (detectCreature._minNumCells.has_value()) {
                        auto& value = detectCreature._minNumCells.value();
                        value = std::max(value, 0);
                    }
                    if (detectCreature._maxNumCells.has_value()) {
                        auto& value = detectCreature._maxNumCells.value();
                        value = std::max(value, 0);
                    }
                    if (detectCreature._restrictToColor.has_value()) {
                        auto& value = detectCreature._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                    detectCreature._restrictToLineage =
                        std::clamp(detectCreature._restrictToLineage, 0, LineageRestriction_Count - 1);
                }

            } else if (nodeType == CellTypeGenome_Generator) {
                auto& generator = std::get<GeneratorGenomeDescription>(node._cellType);
                generator._autoTriggerInterval = std::max(generator._autoTriggerInterval, 0);
                generator._pulseType = std::clamp(generator._pulseType, 0, GeneratorPulseType_Count - 1);
                generator._alternationInterval = std::max(generator._alternationInterval, 1);

            } else if (nodeType == CellTypeGenome_Attacker) {
                auto& attacker = std::get<AttackerGenomeDescription>(node._cellType);
                auto attackerMode = attacker.getMode();
                if (attackerMode == AttackerMode_FreeCell) {
                    auto& freeCell = std::get<AttackFreeCellGenomeDescription>(attacker._mode);
                    if (freeCell._restrictToColor.has_value()) {
                        auto& value = freeCell._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                } else if (attackerMode == AttackerMode_Creature) {
                    auto& creature = std::get<AttackCreatureGenomeDescription>(attacker._mode);
                    if (creature._minNumCells.has_value()) {
                        auto& value = creature._minNumCells.value();
                        value = std::max(value, 0);
                    }
                    if (creature._maxNumCells.has_value()) {
                        auto& value = creature._maxNumCells.value();
                        value = std::max(value, 0);
                    }
                    if (creature._restrictToColor.has_value()) {
                        auto& value = creature._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                    creature._restrictToLineage = std::clamp(creature._restrictToLineage, 0, LineageRestriction_Count - 1);
                }

            } else if (nodeType == CellTypeGenome_Injector) {
                auto& injector = std::get<InjectorGenomeDescription>(node._cellType);
                injector._geneIndex = std::max(injector._geneIndex, 0);

            } else if (nodeType == CellTypeGenome_Muscle) {
                auto& muscle = std::get<MuscleGenomeDescription>(node._cellType);

                // Validate muscle mode based on its variant type
                if (std::holds_alternative<AutoBendingGenomeDescription>(muscle._mode)) {
                    auto& mode = std::get<AutoBendingGenomeDescription>(muscle._mode);
                    mode._maxAngleDeviation = std::clamp(mode._maxAngleDeviation, 0.0f, 1.0f);
                    mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);

                } else if (std::holds_alternative<ManualBendingGenomeDescription>(muscle._mode)) {
                    auto& mode = std::get<ManualBendingGenomeDescription>(muscle._mode);
                    mode._maxAngleDeviation = std::clamp(mode._maxAngleDeviation, 0.0f, 1.0f);
                    mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);

                } else if (std::holds_alternative<AngleBendingGenomeDescription>(muscle._mode)) {
                    auto& mode = std::get<AngleBendingGenomeDescription>(muscle._mode);
                    mode._maxAngleDeviation = std::clamp(mode._maxAngleDeviation, 0.0f, 1.0f);
                    mode._attractionRepulsionRatio = std::clamp(mode._attractionRepulsionRatio, 0.0f, 1.0f);

                } else if (std::holds_alternative<AutoCrawlingGenomeDescription>(muscle._mode)) {
                    auto& mode = std::get<AutoCrawlingGenomeDescription>(muscle._mode);
                    mode._maxDistanceDeviation = std::clamp(mode._maxDistanceDeviation, 0.0f, 1.0f);
                    mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);

                } else if (std::holds_alternative<ManualCrawlingGenomeDescription>(muscle._mode)) {
                    auto& mode = std::get<ManualCrawlingGenomeDescription>(muscle._mode);
                    mode._maxDistanceDeviation = std::clamp(mode._maxDistanceDeviation, 0.0f, 1.0f);
                    mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);
                }
                // DirectMovementGenomeDescription has no attributes to validate

            } else if (nodeType == CellTypeGenome_Defender) {
                auto& defender = std::get<DefenderGenomeDescription>(node._cellType);
                defender._mode = std::clamp(defender._mode, 0, DefenderMode_Count - 1);

            } else if (nodeType == CellTypeGenome_Reconnector) {
                auto& reconnector = std::get<ReconnectorGenomeDescription>(node._cellType);
                auto reconnectorMode = reconnector.getMode();
                if (reconnectorMode == ReconnectorMode_FreeCell) {
                    auto& freeCell = std::get<ReconnectFreeCellGenomeDescription>(reconnector._mode);
                    if (freeCell._restrictToColor.has_value()) {
                        auto& value = freeCell._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                } else if (reconnectorMode == ReconnectorMode_Creature) {
                    auto& creature = std::get<ReconnectCreatureGenomeDescription>(reconnector._mode);
                    if (creature._minNumCells.has_value()) {
                        auto& value = creature._minNumCells.value();
                        value = std::max(value, 0);
                    }
                    if (creature._maxNumCells.has_value()) {
                        auto& value = creature._maxNumCells.value();
                        value = std::max(value, 0);
                    }
                    if (creature._restrictToColor.has_value()) {
                        auto& value = creature._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                    creature._restrictToLineage = std::clamp(creature._restrictToLineage, 0, LineageRestriction_Count - 1);
                }

            } else if (nodeType == CellTypeGenome_Detonator) {
                auto& detonator = std::get<DetonatorGenomeDescription>(node._cellType);
                detonator._countdown = std::max(detonator._countdown, 1);

            } else if (nodeType == CellTypeGenome_Memory) {
                auto& memory = std::get<MemoryGenomeDescription>(node._cellType);
                auto memoryMode = memory.getMode();
                if (memoryMode == MemoryMode_SignalDelay) {
                    auto& signalDelay = std::get<SignalDelayGenomeDescription>(memory._mode);
                    signalDelay._delay = std::clamp(signalDelay._delay, 1, MAX_CELL_MEMORY_ENTRIES);
                } else if (memoryMode == MemoryMode_SignalRecorder) {
                } else if (memoryMode == MemoryMode_SignalStorage) {
                } else if (memoryMode == MemoryMode_SignalIntegrator) {
                    auto& signalIntegrator = std::get<SignalIntegratorGenomeDescription>(memory._mode);
                    signalIntegrator._newSignalWeight = std::clamp(signalIntegrator._newSignalWeight, 0.0f, 1.0f);
                }
                // Validate number of memory entries
                auto numEntries = memory._memoryEntries.size();
                if (numEntries > MAX_CELL_MEMORY_ENTRIES) {
                    memory._memoryEntries.resize(MAX_CELL_MEMORY_ENTRIES);
                }
            }
        }
    }
}
