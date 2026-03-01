#include "GenomeDescValidationService.h"

#include <algorithm>
#include <cmath>

#include <Base/Math.h>

#include "EngineConstants.h"

void GenomeDescValidationService::validateAndCorrect(GenomeDesc& genome)
{
    // Validate genome-level attributes
    // frontAngle is unbounded, so no validation needed
    genome._frontAngle = Math::modulo(genome._frontAngle, 360.0f);

    // Validate mutation rate fields
    genome._lineageId = std::max(genome._lineageId, 0);
    genome._prevLineageId = std::max(genome._prevLineageId, 0);
    genome._lineageMutationProbability = std::clamp(genome._lineageMutationProbability, 0.0f, 1.0f);
    genome._neuronMutation1._probability = std::clamp(genome._neuronMutation1._probability, 0.0f, 1.0f);
    genome._neuronMutation1._weightSigma = std::max(genome._neuronMutation1._weightSigma, 0.0f);
    genome._neuronMutation1._biasSigma = std::max(genome._neuronMutation1._biasSigma, 0.0f);
    genome._neuronMutation1._activationFunctionProbability = std::clamp(genome._neuronMutation1._activationFunctionProbability, 0.0f, 1.0f);
    genome._neuronMutation2._probability = std::clamp(genome._neuronMutation2._probability, 0.0f, 1.0f);
    genome._neuronMutation2._weightSigma = std::max(genome._neuronMutation2._weightSigma, 0.0f);
    genome._neuronMutation2._biasSigma = std::max(genome._neuronMutation2._biasSigma, 0.0f);
    genome._neuronMutation2._activationFunctionProbability = std::clamp(genome._neuronMutation2._activationFunctionProbability, 0.0f, 1.0f);
    genome._connectionMutationRate1._probability = std::clamp(genome._connectionMutationRate1._probability, 0.0f, 1.0f);
    genome._connectionMutationRate1._sigma = std::max(genome._connectionMutationRate1._sigma, 0.0f);
    genome._connectionMutationRate2._probability = std::clamp(genome._connectionMutationRate2._probability, 0.0f, 1.0f);
    genome._connectionMutationRate2._sigma = std::max(genome._connectionMutationRate2._sigma, 0.0f);

    // Validate each gene
    for (auto& gene : genome._genes) {
        // Validate gene-level attributes
        gene._numBranches = std::max(gene._numBranches, 1);
        gene._numConcatenations = std::max(gene._numConcatenations, 1);
        gene._angleAlignment = std::clamp(gene._angleAlignment, 0, ConstructorAngleAlignment_Count - 1);
        gene._stiffness = std::max(gene._stiffness, 0.0f);
        gene._connectionDistance = std::clamp(gene._connectionDistance, 0.5f, 2.0f);
        gene._shape = std::clamp(gene._shape, 1, ConstructorShape_Count - 1);

        // Validate each node in the gene
        for (auto& node : gene._nodes) {
            // Validate node-level attributes
            node._color = std::clamp(node._color, 0, MAX_COLORS - 1);
            node._numAdditionalConnections = std::clamp(node._numAdditionalConnections, 0, MAX_OBJECT_CONNECTIONS - 1);

            // Validate neural network
            for (auto& activationFunction : node._neuralNetwork._activationFunctions) {
                activationFunction =
                    std::clamp(activationFunction, static_cast<ActivationFunction>(0), static_cast<ActivationFunction>(ActivationFunction_Count - 1));
            }
            for (auto& bias : node._neuralNetwork._biases) {
                bias = std::clamp(bias, -2.0f, 2.0f);
            }
            for (auto& weight : node._neuralNetwork._connectionWeights) {
                weight = std::clamp(weight, -1.0f, 1.0f);
            }

            // Validate cell-specific attributes based on type
            auto nodeType = node.getCellType();

            if (nodeType == CellType_Depot) {
                auto& depot = std::get<DepotGenomeDesc>(node._cellType);
                depot._storageLimit = std::clamp(depot._storageLimit, 0.0f, 1000.0f);
                depot._initialStoredUsableEnergy = std::clamp(depot._initialStoredUsableEnergy, 0.0f, depot._storageLimit);

            } else if (nodeType == CellType_Sensor) {
                auto& sensor = std::get<SensorGenomeDesc>(node._cellType);
                sensor._minRange = std::max(0, std::min(512, sensor._minRange));
                sensor._maxRange = std::max(0, std::min(512, sensor._maxRange));

                // Validate mode-specific data
                auto mode = sensor.getMode();
                if (mode == SensorMode_DetectEnergy) {
                    auto& detectEnergy = std::get<DetectEnergyGenomeDesc>(sensor._mode);
                    detectEnergy._minDensity = std::max(detectEnergy._minDensity, 0.0f);
                } else if (mode == SensorMode_DetectFreeCell) {
                    auto& detectFreeCell = std::get<DetectFreeCellGenomeDesc>(sensor._mode);
                    detectFreeCell._minDensity = std::clamp(detectFreeCell._minDensity, 0.0f, 1.0f);
                    if (detectFreeCell._restrictToColor.has_value()) {
                        auto& value = detectFreeCell._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                } else if (mode == SensorMode_DetectCreature) {
                    auto& detectCreature = std::get<DetectCreatureGenomeDesc>(sensor._mode);
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
                    detectCreature._restrictToLineage = std::clamp(detectCreature._restrictToLineage, 0, LineageRestriction_Count - 1);
                }

            } else if (nodeType == CellType_Generator) {
                auto& generator = std::get<GeneratorGenomeDesc>(node._cellType);
                generator._valueOffset = std::clamp(generator._valueOffset, -2.0f, 2.0f);
                generator._timeOffset = std::max(generator._timeOffset, 0);
                // Validate mode-specific data
                auto generatorMode = generator.getMode();
                if (generatorMode == GeneratorMode_SquareSignal) {
                    auto& squareSignal = std::get<SquareSignalGenomeDesc>(generator._mode);
                    squareSignal._period = std::max(squareSignal._period, 1);
                    squareSignal._amplitude = std::clamp(squareSignal._amplitude, 0.0f, 2.0f);
                } else if (generatorMode == GeneratorMode_SawtoothSignal) {
                    auto& sawtoothSignal = std::get<SawtoothSignalGenomeDesc>(generator._mode);
                    sawtoothSignal._period = std::max(sawtoothSignal._period, 1);
                    sawtoothSignal._amplitude = std::clamp(sawtoothSignal._amplitude, 0.0f, 4.0f);
                }

            } else if (nodeType == CellType_Attacker) {
                auto& attacker = std::get<AttackerGenomeDesc>(node._cellType);
                auto attackerMode = attacker.getMode();
                if (attackerMode == AttackerMode_FreeCell) {
                    auto& freeCell = std::get<AttackFreeCellGenomeDesc>(attacker._mode);
                    if (freeCell._restrictToColor.has_value()) {
                        auto& value = freeCell._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                }

            } else if (nodeType == CellType_Injector) {
                auto& injector = std::get<InjectorGenomeDesc>(node._cellType);
                injector._geneIndex = std::max(injector._geneIndex, 0);

            } else if (nodeType == CellType_Muscle) {
                auto& muscle = std::get<MuscleGenomeDesc>(node._cellType);

                // Validate muscle mode based on its variant type
                if (std::holds_alternative<AutoBendingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<AutoBendingGenomeDesc>(muscle._mode);
                    mode._maxAngleDeviation = std::clamp(mode._maxAngleDeviation, 0.0f, 1.0f);
                    mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);

                } else if (std::holds_alternative<ManualBendingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<ManualBendingGenomeDesc>(muscle._mode);
                    mode._maxAngleDeviation = std::clamp(mode._maxAngleDeviation, 0.0f, 1.0f);
                    mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);

                } else if (std::holds_alternative<AngleBendingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<AngleBendingGenomeDesc>(muscle._mode);
                    mode._maxAngleDeviation = std::clamp(mode._maxAngleDeviation, 0.0f, 1.0f);
                    mode._attractionRepulsionRatio = std::clamp(mode._attractionRepulsionRatio, 0.0f, 1.0f);

                } else if (std::holds_alternative<AutoCrawlingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<AutoCrawlingGenomeDesc>(muscle._mode);
                    mode._maxDistanceDeviation = std::clamp(mode._maxDistanceDeviation, 0.0f, 1.0f);
                    mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);

                } else if (std::holds_alternative<ManualCrawlingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<ManualCrawlingGenomeDesc>(muscle._mode);
                    mode._maxDistanceDeviation = std::clamp(mode._maxDistanceDeviation, 0.0f, 1.0f);
                    mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);
                }
                // DirectMovementGenomeDesc has no attributes to validate

            } else if (nodeType == CellType_Defender) {
                auto& defender = std::get<DefenderGenomeDesc>(node._cellType);
                defender._mode = std::clamp(defender._mode, 0, DefenderMode_Count - 1);

            } else if (nodeType == CellType_Reconnector) {
                auto& reconnector = std::get<ReconnectorGenomeDesc>(node._cellType);
                auto reconnectorMode = reconnector.getMode();
                if (reconnectorMode == ReconnectorMode_FreeCell) {
                    auto& freeCell = std::get<ReconnectFreeCellGenomeDesc>(reconnector._mode);
                    if (freeCell._restrictToColor.has_value()) {
                        auto& value = freeCell._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                } else if (reconnectorMode == ReconnectorMode_Creature) {
                    auto& creature = std::get<ReconnectCreatureGenomeDesc>(reconnector._mode);
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

            } else if (nodeType == CellType_Detonator) {
                auto& detonator = std::get<DetonatorGenomeDesc>(node._cellType);
                detonator._countdown = std::max(detonator._countdown, 1);

            } else if (nodeType == CellType_Memory) {
                auto& memory = std::get<MemoryGenomeDesc>(node._cellType);
                auto memoryMode = memory.getMode();
                if (memoryMode == MemoryMode_SignalDelay) {
                    auto& signalDelay = std::get<SignalDelayGenomeDesc>(memory._mode);
                    signalDelay._delay = std::clamp(signalDelay._delay, 0, MAX_CELL_MEMORY_ENTRIES);
                } else if (memoryMode == MemoryMode_SignalRecorder) {
                    auto& signalRecorder = std::get<SignalRecorderGenomeDesc>(memory._mode);
                    signalRecorder._numWrittenSignalEntries =
                        std::clamp(signalRecorder._numWrittenSignalEntries, 0, static_cast<int>(memory._signalEntries.size()));
                } else if (memoryMode == MemoryMode_SignalStorage) {
                } else if (memoryMode == MemoryMode_SignalIntegrator) {
                    auto& signalIntegrator = std::get<SignalIntegratorGenomeDesc>(memory._mode);
                    signalIntegrator._newSignalWeight = std::clamp(signalIntegrator._newSignalWeight, 0.0f, 1.0f);
                }
                // Validate number of memory entries
                auto numEntries = memory._signalEntries.size();
                if (numEntries > MAX_CELL_MEMORY_ENTRIES) {
                    memory._signalEntries.resize(MAX_CELL_MEMORY_ENTRIES);
                }

            } else if (nodeType == CellType_Communicator) {
                auto& communicator = std::get<CommunicatorGenomeDesc>(node._cellType);
                auto communicatorMode = communicator.getMode();
                if (communicatorMode == CommunicatorMode_Sender) {
                    auto& sender = std::get<SenderGenomeDesc>(communicator._mode);
                    sender._range = std::clamp(sender._range, 0.0f, 15.0f);
                    sender._maxTimesSent = std::max(sender._maxTimesSent, 0);
                } else if (communicatorMode == CommunicatorMode_Receiver) {
                    auto& receiver = std::get<ReceiverGenomeDesc>(communicator._mode);
                    if (receiver._restrictToColor.has_value()) {
                        auto& value = receiver._restrictToColor.value();
                        value = std::clamp(value, 0, MAX_COLORS - 1);
                    }
                    receiver._restrictToLineage = std::clamp(receiver._restrictToLineage, 0, LineageRestriction_Count - 1);
                }
            }

            // Validate optional constructor field
            if (node._constructor.has_value()) {
                auto& constructor = node._constructor.value();
                if (constructor._autoTriggerInterval.has_value()) {
                    auto& value = constructor._autoTriggerInterval.value();
                    value = std::max(value, 1);
                }
                constructor._geneIndex = std::max(constructor._geneIndex, 0);
                constructor._constructionActivationTime = std::clamp(constructor._constructionActivationTime, 0, MAX_ACTIVATION_TIME);
                constructor._provideEnergy =
                    std::clamp(constructor._provideEnergy, static_cast<ProvideEnergy>(0), static_cast<ProvideEnergy>(ProvideEnergy_Count - 1));
            }
        }
    }
}
