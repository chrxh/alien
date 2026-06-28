#include "DescValidationService.h"

#include <algorithm>
#include <cmath>

#include <Base/Math.h>

#include "EngineConstants.h"

void DescValidationService::validateAndCorrect(GenomeDesc& genome)
{
    // Validate genome-level attributes
    // frontAngle is unbounded, so no validation needed
    genome._frontAngle = Math::getNormalizedAngle(genome._frontAngle, -180.0f);

    // Validate mutation rate fields
    auto validateCellTypePropertiesMutation = [](CellTypePropertiesMutationDesc& mutation) {
        mutation._nodeProbability = std::clamp(mutation._nodeProbability, 0.0f, 1.0f);
        mutation._valueChangeSigma = std::clamp(mutation._valueChangeSigma, 0.0f, 1.0f);
        mutation._enumChangeProbability = std::clamp(mutation._enumChangeProbability, 0.0f, 1.0f);
    };
    auto validateConstructorMutation = [](ConstructorMutationDesc& mutation) {
        mutation._nodeProbability = std::clamp(mutation._nodeProbability, 0.0f, 1.0f);
        mutation._valueChangeSigma = std::clamp(mutation._valueChangeSigma, 0.0f, 1.0f);
        mutation._enumChangeProbability = std::clamp(mutation._enumChangeProbability, 0.0f, 1.0f);
        mutation._constructorToggleProbability = std::clamp(mutation._constructorToggleProbability, 0.0f, 1.0f);
    };
    for (int i = 0; i < 2; ++i) {
        auto& neuronMutation = genome._mutationRates._neuronMutations[i];
        neuronMutation._nodeProbability = std::clamp(neuronMutation._nodeProbability, 0.0f, 1.0f);
        neuronMutation._weightChangeSigma = std::max(neuronMutation._weightChangeSigma, 0.0f);
        neuronMutation._biasChangeSigma = std::max(neuronMutation._biasChangeSigma, 0.0f);
        neuronMutation._actfnChangeProbability = std::clamp(neuronMutation._actfnChangeProbability, 0.0f, 1.0f);

        auto& connectionMutation = genome._mutationRates._connectionMutations[i];
        connectionMutation._nodeProbability = std::clamp(connectionMutation._nodeProbability, 0.0f, 1.0f);
        connectionMutation._valueChangeSigma = std::max(connectionMutation._valueChangeSigma, 0.0f);

        validateCellTypePropertiesMutation(genome._mutationRates._cellTypePropertiesMutations[i]);
        validateConstructorMutation(genome._mutationRates._constructorMutations[i]);
    }
    genome._mutationRates._cellTypeModeMutation._nodeProbability =
        std::clamp(genome._mutationRates._cellTypeModeMutation._nodeProbability, 0.0f, 1.0f);
    genome._mutationRates._cellTypeMutation._nodeProbability =
        std::clamp(genome._mutationRates._cellTypeMutation._nodeProbability, 0.0f, 1.0f);
    genome._mutationRates._voidMutation._nodeProbability =
        std::clamp(genome._mutationRates._voidMutation._nodeProbability, 0.0f, 1.0f);
    genome._mutationRates._appendNodeMutation._geneProbability =
        std::clamp(genome._mutationRates._appendNodeMutation._geneProbability, 0.0f, 1.0f);
    genome._mutationRates._addNodeMutation._geneProbability =
        std::clamp(genome._mutationRates._addNodeMutation._geneProbability, 0.0f, 1.0f);
    genome._mutationRates._trimNodeMutation._geneProbability =
        std::clamp(genome._mutationRates._trimNodeMutation._geneProbability, 0.0f, 1.0f);
    genome._mutationRates._deleteNodeMutation._geneProbability =
        std::clamp(genome._mutationRates._deleteNodeMutation._geneProbability, 0.0f, 1.0f);
    genome._mutationRates._duplicateGeneMutation._geneProbability =
        std::clamp(genome._mutationRates._duplicateGeneMutation._geneProbability, 0.0f, 1.0f);
    genome._mutationRates._deleteGeneMutation._geneProbability =
        std::clamp(genome._mutationRates._deleteGeneMutation._geneProbability, 0.0f, 1.0f);

    // Validate each gene
    for (auto& gene : genome._genes) {
        // Validate gene-level attributes
        gene._stiffness = std::max(gene._stiffness, 0.05f);
        gene._connectionDistance = std::clamp(gene._connectionDistance, 0.5f, 1.5f);
        gene._shape = std::clamp(gene._shape, 0, ConstructorShape_Count - 1);

        // Validate each node in the gene
        for (auto& node : gene._nodes) {
            // Validate node-level attributes
            node._color = std::clamp(node._color, 0, MAX_COLORS - 1);

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
                depot._storageLimit =
                    std::clamp(depot._storageLimit, Const::DepotStorageLimit_Min, Const::DepotStorageLimit_Max);
                depot._initialStoredUsableEnergy =
                    std::clamp(depot._initialStoredUsableEnergy, Const::DepotInitialStoredUsableEnergy_Min, depot._storageLimit);

            } else if (nodeType == CellType_Sensor) {
                auto& sensor = std::get<SensorGenomeDesc>(node._cellType);
                sensor._minRange = std::clamp(sensor._minRange, Const::SensorRange_Min, Const::SensorRange_Max);
                sensor._maxRange = std::clamp(sensor._maxRange, Const::SensorRange_Min, Const::SensorRange_Max);

                // Validate mode-specific data
                auto mode = sensor.getMode();
                if (mode == SensorMode_DetectEnergy) {
                    auto& detectEnergy = std::get<DetectEnergyGenomeDesc>(sensor._mode);
                    detectEnergy._minDensity = std::max(
                        detectEnergy._minDensity, Const::DetectEnergyMinDensity_Min);
                } else if (mode == SensorMode_DetectFreeCell) {
                    auto& detectFreeCell = std::get<DetectFreeCellGenomeDesc>(sensor._mode);
                    detectFreeCell._minDensity = std::clamp(
                        detectFreeCell._minDensity, Const::DetectFreeCellMinDensity_Min, Const::DetectFreeCellMinDensity_Max);
                    detectFreeCell._restrictToColors = std::clamp(
                        detectFreeCell._restrictToColors, Const::RestrictToColors_Min, Const::RestrictToColors_Max);
                } else if (mode == SensorMode_DetectCreature) {
                    auto& detectCreature = std::get<DetectCreatureGenomeDesc>(sensor._mode);
                    if (detectCreature._minNumCells.has_value()) {
                        auto& value = detectCreature._minNumCells.value();
                        value = std::max(value, Const::CreatureNumCells_Min);
                    }
                    if (detectCreature._maxNumCells.has_value()) {
                        auto& value = detectCreature._maxNumCells.value();
                        value = std::max(value, Const::CreatureNumCells_Min);
                    }
                    detectCreature._restrictToColors = std::clamp(
                        detectCreature._restrictToColors, Const::RestrictToColors_Min, Const::RestrictToColors_Max);
                    detectCreature._restrictToLineage = std::clamp(detectCreature._restrictToLineage, 0, LineageRestriction_Count - 1);
                }

            } else if (nodeType == CellType_Generator) {
                auto& generator = std::get<GeneratorGenomeDesc>(node._cellType);
                generator._minValue = std::clamp(generator._minValue, Const::GeneratorValue_Min, Const::GeneratorValue_Max);
                generator._maxValue = std::clamp(generator._maxValue, Const::GeneratorValue_Min, Const::GeneratorValue_Max);
                if (generator._minValue > generator._maxValue) {
                    std::swap(generator._minValue, generator._maxValue);
                }
                generator._timeOffset = std::max(generator._timeOffset, Const::GeneratorTimeOffset_Min);

                // Validate mode-specific data
                auto generatorMode = generator.getMode();
                if (generatorMode == GeneratorMode_SquareSignal) {
                    auto& squareSignal = std::get<SquareSignalGenomeDesc>(generator._mode);
                    squareSignal._period = std::max(squareSignal._period, Const::GeneratorPeriod_Min);
                } else if (generatorMode == GeneratorMode_SawtoothSignal) {
                    auto& sawtoothSignal = std::get<SawtoothSignalGenomeDesc>(generator._mode);
                    sawtoothSignal._period = std::max(sawtoothSignal._period, Const::GeneratorPeriod_Min);
                }

            } else if (nodeType == CellType_Attacker) {
                auto& attacker = std::get<AttackerGenomeDesc>(node._cellType);
                auto attackerMode = attacker.getMode();
                if (attackerMode == AttackerMode_FreeCell) {
                    auto& freeCell = std::get<AttackFreeCellGenomeDesc>(attacker._mode);
                    freeCell._restrictToColors =
                        std::clamp(freeCell._restrictToColors, Const::RestrictToColors_Min, Const::RestrictToColors_Max);
                }

            } else if (nodeType == CellType_Injector) {
                auto& injector = std::get<InjectorGenomeDesc>(node._cellType);
                injector._geneIndex = std::max(injector._geneIndex, 0);

            } else if (nodeType == CellType_Muscle) {
                auto& muscle = std::get<MuscleGenomeDesc>(node._cellType);

                // Validate muscle mode based on its variant type
                if (std::holds_alternative<AutoBendingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<AutoBendingGenomeDesc>(muscle._mode);
                    mode._maxAngleDeviation =
                        std::clamp(mode._maxAngleDeviation, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);
                    mode._forwardBackwardRatio =
                        std::clamp(mode._forwardBackwardRatio, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);

                } else if (std::holds_alternative<ManualBendingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<ManualBendingGenomeDesc>(muscle._mode);
                    mode._maxAngleDeviation =
                        std::clamp(mode._maxAngleDeviation, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);
                    mode._forwardBackwardRatio =
                        std::clamp(mode._forwardBackwardRatio, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);

                } else if (std::holds_alternative<AngleBendingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<AngleBendingGenomeDesc>(muscle._mode);
                    mode._maxAngleDeviation =
                        std::clamp(mode._maxAngleDeviation, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);
                    mode._attractionRepulsionRatio =
                        std::clamp(mode._attractionRepulsionRatio, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);

                } else if (std::holds_alternative<AutoCrawlingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<AutoCrawlingGenomeDesc>(muscle._mode);
                    mode._maxDistanceDeviation =
                        std::clamp(mode._maxDistanceDeviation, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);
                    mode._forwardBackwardRatio =
                        std::clamp(mode._forwardBackwardRatio, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);

                } else if (std::holds_alternative<ManualCrawlingGenomeDesc>(muscle._mode)) {
                    auto& mode = std::get<ManualCrawlingGenomeDesc>(muscle._mode);
                    mode._maxDistanceDeviation =
                        std::clamp(mode._maxDistanceDeviation, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);
                    mode._forwardBackwardRatio =
                        std::clamp(mode._forwardBackwardRatio, Const::MuscleModeRatio_Min, Const::MuscleModeRatio_Max);
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
                    freeCell._restrictToColors =
                        std::clamp(freeCell._restrictToColors, Const::RestrictToColors_Min, Const::RestrictToColors_Max);
                } else if (reconnectorMode == ReconnectorMode_Creature) {
                    auto& creature = std::get<ReconnectCreatureGenomeDesc>(reconnector._mode);
                    if (creature._minNumCells.has_value()) {
                        auto& value = creature._minNumCells.value();
                        value = std::clamp(value, Const::CreatureNumCells_Min, 100);
                    }
                    if (creature._maxNumCells.has_value()) {
                        auto& value = creature._maxNumCells.value();
                        value = std::clamp(value, Const::CreatureNumCells_Min, 100);
                    }
                    creature._restrictToColors =
                        std::clamp(creature._restrictToColors, Const::RestrictToColors_Min, Const::RestrictToColors_Max);
                    creature._restrictToLineage = std::clamp(creature._restrictToLineage, 0, LineageRestriction_Count - 1);
                }

            } else if (nodeType == CellType_Detonator) {
                auto& detonator = std::get<DetonatorGenomeDesc>(node._cellType);
                detonator._countdown =
                    std::max(detonator._countdown, Const::DetonatorCountdown_Min);

            } else if (nodeType == CellType_Digestor) {
                auto& digestor = std::get<DigestorGenomeDesc>(node._cellType);
                digestor._rawEnergyConductivity = std::clamp(
                    digestor._rawEnergyConductivity,
                    Const::DigestorRawEnergyConductivity_Min,
                    Const::DigestorRawEnergyConductivity_Max);
            } else if (nodeType == CellType_Memory) {
                auto& memory = std::get<MemoryGenomeDesc>(node._cellType);
                memory._channelBitMask =
                    std::clamp(memory._channelBitMask, Const::MemoryChannelBitMask_Min, Const::MemoryChannelBitMask_Max);
                auto memoryMode = memory.getMode();
                if (memoryMode == MemoryMode_SignalDelay) {
                    auto& signalDelay = std::get<SignalDelayGenomeDesc>(memory._mode);
                    signalDelay._delay = std::clamp(signalDelay._delay, Const::SignalDelay_Min, Const::SignalDelay_Max);
                } else if (memoryMode == MemoryMode_SignalRecorder) {
                    auto& signalRecorder = std::get<SignalRecorderGenomeDesc>(memory._mode);
                    signalRecorder._numWrittenSignalEntries =
                        std::clamp(signalRecorder._numWrittenSignalEntries, 0, static_cast<int>(memory._signalEntries.size()));
                } else if (memoryMode == MemoryMode_SignalStorage) {
                } else if (memoryMode == MemoryMode_SignalIntegrator) {
                    auto& signalIntegrator = std::get<SignalIntegratorGenomeDesc>(memory._mode);
                    signalIntegrator._newSignalWeight = std::clamp(
                        signalIntegrator._newSignalWeight,
                        Const::SignalIntegratorNewSignalWeight_Min,
                        Const::SignalIntegratorNewSignalWeight_Max);
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
                    sender._range = std::clamp(sender._range, Const::CommunicatorRange_Min, Const::CommunicatorRange_Max);
                    sender._maxTimesSent = std::clamp(sender._maxTimesSent, Const::CommunicatorMaxTimesSent_Min, 10);
                } else if (communicatorMode == CommunicatorMode_Receiver) {
                    auto& receiver = std::get<ReceiverGenomeDesc>(communicator._mode);
                    receiver._restrictToColors =
                        std::clamp(receiver._restrictToColors, Const::RestrictToColors_Min, Const::RestrictToColors_Max);
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
                if (constructor._autoTriggerInterval.has_value()) {
                    constructor._autoTriggerInterval = std::max(*constructor._autoTriggerInterval, Const::ConstructorConstructionActivationTime_Min);
                }
                constructor._constructionActivationTime = std::clamp(
                    constructor._constructionActivationTime,
                    Const::ConstructorConstructionActivationTime_Min,
                    Const::ConstructorConstructionActivationTime_Max);
                constructor._provideEnergy =
                    std::clamp(constructor._provideEnergy, static_cast<ProvideEnergy>(0), static_cast<ProvideEnergy>(ProvideEnergy_Count - 1));
                constructor._reservedEnergy = std::max(0.0f, constructor._reservedEnergy);
                constructor._numBranches = std::clamp(constructor._numBranches, 1, 6);
                constructor._numConcatenations = std::max(constructor._numConcatenations, 1);
                constructor._constructionAngle =
                    std::clamp(constructor._constructionAngle, Const::ConstructorConstructionAngle_Min, Const::ConstructorConstructionAngle_Max);
            }
        }
    }
}

void DescValidationService::validateAndCorrect(ExtendedObjectDesc& extendedObject)
{
    auto& object = extendedObject.object;
    object._stiffness = std::clamp(object._stiffness, 0.0f, 1.0f);
    object._color = std::clamp(object._color, 0, MAX_COLORS - 1);

    if (extendedObject.creature.has_value()) {
        auto& creature = extendedObject.creature.value();
        creature._lineageId = std::max(creature._lineageId, 0);
        creature._accumulatedMutations = std::max(creature._accumulatedMutations, 0.0f);
    }

    if (object.getObjectType() == ObjectType_Cell) {
        auto& cell = object.getCellRef();
        cell._usableEnergy = std::max(0.0f, cell._usableEnergy);
        cell._rawEnergy = std::max(0.0f, cell._rawEnergy);
        cell._age = std::max(0, cell._age);

        // Validate neural network
        for (auto& activationFunction : cell._neuralNetwork._activationFunctions) {
            activationFunction =
                std::clamp(activationFunction, static_cast<ActivationFunction>(0), static_cast<ActivationFunction>(ActivationFunction_Count - 1));
        }
        for (auto& bias : cell._neuralNetwork._biases) {
            bias = std::clamp(bias, -2.0f, 2.0f);
        }
        for (auto& weight : cell._neuralNetwork._connectionWeights) {
            weight = std::clamp(weight, -1.0f, 1.0f);
        }

        // Validate cell-specific attributes based on type
        auto cellType = cell.getCellType();

        if (cellType == CellType_Depot) {
            auto& depot = std::get<DepotDesc>(cell._cellType);
            depot._storageLimit = std::clamp(depot._storageLimit, 0.0f, 1000.0f);
            depot._storedUsableEnergy = std::clamp(depot._storedUsableEnergy, 0.0f, depot._storageLimit);

        } else if (cellType == CellType_Sensor) {
            auto& sensor = std::get<SensorDesc>(cell._cellType);
            sensor._minRange = std::clamp(sensor._minRange, 0, 255);
            sensor._maxRange = std::clamp(sensor._maxRange, 0, 255);

            // Validate mode-specific data
            auto mode = sensor.getMode();
            if (mode == SensorMode_DetectEnergy) {
                auto& detectEnergy = std::get<DetectEnergyDesc>(sensor._mode);
                detectEnergy._minDensity = std::max(detectEnergy._minDensity, 0.0f);
            } else if (mode == SensorMode_DetectFreeCell) {
                auto& detectFreeCell = std::get<DetectFreeCellDesc>(sensor._mode);
                detectFreeCell._minDensity = std::clamp(detectFreeCell._minDensity, 0.0f, 1.0f);
                detectFreeCell._restrictToColors &= (1 << MAX_COLORS) - 1;
            } else if (mode == SensorMode_DetectCreature) {
                auto& detectCreature = std::get<DetectCreatureDesc>(sensor._mode);
                if (detectCreature._minNumCells.has_value()) {
                    auto& value = detectCreature._minNumCells.value();
                    value = std::max(value, 0);
                }
                if (detectCreature._maxNumCells.has_value()) {
                    auto& value = detectCreature._maxNumCells.value();
                    value = std::max(value, 0);
                }
                detectCreature._restrictToColors &= (1 << MAX_COLORS) - 1;
                detectCreature._restrictToLineage = std::clamp(detectCreature._restrictToLineage, 0, LineageRestriction_Count - 1);
            }

        } else if (cellType == CellType_Generator) {
            auto& generator = std::get<GeneratorDesc>(cell._cellType);
            generator._minValue = std::clamp(generator._minValue, -2.0f, 2.0f);
            generator._maxValue = std::clamp(generator._maxValue, -2.0f, 2.0f);
            if (generator._minValue > generator._maxValue) {
                std::swap(generator._minValue, generator._maxValue);
            }
            generator._timeOffset = std::max(generator._timeOffset, 0);
            auto generatorMode = generator.getMode();
            if (generatorMode == GeneratorMode_SquareSignal) {
                auto& squareSignal = std::get<SquareSignalDesc>(generator._mode);
                squareSignal._period = std::max(squareSignal._period, 1);
            } else if (generatorMode == GeneratorMode_SawtoothSignal) {
                auto& sawtoothSignal = std::get<SawtoothSignalDesc>(generator._mode);
                sawtoothSignal._period = std::max(sawtoothSignal._period, 1);
            }

        } else if (cellType == CellType_Attacker) {
            auto& attacker = std::get<AttackerDesc>(cell._cellType);
            auto attackerMode = attacker.getMode();
            if (attackerMode == AttackerMode_FreeCell) {
                auto& freeCell = std::get<AttackFreeCellDesc>(attacker._mode);
                freeCell._restrictToColors &= (1 << MAX_COLORS) - 1;
            }

        } else if (cellType == CellType_Injector) {
            auto& injector = std::get<InjectorDesc>(cell._cellType);
            injector._geneIndex = std::max(injector._geneIndex, 0);

        } else if (cellType == CellType_Muscle) {
            auto& muscle = std::get<MuscleDesc>(cell._cellType);

            if (std::holds_alternative<AutoBendingDesc>(muscle._mode)) {
                auto& mode = std::get<AutoBendingDesc>(muscle._mode);
                mode._maxAngleDeviation = std::clamp(mode._maxAngleDeviation, 0.0f, 1.0f);
                mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);

            } else if (std::holds_alternative<ManualBendingDesc>(muscle._mode)) {
                auto& mode = std::get<ManualBendingDesc>(muscle._mode);
                mode._maxAngleDeviation = std::clamp(mode._maxAngleDeviation, 0.0f, 1.0f);
                mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);

            } else if (std::holds_alternative<AngleBendingDesc>(muscle._mode)) {
                auto& mode = std::get<AngleBendingDesc>(muscle._mode);
                mode._maxAngleDeviation = std::clamp(mode._maxAngleDeviation, 0.0f, 1.0f);
                mode._attractionRepulsionRatio = std::clamp(mode._attractionRepulsionRatio, 0.0f, 1.0f);

            } else if (std::holds_alternative<AutoCrawlingDesc>(muscle._mode)) {
                auto& mode = std::get<AutoCrawlingDesc>(muscle._mode);
                mode._maxDistanceDeviation = std::clamp(mode._maxDistanceDeviation, 0.0f, 1.0f);
                mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);

            } else if (std::holds_alternative<ManualCrawlingDesc>(muscle._mode)) {
                auto& mode = std::get<ManualCrawlingDesc>(muscle._mode);
                mode._maxDistanceDeviation = std::clamp(mode._maxDistanceDeviation, 0.0f, 1.0f);
                mode._forwardBackwardRatio = std::clamp(mode._forwardBackwardRatio, 0.0f, 1.0f);
            }

        } else if (cellType == CellType_Defender) {
            auto& defender = std::get<DefenderDesc>(cell._cellType);
            defender._mode = std::clamp(defender._mode, 0, DefenderMode_Count - 1);

        } else if (cellType == CellType_Reconnector) {
            auto& reconnector = std::get<ReconnectorDesc>(cell._cellType);
            auto reconnectorMode = reconnector.getMode();
            if (reconnectorMode == ReconnectorMode_FreeCell) {
                auto& freeCell = std::get<ReconnectFreeCellDesc>(reconnector._mode);
                freeCell._restrictToColors &= (1 << MAX_COLORS) - 1;
            } else if (reconnectorMode == ReconnectorMode_Creature) {
                auto& creature = std::get<ReconnectCreatureDesc>(reconnector._mode);
                if (creature._minNumCells.has_value()) {
                    auto& value = creature._minNumCells.value();
                    value = std::max(value, 0);
                }
                if (creature._maxNumCells.has_value()) {
                    auto& value = creature._maxNumCells.value();
                    value = std::max(value, 0);
                }
                creature._restrictToColors &= (1 << MAX_COLORS) - 1;
                creature._restrictToLineage = std::clamp(creature._restrictToLineage, 0, LineageRestriction_Count - 1);
            }

        } else if (cellType == CellType_Detonator) {
            auto& detonator = std::get<DetonatorDesc>(cell._cellType);
            detonator._countdown = std::max(detonator._countdown, 1);

        } else if (cellType == CellType_Memory) {
            auto& memory = std::get<MemoryDesc>(cell._cellType);
            auto memoryMode = memory.getMode();
            if (memoryMode == MemoryMode_SignalDelay) {
                auto& signalDelay = std::get<SignalDelayDesc>(memory._mode);
                signalDelay._delay = std::clamp(signalDelay._delay, 0, MAX_CELL_MEMORY_ENTRIES);
            } else if (memoryMode == MemoryMode_SignalRecorder) {
                auto& signalRecorder = std::get<SignalRecorderDesc>(memory._mode);
                signalRecorder._numWrittenSignalEntries =
                    std::clamp(signalRecorder._numWrittenSignalEntries, 0, static_cast<int>(memory._signalEntries.size()));
            } else if (memoryMode == MemoryMode_SignalStorage) {
            } else if (memoryMode == MemoryMode_SignalIntegrator) {
                auto& signalIntegrator = std::get<SignalIntegratorDesc>(memory._mode);
                signalIntegrator._newSignalWeight = std::clamp(signalIntegrator._newSignalWeight, 0.0f, 1.0f);
            }
            auto numEntries = memory._signalEntries.size();
            if (numEntries > MAX_CELL_MEMORY_ENTRIES) {
                memory._signalEntries.resize(MAX_CELL_MEMORY_ENTRIES);
            }

        } else if (cellType == CellType_Communicator) {
            auto& communicator = std::get<CommunicatorDesc>(cell._cellType);
            auto communicatorMode = communicator.getMode();
            if (communicatorMode == CommunicatorMode_Sender) {
                auto& sender = std::get<SenderDesc>(communicator._mode);
                sender._range = std::clamp(sender._range, 0, 20);
                sender._maxTimesSent = std::max(sender._maxTimesSent, 0);
            } else if (communicatorMode == CommunicatorMode_Receiver) {
                auto& receiver = std::get<ReceiverDesc>(communicator._mode);
                receiver._restrictToColors &= (1 << MAX_COLORS) - 1;
                receiver._restrictToLineage = std::clamp(receiver._restrictToLineage, 0, LineageRestriction_Count - 1);
            }
        }

        // Validate optional constructor field
        if (cell._constructor.has_value()) {
            auto& constructor = cell._constructor.value();
            if (constructor._autoTriggerInterval.has_value()) {
                auto& value = constructor._autoTriggerInterval.value();
                value = std::max(value, Const::ConstructorAutoTriggerInterval_Min);
            }
            constructor._geneIndex = std::max(constructor._geneIndex, 0);
            constructor._constructionActivationTime = std::clamp(
                constructor._constructionActivationTime, Const::ConstructorConstructionActivationTime_Min, Const::ConstructorConstructionActivationTime_Max);
            constructor._provideEnergy =
                std::clamp(constructor._provideEnergy, static_cast<ProvideEnergy>(0), static_cast<ProvideEnergy>(ProvideEnergy_Count - 1));
            constructor._reservedEnergy = std::max(0.0f, constructor._reservedEnergy);
            constructor._numBranches = std::clamp(constructor._numBranches, 1, 6);
            constructor._numConcatenations = std::max(constructor._numConcatenations, 1);
            constructor._constructionAngle =
                std::clamp(constructor._constructionAngle, Const::ConstructorConstructionAngle_Min, Const::ConstructorConstructionAngle_Max);
        }
    }
}
