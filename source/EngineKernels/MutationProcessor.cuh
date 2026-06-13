#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <cooperative_groups.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/NeuralNetWeight.h>
#include <EngineInterface/SimulationParameters.h>

#include "ConstructorHelper.cuh"
#include "EntityFactory.cuh"
#include "SimulationStatistics.cuh"

namespace cg_mutation = cooperative_groups;

class MutationProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);
    __inline__ __device__ static void applyMutations(SimulationData& data, Genome* genome);

private:
    __inline__ __device__ static void applyMutations_neurons(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static void applyMutations_connections(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static void applyMutations_cellTypeProperties(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static void applyMutations_cellTypeMode(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static void applyMutations_cellType(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static void applyMutations_void(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static void applyMutations_constructor(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static void applyMutations_meta(SimulationData& data, Genome* genome);

    __inline__ __device__ static void resetCellTypeModeToDefault(Node& node);
    __inline__ __device__ static void resetCellTypeToDefault(Node& node);

    __inline__ __device__ static void updateAccumulatedMutationsAndLineageId(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static float generateGaussian(SimulationData& data);
    __inline__ __device__ static bool isRandomEvent(SimulationData& data, float probability);

    __inline__ __device__ static bool hasMinimalEnergyForConstruction(Object* object);
    __inline__ __device__ static int getNumberOfNodes(Genome* genome);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void MutationProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    DEVICE_CHECK(blockDim.x == NEURONS_PER_CELL);

    auto block = cg_mutation::this_thread_block();
    auto laneId = block.thread_rank();

    auto& objects = data.entities.objects;
    auto partition = calcBlockPartition(objects.getNumEntries());

    EntityFactory factory;
    factory.init(&data);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& object = objects.at(index);

        __shared__ Genome* clonedGenome;

        if (laneId == 0) {
            clonedGenome = nullptr;
            if (object->type == ObjectType_Cell) {
                auto const& cell = object->typeData.cell;
                // Performance optimization:
                // Only mutate genome if it has a constructor for new offspring and a minimal amount of energy for construction
                // (=> prevents creation of genomes which are not used)
                if (cell.constructorAvailable && (cell.constructor.geneIndex == 0 || cell.constructor.separation) && cell.constructor.offspring == nullptr
                    && hasMinimalEnergyForConstruction(object)) {
                    auto& creature = object->typeData.cell.creature;
                    int origMutationState = atomicExch(&creature->mutationState, MutationState_Mutated);
                    if (origMutationState == MutationState_NotMutated) {
                        clonedGenome = factory.cloneGenome(creature->genome);
                    }
                }
            }
        }
        block.sync();

        if (clonedGenome != nullptr) {

            // Apply mutations to cloned genome
            applyMutations(data, clonedGenome);

            if (laneId == 0) {
                object->typeData.cell.creature->genome = clonedGenome;
            }
        }
        block.sync();
    }
}

__inline__ __device__ void MutationProcessor::applyMutations(SimulationData& data, Genome* genome)
{
    __shared__ float accumulatedMutations;
    auto block = cg_mutation::this_thread_block();
    auto laneId = block.thread_rank();

    if (laneId == 0) {
        accumulatedMutations = 0.0f;
    }
    block.sync();

    applyMutations_meta(data, genome);
    applyMutations_neurons(data, genome, accumulatedMutations);
    applyMutations_connections(data, genome, accumulatedMutations);
    applyMutations_cellTypeProperties(data, genome, accumulatedMutations);
    applyMutations_cellTypeMode(data, genome, accumulatedMutations);
    applyMutations_cellType(data, genome, accumulatedMutations);
    applyMutations_void(data, genome, accumulatedMutations);
    applyMutations_constructor(data, genome, accumulatedMutations);

    block.sync();
    updateAccumulatedMutationsAndLineageId(data, genome, accumulatedMutations);
}

__inline__ __device__ void MutationProcessor::applyMutations_neurons(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    NeuronMutation rates[2] = {genome->mutationRates.neuronMutations[0], genome->mutationRates.neuronMutations[1]};

    for (int rateIndex = 0; rateIndex < 2; ++rateIndex) {
        auto const& rate = rates[rateIndex];
        if (rate.eventProbability <= 0 || (rate.weightSigma <= 0 && rate.biasSigma <= 0 && rate.activationFunctionProbability <= 0)) {
            continue;
        }
        for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
                auto& node = gene.nodes[nodeIndex];
                if (laneId < NEURONS_PER_CELL) {
                    int neuronIndex = laneId;
                    if (data.primaryNumberGen.random() < rate.eventProbability) {
                        if (rate.weightSigma > 0) {
                            auto accumulatedWeightLocal = 0.0f;
                            for (int weightIndex = 0; weightIndex < NEURONS_PER_CELL; ++weightIndex) {
                                auto& weight = node.neuralNetwork.weights[neuronIndex * NEURONS_PER_CELL + weightIndex];
                                auto delta = generateGaussian(data) * rate.weightSigma;
                                float newValue = weight.getValue() + delta;
                                newValue = max(-2.0f, min(2.0f, newValue));
                                weight = NeuralNetWeight(newValue);
                                accumulatedWeightLocal += abs(delta);
                            }
                            atomicAdd_block(&accumulatedMutations, accumulatedWeightLocal / (NEURONS_PER_CELL * NEURONS_PER_CELL));
                        }
                        if (rate.biasSigma > 0) {
                            auto& bias = node.neuralNetwork.biases[neuronIndex];
                            auto delta = generateGaussian(data) * rate.biasSigma;
                            float newBias = bias + delta;
                            newBias = max(-2.0f, min(2.0f, newBias));
                            bias = newBias;
                            atomicAdd_block(&accumulatedMutations, abs(delta) / NEURONS_PER_CELL);
                        }
                        if (rate.activationFunctionProbability > 0 && data.primaryNumberGen.random() < rate.activationFunctionProbability) {
                            node.neuralNetwork.activationFunctions[neuronIndex] =
                                static_cast<ActivationFunction>(data.primaryNumberGen.random(ActivationFunction_Count - 1));
                            atomicAdd_block(&accumulatedMutations, 1.0f / NEURONS_PER_CELL);
                        }
                    }
                }
            }
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_connections(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    ConnectionMutation rates[2] = {genome->mutationRates.connectionMutations[0], genome->mutationRates.connectionMutations[1]};

    for (int rateIndex = 0; rateIndex < 2; ++rateIndex) {
        auto const& rate = rates[rateIndex];
        if (rate.eventProbability <= 0 || rate.sigma <= 0) {
            continue;
        }
        for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
                auto& node = gene.nodes[nodeIndex];
                if (laneId < MAX_OBJECT_CONNECTIONS) {
                    if (data.primaryNumberGen.random() < rate.eventProbability) {
                        auto& weight = node.neuralNetwork.connectionWeights[laneId];
                        auto delta = generateGaussian(data) * rate.sigma;
                        float newValue = weight + delta;
                        newValue = max(-2.0f, min(2.0f, newValue));
                        weight = newValue;
                        atomicAdd_block(&accumulatedMutations, abs(delta) / MAX_OBJECT_CONNECTIONS);
                    }
                }
            }
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_cellTypeProperties(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto block = cg_mutation::this_thread_block();
    auto laneId = block.thread_rank();
    CellTypePropertiesMutation rates[2] = {genome->mutationRates.cellTypePropertiesMutations[0], genome->mutationRates.cellTypePropertiesMutations[1]};

    for (int rateIndex = 0; rateIndex < 2; ++rateIndex) {
        auto const& rate = rates[rateIndex];
        if (rate.eventProbability <= 0 || (rate.sigma <= 0 && rate.probability <= 0)) {
            continue;
        }

        for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = laneId; nodeIndex < gene.numNodes; nodeIndex += blockDim.x) {
                if (data.primaryNumberGen.random() >= rate.eventProbability) {
                    continue;
                }
                auto& node = gene.nodes[nodeIndex];

                auto mutateNumber = [&](auto& value, auto minValue, auto maxValue) {
                    using ValueType = std::decay_t<decltype(value)>;
                    if (rate.sigma <= 0) {
                        return;
                    }
                    auto delta = generateGaussian(data) * rate.sigma * (maxValue - minValue);
                    if constexpr (std::is_integral_v<ValueType>) {
                        auto roundedDelta = static_cast<int>(std::round(delta));
                        auto newValue = max(static_cast<int>(minValue), min(static_cast<int>(maxValue), static_cast<int>(value) + roundedDelta));
                        value = static_cast<ValueType>(newValue);
                        atomicAdd_block(&accumulatedMutations, static_cast<float>(std::abs(roundedDelta)));
                    } else {
                        value = max(static_cast<ValueType>(minValue), min(static_cast<ValueType>(maxValue), value + delta));
                        atomicAdd_block(&accumulatedMutations, std::abs(delta));
                    }
                };

                auto mutateBoolField = [&](bool& value) {
                    if (data.primaryNumberGen.random() < rate.probability) {
                        value = !value;
                        atomicAdd_block(&accumulatedMutations, 1.0f);
                    }
                };

                auto mutateBitset = [&](auto& value, auto mask) {
                    using ValueType = std::decay_t<decltype(value)>;
                    using UnsignedType = std::make_unsigned_t<ValueType>;
                    auto newValue = static_cast<UnsignedType>(value);
                    auto const maskValue = static_cast<UnsignedType>(mask);
                    for (int bitIndex = 0; bitIndex < sizeof(UnsignedType) * 8; ++bitIndex) {
                        auto const bit = UnsignedType{1} << bitIndex;
                        if ((maskValue & bit) != 0 && data.primaryNumberGen.random() < rate.probability) {
                            newValue ^= bit;
                        }
                    }
                    if (newValue != static_cast<UnsignedType>(value)) {
                        value = static_cast<ValueType>(newValue);
                        atomicAdd_block(&accumulatedMutations, 1.0f);
                    }
                };

                auto mutateEnumField = [&](auto& value, int count) {
                    using ValueType = std::decay_t<decltype(value)>;
                    if (count > 1 && data.primaryNumberGen.random() < rate.probability) {
                        auto currentValue = static_cast<int>(value);
                        auto newValue = data.primaryNumberGen.random(count - 2);
                        if (newValue >= currentValue) {
                            ++newValue;
                        }
                        value = static_cast<ValueType>(newValue);
                        atomicAdd_block(&accumulatedMutations, 1.0f);
                    }
                };

                switch (node.cellType) {
                case CellType_Base:
                    break;
                case CellType_Depot:
                    mutateNumber(
                        node.cellTypeData.depot.storageLimit, Const::DepotStorageLimit_Min, Const::DepotStorageLimit_Max);
                    mutateNumber(
                        node.cellTypeData.depot.initialStoredUsableEnergy,
                        Const::DepotInitialStoredUsableEnergy_Min,
                        node.cellTypeData.depot.storageLimit);
                    break;
                case CellType_Sensor:
                    mutateBoolField(node.cellTypeData.sensor.autoTrigger);
                    mutateBoolField(node.cellTypeData.sensor.tagForAttackers);
                    mutateNumber(node.cellTypeData.sensor.minRange, Const::SensorRange_Min, Const::SensorRange_Max);
                    mutateNumber(node.cellTypeData.sensor.maxRange, Const::SensorRange_Min, Const::SensorRange_Max);
                    switch (node.cellTypeData.sensor.mode) {
                    case SensorMode_Telemetry:
                        break;
                    case SensorMode_DetectEnergy:
                        mutateNumber(
                            node.cellTypeData.sensor.modeData.detectEnergy.minDensity,
                            Const::DetectEnergyMinDensity_Min,
                            Const::DetectEnergyMinDensity_Max);
                        break;
                    case SensorMode_DetectSolid:
                        break;
                    case SensorMode_DetectFreeCell:
                        mutateNumber(
                            node.cellTypeData.sensor.modeData.detectFreeCell.minDensity,
                            Const::DetectFreeCellMinDensity_Min,
                            Const::DetectFreeCellMinDensity_Max);
                        mutateBitset(
                            node.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors,
                            Const::RestrictToColors_Max);
                        break;
                    case SensorMode_DetectCreature:
                        mutateNumber(node.cellTypeData.sensor.modeData.detectCreature.minNumCells, Const::CreatureNumCells_Min, 100);
                        mutateNumber(node.cellTypeData.sensor.modeData.detectCreature.maxNumCells, Const::CreatureNumCells_Min, 100);
                        mutateBitset(
                            node.cellTypeData.sensor.modeData.detectCreature.restrictToColors,
                            Const::RestrictToColors_Max);
                        mutateEnumField(node.cellTypeData.sensor.modeData.detectCreature.restrictToLineage, LineageRestriction_Count);
                        break;
                    }
                    break;
                case CellType_Generator:
                    mutateBoolField(node.cellTypeData.generator.additive);
                    mutateNumber(node.cellTypeData.generator.minValue, Const::GeneratorValue_Min, Const::GeneratorValue_Max);
                    mutateNumber(node.cellTypeData.generator.maxValue, Const::GeneratorValue_Min, Const::GeneratorValue_Max);
                    mutateNumber(node.cellTypeData.generator.timeOffset, Const::GeneratorTimeOffset_Min, 100);
                    switch (node.cellTypeData.generator.mode) {
                    case GeneratorMode_SquareSignal:
                        mutateNumber(node.cellTypeData.generator.modeData.squareSignal.period, Const::GeneratorPeriod_Min, 100);
                        break;
                    case GeneratorMode_SawtoothSignal:
                        mutateNumber(node.cellTypeData.generator.modeData.sawtoothSignal.period, Const::GeneratorPeriod_Min, 100);
                        break;
                    }
                    if (node.cellTypeData.generator.minValue > node.cellTypeData.generator.maxValue) {
                        auto const temp = node.cellTypeData.generator.minValue;
                        node.cellTypeData.generator.minValue = node.cellTypeData.generator.maxValue;
                        node.cellTypeData.generator.maxValue = temp;
                    }
                    break;
                case CellType_Attacker:
                    switch (node.cellTypeData.attacker.mode) {
                    case AttackerMode_FreeCell:
                        mutateBitset(
                            node.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors,
                            Const::RestrictToColors_Max);
                        break;
                    case AttackerMode_Creature:
                        break;
                    }
                    break;
                case CellType_Injector:
                    mutateNumber(node.cellTypeData.injector.geneIndex, 0, max(0, genome->numGenes - 1));
                    break;
                case CellType_Muscle:
                    switch (node.cellTypeData.muscle.mode) {
                    case MuscleMode_AutoBending:
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        break;
                    case MuscleMode_ManualBending:
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        break;
                    case MuscleMode_AngleBending:
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        break;
                    case MuscleMode_AutoCrawling:
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        break;
                    case MuscleMode_ManualCrawling:
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio,
                            Const::MuscleModeRatio_Min,
                            Const::MuscleModeRatio_Max);
                        break;
                    case MuscleMode_DirectMovement:
                        break;
                    }
                    break;
                case CellType_Defender:
                    break;
                case CellType_Reconnector:
                    switch (node.cellTypeData.reconnector.mode) {
                    case ReconnectorMode_Solid:
                        break;
                    case ReconnectorMode_FreeCell:
                        mutateBitset(
                            node.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors,
                            Const::RestrictToColors_Max);
                        break;
                    case ReconnectorMode_Creature:
                        mutateNumber(node.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells, Const::CreatureNumCells_Min, 100);
                        mutateNumber(node.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells, Const::CreatureNumCells_Min, 100);
                        mutateBitset(
                            node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors,
                            Const::RestrictToColors_Max);
                        mutateEnumField(node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage, LineageRestriction_Count);
                        break;
                    }
                    break;
                case CellType_Detonator:
                    mutateNumber(
                        node.cellTypeData.detonator.countdown, Const::DetonatorCountdown_Min, 100);
                    break;
                case CellType_Digestor:
                    mutateNumber(
                        node.cellTypeData.digestor.rawEnergyConductivity,
                        Const::DigestorRawEnergyConductivity_Min,
                        Const::DigestorRawEnergyConductivity_Max);
                    break;
                case CellType_Memory: {
                    mutateBitset(
                        node.cellTypeData.memory.channelBitMask,
                        Const::MemoryChannelBitMask_Max);

                    if (rate.sigma > 0) {
                        auto& memory = node.cellTypeData.memory;
                        auto oldNumSignalEntries = toInt(memory.numSignalEntries);
                        auto roundedDelta = toInt(std::round(generateGaussian(data) * rate.sigma));
                        auto newNumSignalEntries =
                            max(Const::MemoryNumSignalEntries_Min, min(Const::MemoryNumSignalEntries_Max, oldNumSignalEntries + roundedDelta));
                        if (newNumSignalEntries > oldNumSignalEntries) {
                            auto newSignalEntries = data.entities.heap.getTypedSubArray<SignalEntryGenome>(newNumSignalEntries);
                            for (int entryIndex = 0; entryIndex < newNumSignalEntries; ++entryIndex) {
                                for (int channelIndex = 0; channelIndex < NEURONS_PER_CELL; ++channelIndex) {
                                    newSignalEntries[entryIndex].channels[channelIndex] =
                                        entryIndex < oldNumSignalEntries ? memory.signalEntries[entryIndex].channels[channelIndex] : 0.0f;
                                }
                            }
                            memory.signalEntries = newSignalEntries;
                        }
                        memory.numSignalEntries = static_cast<uint8_t>(newNumSignalEntries);
                        atomicAdd_block(&accumulatedMutations, toFloat(std::abs(roundedDelta)));

                        if (newNumSignalEntries > 0) {
                            float signalMutations = 0.0f;
                            for (int entryIndex = 0; entryIndex < newNumSignalEntries; ++entryIndex) {
                                auto& entry = memory.signalEntries[entryIndex];
                                for (int channelIndex = 0; channelIndex < NEURONS_PER_CELL; ++channelIndex) {
                                    auto delta = generateGaussian(data) * rate.sigma;
                                    auto newValue = max(-2.0f, min(2.0f, entry.channels[channelIndex] + delta));
                                    signalMutations += abs(newValue - entry.channels[channelIndex]);
                                    entry.channels[channelIndex] = newValue;
                                }
                            }
                            atomicAdd_block(&accumulatedMutations, signalMutations / newNumSignalEntries);
                        }
                    }

                    switch (node.cellTypeData.memory.mode) {
                    case MemoryMode_SignalDelay:
                        mutateNumber(
                            node.cellTypeData.memory.modeData.signalDelay.delay,
                            Const::SignalDelay_Min,
                            Const::SignalDelay_Max);
                        break;
                    case MemoryMode_SignalRecorder:
                        mutateBoolField(node.cellTypeData.memory.modeData.signalRecorder.readOnly);
                        mutateNumber(
                            node.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries,
                            0,
                            static_cast<int>(node.cellTypeData.memory.numSignalEntries));
                        break;
                    case MemoryMode_SignalStorage:
                        mutateBoolField(node.cellTypeData.memory.modeData.signalStorage.readOnly);
                        break;
                    case MemoryMode_SignalIntegrator:
                        mutateNumber(
                            node.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight,
                            Const::SignalIntegratorNewSignalWeight_Min,
                            Const::SignalIntegratorNewSignalWeight_Max);
                        break;
                    }
                    break;
                }
                case CellType_Communicator:
                    switch (node.cellTypeData.communicator.mode) {
                    case CommunicatorMode_Sender:
                        mutateNumber(
                            node.cellTypeData.communicator.modeData.sender.range,
                            Const::CommunicatorRange_Min,
                            Const::CommunicatorRange_Max);
                        mutateNumber(node.cellTypeData.communicator.modeData.sender.maxTimesSent, Const::CommunicatorMaxTimesSent_Min, 10);
                        break;
                    case CommunicatorMode_Receiver:
                        mutateBitset(
                            node.cellTypeData.communicator.modeData.receiver.restrictToColors,
                            Const::RestrictToColors_Max);
                        mutateEnumField(node.cellTypeData.communicator.modeData.receiver.restrictToLineage, LineageRestriction_Count);
                        break;
                    }
                    break;
                case CellType_Void:
                    break;
                }
            }
        }
    }
}

__inline__ __device__ void MutationProcessor::resetCellTypeModeToDefault(Node& node)
{
    // Initializes the data of the currently selected mode with the shared default attribute values
    // (see the *_Default constants in CellTypeConstants.h and the host-side defaults in GenomeDesc.h).
    switch (node.cellType) {
    case CellType_Sensor: {
        auto& sensor = node.cellTypeData.sensor;
        switch (sensor.mode) {
        case SensorMode_Telemetry:
            sensor.modeData.telemetry = {};
            break;
        case SensorMode_DetectEnergy:
            sensor.modeData.detectEnergy = {Const::DetectEnergyMinDensity_Default};
            break;
        case SensorMode_DetectSolid:
            sensor.modeData.detectSolid = {};
            break;
        case SensorMode_DetectFreeCell:
            sensor.modeData.detectFreeCell = {Const::DetectFreeCellMinDensity_Default, Const::RestrictToColors_Default};
            break;
        case SensorMode_DetectCreature:
            sensor.modeData.detectCreature = {0, 0, Const::RestrictToColors_Default, LineageRestriction_No};
            break;
        }
    } break;
    case CellType_Generator: {
        auto& generator = node.cellTypeData.generator;
        switch (generator.mode) {
        case GeneratorMode_SquareSignal:
            generator.modeData.squareSignal = {Const::GeneratorPeriod_Default};
            break;
        case GeneratorMode_SawtoothSignal:
            generator.modeData.sawtoothSignal = {Const::GeneratorPeriod_Default};
            break;
        }
    } break;
    case CellType_Attacker: {
        auto& attacker = node.cellTypeData.attacker;
        switch (attacker.mode) {
        case AttackerMode_FreeCell:
            attacker.modeData.attackFreeCell = {Const::RestrictToColors_Default};
            break;
        case AttackerMode_Creature:
            attacker.modeData.attackCreature = {};
            break;
        }
    } break;
    case CellType_Muscle: {
        auto& muscle = node.cellTypeData.muscle;
        switch (muscle.mode) {
        case MuscleMode_AutoBending:
            muscle.modeData.autoBending = {Const::MuscleMaxAngleDeviation_Default, Const::MuscleForwardBackwardRatio_Default};
            break;
        case MuscleMode_ManualBending:
            muscle.modeData.manualBending = {Const::MuscleMaxAngleDeviation_Default, Const::MuscleForwardBackwardRatio_Default};
            break;
        case MuscleMode_AngleBending:
            muscle.modeData.angleBending = {Const::MuscleMaxAngleDeviation_Default, Const::MuscleAttractionRepulsionRatio_Default};
            break;
        case MuscleMode_AutoCrawling:
            muscle.modeData.autoCrawling = {Const::MuscleMaxDistanceDeviation_Default, Const::MuscleForwardBackwardRatio_Default};
            break;
        case MuscleMode_ManualCrawling:
            muscle.modeData.manualCrawling = {Const::MuscleMaxDistanceDeviation_Default, Const::MuscleForwardBackwardRatio_Default};
            break;
        case MuscleMode_DirectMovement:
            muscle.modeData.directMovement = {};
            break;
        }
    } break;
    case CellType_Reconnector: {
        auto& reconnector = node.cellTypeData.reconnector;
        switch (reconnector.mode) {
        case ReconnectorMode_Solid:
            reconnector.modeData.reconnectSolid = {};
            break;
        case ReconnectorMode_FreeCell:
            reconnector.modeData.reconnectFreeCell = {Const::RestrictToColors_Default};
            break;
        case ReconnectorMode_Creature:
            reconnector.modeData.reconnectCreature = {0, 0, Const::RestrictToColors_Default, LineageRestriction_No};
            break;
        }
    } break;
    case CellType_Memory: {
        auto& memory = node.cellTypeData.memory;
        switch (memory.mode) {
        case MemoryMode_SignalDelay:
            memory.modeData.signalDelay = {Const::SignalDelay_Default};
            break;
        case MemoryMode_SignalRecorder:
            memory.modeData.signalRecorder = {true, 0};
            break;
        case MemoryMode_SignalStorage:
            memory.modeData.signalStorage = {true};
            break;
        case MemoryMode_SignalIntegrator:
            memory.modeData.signalIntegrator = {Const::SignalIntegratorNewSignalWeight_Default};
            break;
        }
    } break;
    case CellType_Communicator: {
        auto& communicator = node.cellTypeData.communicator;
        switch (communicator.mode) {
        case CommunicatorMode_Sender:
            communicator.modeData.sender = {Const::CommunicatorRange_Default, Const::CommunicatorMaxTimesSent_Default};
            break;
        case CommunicatorMode_Receiver:
            communicator.modeData.receiver = {Const::RestrictToColors_Default, LineageRestriction_No};
            break;
        }
    } break;
    default:
        // Cell types without mode-specific data (or without a mode) need no reset.
        break;
    }
}

__inline__ __device__ void MutationProcessor::resetCellTypeToDefault(Node& node)
{
    // Initializes all attributes of the node's current cell type with their shared default values
    // (see the *_Default constants in CellTypeConstants.h and the host-side defaults in GenomeDesc.h).
    auto& cellTypeData = node.cellTypeData;
    switch (node.cellType) {
    case CellType_Base:
        cellTypeData.base = {};
        break;
    case CellType_Depot:
        cellTypeData.depot = {Const::DepotStorageLimit_Default, Const::DepotInitialStoredUsableEnergy_Default};
        break;
    case CellType_Sensor: {
        auto& sensor = cellTypeData.sensor;
        sensor.autoTrigger = Const::SensorAutoTrigger_Default;
        sensor.tagForAttackers = Const::SensorTagForAttackers_Default;
        sensor.mode = SensorMode_DetectCreature;
        sensor.minRange = Const::SensorMinRange_Default;
        sensor.maxRange = Const::SensorMaxRange_Default;
    } break;
    case CellType_Generator: {
        auto& generator = cellTypeData.generator;
        generator.additive = Const::GeneratorAdditive_Default;
        generator.minValue = Const::GeneratorMinValue_Default;
        generator.maxValue = Const::GeneratorMaxValue_Default;
        generator.timeOffset = Const::GeneratorTimeOffset_Default;
        generator.mode = GeneratorMode_SquareSignal;
    } break;
    case CellType_Attacker:
        cellTypeData.attacker.mode = AttackerMode_Creature;
        break;
    case CellType_Injector:
        cellTypeData.injector = {Const::InjectorGeneIndex_Default};
        break;
    case CellType_Muscle:
        cellTypeData.muscle.mode = MuscleMode_AutoBending;
        break;
    case CellType_Defender:
        cellTypeData.defender = {DefenderMode_DefendAgainstAttacker};
        break;
    case CellType_Reconnector:
        cellTypeData.reconnector.mode = ReconnectorMode_Creature;
        break;
    case CellType_Detonator:
        cellTypeData.detonator = {Const::DetonatorCountdown_Default};
        break;
    case CellType_Digestor:
        cellTypeData.digestor = {Const::DigestorRawEnergyConductivity_Default};
        break;
    case CellType_Memory: {
        auto& memory = cellTypeData.memory;
        memory.mode = MemoryMode_SignalDelay;
        memory.numSignalEntries = 0;
        memory.channelBitMask = Const::MemoryChannelBitMask_Max;
        memory.signalEntries = nullptr;
    } break;
    case CellType_Communicator:
        cellTypeData.communicator.mode = CommunicatorMode_Sender;
        break;
    case CellType_Void:
        cellTypeData.voidCell = {};
        break;
    }

    resetCellTypeModeToDefault(node);
}

__inline__ __device__ void MutationProcessor::applyMutations_cellTypeMode(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();
    auto const& rate = genome->mutationRates.cellTypeModeMutation;

    if (rate.eventProbability <= 0) {
        return;
    }

    auto pickNewMode = [&](int currentMode, int count) {
        auto newMode = data.primaryNumberGen.random(count - 2);
        if (newMode >= currentMode) {
            ++newMode;
        }
        return newMode;
    };

    for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
        auto& gene = genome->genes[geneIndex];
        for (int nodeIndex = laneId; nodeIndex < gene.numNodes; nodeIndex += blockDim.x) {
            if (data.primaryNumberGen.random() >= rate.eventProbability) {
                continue;
            }
            auto& node = gene.nodes[nodeIndex];

            // Pick a new mode for the node's cell type; the new mode's data is then reset to its defaults below.
            bool changed = true;
            switch (node.cellType) {
            case CellType_Sensor:
                node.cellTypeData.sensor.mode = static_cast<SensorMode>(pickNewMode(node.cellTypeData.sensor.mode, SensorMode_Count));
                break;
            case CellType_Generator:
                node.cellTypeData.generator.mode = static_cast<GeneratorMode>(pickNewMode(node.cellTypeData.generator.mode, GeneratorMode_Count));
                break;
            case CellType_Attacker:
                node.cellTypeData.attacker.mode = static_cast<AttackerMode>(pickNewMode(node.cellTypeData.attacker.mode, AttackerMode_Count));
                break;
            case CellType_Muscle:
                node.cellTypeData.muscle.mode = static_cast<MuscleMode>(pickNewMode(node.cellTypeData.muscle.mode, MuscleMode_Count));
                break;
            case CellType_Defender:
                node.cellTypeData.defender.mode = static_cast<DefenderMode>(pickNewMode(node.cellTypeData.defender.mode, DefenderMode_Count));
                break;
            case CellType_Reconnector:
                node.cellTypeData.reconnector.mode = static_cast<ReconnectorMode>(pickNewMode(node.cellTypeData.reconnector.mode, ReconnectorMode_Count));
                break;
            case CellType_Memory:
                node.cellTypeData.memory.mode = static_cast<MemoryMode>(pickNewMode(node.cellTypeData.memory.mode, MemoryMode_Count));
                break;
            case CellType_Communicator:
                node.cellTypeData.communicator.mode = static_cast<CommunicatorMode>(pickNewMode(node.cellTypeData.communicator.mode, CommunicatorMode_Count));
                break;
            default:
                // Cell types without a mode field are not affected.
                changed = false;
                break;
            }

            if (changed) {
                resetCellTypeModeToDefault(node);
                atomicAdd_block(&accumulatedMutations, 1.0f);
            }
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_cellType(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();
    auto const& rate = genome->mutationRates.cellTypeMutation;

    if (rate.eventProbability <= 0) {
        return;
    }

    // Pick a new non-void cell type that differs from the current one; void nodes keep their type
    auto pickNewCellType = [&](int currentCellType) {
        auto newCellType = data.primaryNumberGen.random(CellType_Count - 3);
        if (newCellType >= currentCellType) {
            ++newCellType;
        }
        if (newCellType >= CellType_Void) {
            ++newCellType;
        }
        return newCellType;
    };

    for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
        auto& gene = genome->genes[geneIndex];

        if (laneId == 0 && data.primaryNumberGen.random() < rate.eventProbability) {
            gene.homogeneCellType = !gene.homogeneCellType;
            atomicAdd_block(&accumulatedMutations, 1.0f);
        }

        for (int nodeIndex = laneId; nodeIndex < gene.numNodes; nodeIndex += blockDim.x) {
            auto& node = gene.nodes[nodeIndex];

            if (node.cellType == CellType_Void) {
                continue;
            }
            if (data.primaryNumberGen.random() >= rate.eventProbability) {
                continue;
            }

            node.cellType = pickNewCellType(node.cellType);
            resetCellTypeToDefault(node);
            atomicAdd_block(&accumulatedMutations, 1.0f);
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_void(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();
    auto const& rate = genome->mutationRates.voidMutation;

    if (rate.eventProbability <= 0) {
        return;
    }

    // Pick a non-void cell type used when a void node is turned into a non-void node.
    // Void is the last entry before CellType_Count, so the range [0, CellType_Count - 2] covers exactly the non-void types.
    auto pickNonVoidCellType = [&]() { return data.primaryNumberGen.random(CellType_Count - 2); };

    for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
        auto& gene = genome->genes[geneIndex];
        for (int nodeIndex = laneId; nodeIndex < gene.numNodes; nodeIndex += blockDim.x) {
            // The first and last node of a gene must not be void, so they are never mutated.
            if (nodeIndex == 0 || nodeIndex == gene.numNodes - 1) {
                continue;
            }
            if (data.primaryNumberGen.random() >= rate.eventProbability) {
                continue;
            }
            auto& node = gene.nodes[nodeIndex];

            // Toggle the node between void and non-void; the new cell type is reset to its default attribute values.
            node.cellType = node.cellType == CellType_Void ? pickNonVoidCellType() : CellType_Void;
            resetCellTypeToDefault(node);
            atomicAdd_block(&accumulatedMutations, 1.0f);
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_constructor(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto block = cg_mutation::this_thread_block();
    auto laneId = block.thread_rank();
    ConstructorMutation rates[2] = {genome->mutationRates.constructorMutations[0], genome->mutationRates.constructorMutations[1]};

    for (int rateIndex = 0; rateIndex < 2; ++rateIndex) {
        auto const& rate = rates[rateIndex];
        if (rate.eventProbability <= 0 || (rate.sigma <= 0 && rate.probability <= 0)) {
            continue;
        }

        for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = laneId; nodeIndex < gene.numNodes; nodeIndex += blockDim.x) {
                if (data.primaryNumberGen.random() >= rate.eventProbability) {
                    continue;
                }
                auto& node = gene.nodes[nodeIndex];
                auto& constructor = node.constructor;

                // Mutate real and integer attributes by a Gaussian step scaled by the value range; the clamping mirrors DescValidationService.
                auto mutateNumber = [&](auto& value, auto minValue, auto maxValue) {
                    using ValueType = std::decay_t<decltype(value)>;
                    if (rate.sigma <= 0) {
                        return;
                    }
                    auto delta = generateGaussian(data) * rate.sigma * (maxValue - minValue);
                    if constexpr (std::is_integral_v<ValueType>) {
                        auto roundedDelta = static_cast<int>(std::round(delta));
                        auto newValue = max(static_cast<int>(minValue), min(static_cast<int>(maxValue), static_cast<int>(value) + roundedDelta));
                        value = static_cast<ValueType>(newValue);
                        atomicAdd_block(&accumulatedMutations, static_cast<float>(std::abs(roundedDelta)));
                    } else {
                        value = max(static_cast<ValueType>(minValue), min(static_cast<ValueType>(maxValue), value + delta));
                        atomicAdd_block(&accumulatedMutations, std::abs(delta));
                    }
                };

                auto mutateBoolField = [&](bool& value) {
                    if (data.primaryNumberGen.random() < rate.probability) {
                        value = !value;
                        atomicAdd_block(&accumulatedMutations, 1.0f);
                    }
                };

                // Mutate the attributes of an existing constructor (real/integer via sigma, bool via probability).
                if (node.constructorAvailable) {
                    if (constructor.autoTriggerInterval > 0) {
                        // Only an already auto-triggering constructor is mutated, so it keeps auto triggering.
                        mutateNumber(constructor.autoTriggerInterval, Const::ConstructorAutoTriggerInterval_Min, Const::ConstructorAutoTriggerInterval_Min + 100);
                    }
                    mutateNumber(constructor.geneIndex, 0, max(0, genome->numGenes - 1));
                    mutateNumber(
                        constructor.constructionActivationTime,
                        Const::ConstructorConstructionActivationTime_Min,
                        Const::ConstructorConstructionActivationTime_Max);
                    mutateNumber(constructor.constructionAngle, Const::ConstructorConstructionAngle_Min, Const::ConstructorConstructionAngle_Max);
                    mutateNumber(constructor.reservedEnergy, 0.0f, 300.0f);
                    mutateNumber(constructor.numBranches, 1, 6);
                    if (!ConstructorHelper::hasInfiniteConcatenations(constructor)) {
                        mutateNumber(constructor.numConcatenations, 1, 100);
                    }
                    mutateBoolField(constructor.separation);
                }

                // Mutate whether the node has a constructor at all; enabling one initializes it with default values.
                bool wasAvailable = node.constructorAvailable;
                if (data.primaryNumberGen.random() < rate.probability) {
                    node.constructorAvailable = !node.constructorAvailable;
                    atomicAdd_block(&accumulatedMutations, 1.0f);
                    if (node.constructorAvailable && !wasAvailable) {
                        constructor = {};
                        constructor.autoTriggerInterval = Const::ConstructorAutoTriggerInterval_Default;
                        constructor.constructionActivationTime = Const::ConstructorConstructionActivationTime_Default;
                        constructor.numBranches = 1;
                        constructor.numConcatenations = 1;
                    }
                }
            }
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_meta(SimulationData& data, Genome* genome)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    if (laneId == 0) {
        // Meta-mutate neuron mutation rates
        float neuronSigma = cudaSimulationParameters.metaMutationNeuronsSigma.value;
        if (neuronSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * neuronSigma)); };
            for (int i = 0; i < 2; ++i) {
                mutateFloat(genome->mutationRates.neuronMutations[i].eventProbability);
                mutateFloat(genome->mutationRates.neuronMutations[i].weightSigma);
                mutateFloat(genome->mutationRates.neuronMutations[i].biasSigma);
                mutateFloat(genome->mutationRates.neuronMutations[i].activationFunctionProbability);
            }
        }

        // Meta-mutate connection mutation rates
        float connSigma = cudaSimulationParameters.metaMutationConnectionsSigma.value;
        if (connSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * connSigma)); };
            for (int i = 0; i < 2; ++i) {
                mutateFloat(genome->mutationRates.connectionMutations[i].eventProbability);
                mutateFloat(genome->mutationRates.connectionMutations[i].sigma);
            }
        }

        float cellTypeSigma = cudaSimulationParameters.metaMutationCellTypePropertiesSigma.value;
        if (cellTypeSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * cellTypeSigma)); };
            for (int i = 0; i < 2; ++i) {
                mutateFloat(genome->mutationRates.cellTypePropertiesMutations[i].eventProbability);
                mutateFloat(genome->mutationRates.cellTypePropertiesMutations[i].sigma);
                mutateFloat(genome->mutationRates.cellTypePropertiesMutations[i].probability);
            }
        }

        float cellTypeModeSigma = cudaSimulationParameters.metaMutationCellTypeModeSigma.value;
        if (cellTypeModeSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * cellTypeModeSigma)); };
            mutateFloat(genome->mutationRates.cellTypeModeMutation.eventProbability);
        }

        float cellTypeMutationSigma = cudaSimulationParameters.metaMutationCellTypeSigma.value;
        if (cellTypeMutationSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * cellTypeMutationSigma)); };
            mutateFloat(genome->mutationRates.cellTypeMutation.eventProbability);
        }

        float voidMutationSigma = cudaSimulationParameters.metaMutationVoidSigma.value;
        if (voidMutationSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * voidMutationSigma)); };
            mutateFloat(genome->mutationRates.voidMutation.eventProbability);
        }

        float constructorSigma = cudaSimulationParameters.metaMutationConstructorSigma.value;
        if (constructorSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * constructorSigma)); };
            for (int i = 0; i < 2; ++i) {
                mutateFloat(genome->mutationRates.constructorMutations[i].eventProbability);
                mutateFloat(genome->mutationRates.constructorMutations[i].sigma);
                mutateFloat(genome->mutationRates.constructorMutations[i].probability);
            }
        }
    }
}

__inline__ __device__ void MutationProcessor::updateAccumulatedMutationsAndLineageId(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();
    if (laneId == 0) {
        auto numberOfNodes = getNumberOfNodes(genome);
        auto denominator = numberOfNodes > 0 ? toFloat(numberOfNodes) : 1.0f;


        genome->accumulatedMutations += accumulatedMutations / denominator;
        if (genome->accumulatedMutations > cudaSimulationParameters.newLineageThreshold.value) {
            genome->prevLineageId = genome->lineageId;
            genome->lineageId = data.primaryNumberGen.createLineageId();
            genome->accumulatedMutations = 0;
        }
    }
}

__inline__ __device__ float MutationProcessor::generateGaussian(SimulationData& data)
{
    // Box-Muller transform for Gaussian random
    float u1 = max(1.0e-7f, data.primaryNumberGen.random());
    float u2 = data.primaryNumberGen.random();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * Const::PI * u2);
}

__inline__ __device__ bool MutationProcessor::isRandomEvent(SimulationData& data, float probability)
{
    if (probability > 0.001f) {
        return data.primaryNumberGen.random() < probability;
    } else {
        return data.primaryNumberGen.random() < probability * 1000 && data.secondaryNumberGen.random() < 0.001f;
    }
}

__inline__ __device__ bool MutationProcessor::hasMinimalEnergyForConstruction(Object* object)
{
    auto& cell = object->typeData.cell;
    auto& constructor = cell.constructor;
    if (constructor.provideEnergy == ProvideEnergy_Free) {
        return true;
    }

    auto normalCellEnergy = cudaSimulationParameters.normalCellEnergy.value[object->color];
    auto availableEnergyForConstruction = cell.usableEnergy + constructor.reservedEnergy - normalCellEnergy;

    return availableEnergyForConstruction >= normalCellEnergy - NEAR_ZERO;
}

__inline__ __device__ int MutationProcessor::getNumberOfNodes(Genome* genome)
{
    auto totalNodes = 0;
    for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
        totalNodes += genome->genes[geneIndex].numNodes;
    }
    return totalNodes;
}
