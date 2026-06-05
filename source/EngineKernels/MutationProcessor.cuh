#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <cooperative_groups.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/NeuralNetWeight.h>
#include <EngineInterface/SimulationParameters.h>

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
    __inline__ __device__ static void applyMutations_meta(SimulationData& data, Genome* genome);
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

    block.sync();
    updateAccumulatedMutationsAndLineageId(data, genome, accumulatedMutations);
}

__inline__ __device__ void MutationProcessor::applyMutations_neurons(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    NeuronMutation rates[2] = {genome->mutationRates.neuronMutation1, genome->mutationRates.neuronMutation2};

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
                            for (int weightIndex = 0; weightIndex < NEURONS_PER_CELL; ++weightIndex) {
                                auto& weight = node.neuralNetwork.weights[neuronIndex * NEURONS_PER_CELL + weightIndex];
                                auto delta = generateGaussian(data) * rate.weightSigma;
                                float newValue = weight.getValue() + delta;
                                newValue = max(-2.0f, min(2.0f, newValue));
                                weight = NeuralNetWeight(newValue);
                                atomicAdd_block(&accumulatedMutations, abs(delta));
                            }
                        }
                        if (rate.biasSigma > 0) {
                            auto& bias = node.neuralNetwork.biases[neuronIndex];
                            auto delta = generateGaussian(data) * rate.biasSigma;
                            float newBias = bias + delta;
                            newBias = max(-2.0f, min(2.0f, newBias));
                            bias = newBias;
                            atomicAdd_block(&accumulatedMutations, abs(delta));
                        }
                        if (rate.activationFunctionProbability > 0 && data.primaryNumberGen.random() < rate.activationFunctionProbability) {
                            node.neuralNetwork.activationFunctions[neuronIndex] =
                                static_cast<ActivationFunction>(data.primaryNumberGen.random(ActivationFunction_Count - 1));
                            atomicAdd_block(&accumulatedMutations, 1.0f);
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

    ConnectionMutation rates[2] = {genome->mutationRates.connectionMutation1, genome->mutationRates.connectionMutation2};

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
                        atomicAdd_block(&accumulatedMutations, abs(delta));
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
    CellTypePropertiesMutation rates[2] = {genome->mutationRates.cellTypePropertiesMutation1, genome->mutationRates.cellTypePropertiesMutation2};

    for (int rateIndex = 0; rateIndex < 2; ++rateIndex) {
        auto const& rate = rates[rateIndex];
        if (rate.eventProbability <= 0 || (rate.sigma <= 0 && rate.probability <= 0)) {
            continue;
        }

        auto nodeOrdinal = 0;
        for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex, ++nodeOrdinal) {
                if (nodeOrdinal % blockDim.x != laneId || data.primaryNumberGen.random() >= rate.eventProbability) {
                    continue;
                }
                auto& node = gene.nodes[nodeIndex];

                auto mutateNumber = [&](auto& value, auto minValue, auto maxValue) {
                    using ValueType = std::decay_t<decltype(value)>;
                    if (rate.sigma <= 0) {
                        return;
                    }
                    auto delta = generateGaussian(data) * rate.sigma;
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
                    if (rate.probability > 0 && data.primaryNumberGen.random() < rate.probability) {
                        value = !value;
                        atomicAdd_block(&accumulatedMutations, 1.0f);
                    }
                };

                auto mutateEnumField = [&](auto& value, int count) {
                    using ValueType = std::decay_t<decltype(value)>;
                    if (rate.probability > 0 && count > 1 && data.primaryNumberGen.random() < rate.probability) {
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
                        node.cellTypeData.depot.storageLimit, CellTypePropertyLimits::DepotStorageLimit_Min, CellTypePropertyLimits::DepotStorageLimit_Max);
                    mutateNumber(
                        node.cellTypeData.depot.initialStoredUsableEnergy,
                        CellTypePropertyLimits::DepotInitialStoredUsableEnergy_Min,
                        node.cellTypeData.depot.storageLimit);
                    break;
                case CellType_Sensor:
                    mutateBoolField(node.cellTypeData.sensor.autoTrigger);
                    mutateBoolField(node.cellTypeData.sensor.tagForAttackers);
                    mutateNumber(node.cellTypeData.sensor.minRange, CellTypePropertyLimits::SensorRange_Min, CellTypePropertyLimits::SensorRange_Max);
                    mutateNumber(node.cellTypeData.sensor.maxRange, CellTypePropertyLimits::SensorRange_Min, CellTypePropertyLimits::SensorRange_Max);
                    switch (node.cellTypeData.sensor.mode) {
                    case SensorMode_Telemetry:
                        break;
                    case SensorMode_DetectEnergy:
                        mutateNumber(
                            node.cellTypeData.sensor.modeData.detectEnergy.minDensity,
                            CellTypePropertyLimits::DetectEnergyMinDensity_Min,
                            CellTypePropertyLimits::DetectEnergyMinDensity_Max);
                        break;
                    case SensorMode_DetectSolid:
                        break;
                    case SensorMode_DetectFreeCell:
                        mutateNumber(
                            node.cellTypeData.sensor.modeData.detectFreeCell.minDensity,
                            CellTypePropertyLimits::DetectFreeCellMinDensity_Min,
                            CellTypePropertyLimits::DetectFreeCellMinDensity_Max);
                        mutateNumber(
                            node.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors,
                            CellTypePropertyLimits::RestrictToColors_Min,
                            CellTypePropertyLimits::RestrictToColors_Max);
                        break;
                    case SensorMode_DetectCreature:
                        mutateNumber(node.cellTypeData.sensor.modeData.detectCreature.minNumCells, CellTypePropertyLimits::CreatureNumCells_Min, 100);
                        mutateNumber(node.cellTypeData.sensor.modeData.detectCreature.maxNumCells, CellTypePropertyLimits::CreatureNumCells_Min, 100);
                        mutateNumber(
                            node.cellTypeData.sensor.modeData.detectCreature.restrictToColors,
                            CellTypePropertyLimits::RestrictToColors_Min,
                            CellTypePropertyLimits::RestrictToColors_Max);
                        mutateEnumField(node.cellTypeData.sensor.modeData.detectCreature.restrictToLineage, LineageRestriction_Count);
                        break;
                    }
                    break;
                case CellType_Generator:
                    mutateBoolField(node.cellTypeData.generator.additive);
                    mutateNumber(node.cellTypeData.generator.minValue, CellTypePropertyLimits::GeneratorValue_Min, CellTypePropertyLimits::GeneratorValue_Max);
                    mutateNumber(node.cellTypeData.generator.maxValue, CellTypePropertyLimits::GeneratorValue_Min, CellTypePropertyLimits::GeneratorValue_Max);
                    mutateNumber(node.cellTypeData.generator.timeOffset, CellTypePropertyLimits::GeneratorTimeOffset_Min, 100);
                    switch (node.cellTypeData.generator.mode) {
                    case GeneratorMode_SquareSignal:
                        mutateNumber(node.cellTypeData.generator.modeData.squareSignal.period, CellTypePropertyLimits::GeneratorPeriod_Min, 100);
                        break;
                    case GeneratorMode_SawtoothSignal:
                        mutateNumber(node.cellTypeData.generator.modeData.sawtoothSignal.period, CellTypePropertyLimits::GeneratorPeriod_Min, 100);
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
                        mutateNumber(
                            node.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors,
                            CellTypePropertyLimits::RestrictToColors_Min,
                            CellTypePropertyLimits::RestrictToColors_Max);
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
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio,
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
                        break;
                    case MuscleMode_ManualBending:
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation,
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio,
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
                        break;
                    case MuscleMode_AngleBending:
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation,
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio,
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
                        break;
                    case MuscleMode_AutoCrawling:
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation,
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio,
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
                        break;
                    case MuscleMode_ManualCrawling:
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation,
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
                        mutateNumber(
                            node.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio,
                            CellTypePropertyLimits::MuscleModeRatio_Min,
                            CellTypePropertyLimits::MuscleModeRatio_Max);
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
                        mutateNumber(
                            node.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors,
                            CellTypePropertyLimits::RestrictToColors_Min,
                            CellTypePropertyLimits::RestrictToColors_Max);
                        break;
                    case ReconnectorMode_Creature:
                        mutateNumber(node.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells, CellTypePropertyLimits::CreatureNumCells_Min, 100);
                        mutateNumber(node.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells, CellTypePropertyLimits::CreatureNumCells_Min, 100);
                        mutateNumber(
                            node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors,
                            CellTypePropertyLimits::RestrictToColors_Min,
                            CellTypePropertyLimits::RestrictToColors_Max);
                        mutateEnumField(node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage, LineageRestriction_Count);
                        break;
                    }
                    break;
                case CellType_Detonator:
                    mutateNumber(
                        node.cellTypeData.detonator.countdown, CellTypePropertyLimits::DetonatorCountdown_Min, CellTypePropertyLimits::DetonatorCountdown_Max);
                    break;
                case CellType_Digestor:
                    mutateNumber(
                        node.cellTypeData.digestor.rawEnergyConductivity,
                        CellTypePropertyLimits::DigestorRawEnergyConductivity_Min,
                        CellTypePropertyLimits::DigestorRawEnergyConductivity_Max);
                    break;
                case CellType_Memory:
                    mutateNumber(
                        node.cellTypeData.memory.channelBitMask,
                        CellTypePropertyLimits::MemoryChannelBitMask_Min,
                        CellTypePropertyLimits::MemoryChannelBitMask_Max);
                    switch (node.cellTypeData.memory.mode) {
                    case MemoryMode_SignalDelay:
                        mutateNumber(
                            node.cellTypeData.memory.modeData.signalDelay.delay,
                            CellTypePropertyLimits::SignalDelay_Min,
                            CellTypePropertyLimits::SignalDelay_Max);
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
                            CellTypePropertyLimits::SignalIntegratorNewSignalWeight_Min,
                            CellTypePropertyLimits::SignalIntegratorNewSignalWeight_Max);
                        break;
                    }
                    break;
                case CellType_Communicator:
                    switch (node.cellTypeData.communicator.mode) {
                    case CommunicatorMode_Sender:
                        mutateNumber(
                            node.cellTypeData.communicator.modeData.sender.range,
                            CellTypePropertyLimits::CommunicatorRange_Min,
                            CellTypePropertyLimits::CommunicatorRange_Max);
                        mutateNumber(node.cellTypeData.communicator.modeData.sender.maxTimesSent, CellTypePropertyLimits::CommunicatorMaxTimesSent_Min, 10);
                        break;
                    case CommunicatorMode_Receiver:
                        mutateNumber(
                            node.cellTypeData.communicator.modeData.receiver.restrictToColors,
                            CellTypePropertyLimits::RestrictToColors_Min,
                            CellTypePropertyLimits::RestrictToColors_Max);
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

__inline__ __device__ void MutationProcessor::applyMutations_meta(SimulationData& data, Genome* genome)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    if (laneId == 0) {
        // Meta-mutate neuron mutation rates
        float neuronSigma = cudaSimulationParameters.metaMutationNeuronsSigma.value;
        if (neuronSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * neuronSigma)); };
            mutateFloat(genome->mutationRates.neuronMutation1.eventProbability);
            mutateFloat(genome->mutationRates.neuronMutation1.weightSigma);
            mutateFloat(genome->mutationRates.neuronMutation1.biasSigma);
            mutateFloat(genome->mutationRates.neuronMutation1.activationFunctionProbability);
            mutateFloat(genome->mutationRates.neuronMutation2.eventProbability);
            mutateFloat(genome->mutationRates.neuronMutation2.weightSigma);
            mutateFloat(genome->mutationRates.neuronMutation2.biasSigma);
            mutateFloat(genome->mutationRates.neuronMutation2.activationFunctionProbability);
        }

        // Meta-mutate connection mutation rates
        float connSigma = cudaSimulationParameters.metaMutationConnectionsSigma.value;
        if (connSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * connSigma)); };
            mutateFloat(genome->mutationRates.connectionMutation1.eventProbability);
            mutateFloat(genome->mutationRates.connectionMutation1.sigma);
            mutateFloat(genome->mutationRates.connectionMutation2.eventProbability);
            mutateFloat(genome->mutationRates.connectionMutation2.sigma);
        }

        float cellTypeSigma = cudaSimulationParameters.metaMutationCellTypePropertiesSigma.value;
        if (cellTypeSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * cellTypeSigma)); };
            mutateFloat(genome->mutationRates.cellTypePropertiesMutation1.eventProbability);
            mutateFloat(genome->mutationRates.cellTypePropertiesMutation1.sigma);
            mutateFloat(genome->mutationRates.cellTypePropertiesMutation1.probability);
            mutateFloat(genome->mutationRates.cellTypePropertiesMutation2.eventProbability);
            mutateFloat(genome->mutationRates.cellTypePropertiesMutation2.sigma);
            mutateFloat(genome->mutationRates.cellTypePropertiesMutation2.probability);
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
