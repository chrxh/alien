#pragma once

#include <algorithm>
#include <cmath>
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
    __inline__ __device__ static void applyMutations_cellTypeProperties(
        SimulationData& data, Genome* genome, float& accumulatedMutations, int& cellTypePropertiesCount);
    __inline__ __device__ static void applyMutations_meta(SimulationData& data, Genome* genome);
    __inline__ __device__ static void updateAccumulatedMutationsAndLineageId(
        SimulationData& data, Genome* genome, float& accumulatedMutations, int cellTypePropertiesCount);
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
    __shared__ int cellTypePropertiesCount;
    auto block = cg_mutation::this_thread_block();
    auto laneId = block.thread_rank();

    if (laneId == 0) {
        accumulatedMutations = 0.0f;
        cellTypePropertiesCount = 0;
    }
    block.sync();

    applyMutations_meta(data, genome);
    applyMutations_neurons(data, genome, accumulatedMutations);
    applyMutations_connections(data, genome, accumulatedMutations);
    applyMutations_cellTypeProperties(data, genome, accumulatedMutations, cellTypePropertiesCount);

    block.sync();
    updateAccumulatedMutationsAndLineageId(data, genome, accumulatedMutations, cellTypePropertiesCount);
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

__inline__ __device__ void
MutationProcessor::applyMutations_connections(SimulationData& data, Genome* genome, float& accumulatedMutations)
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

__inline__ __device__ void MutationProcessor::applyMutations_cellTypeProperties(
    SimulationData& data, Genome* genome, float& accumulatedMutations, int& cellTypePropertiesCount)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();
    if (laneId != 0) {
        return;
    }

    auto const& rate = genome->mutationRates.cellTypePropertiesMutation;
    auto mutateFloat = [&](float& value, float minValue, float maxValue) {
        if (rate.eventProbability > 0 && data.primaryNumberGen.random() < rate.eventProbability && rate.sigma > 0) {
            auto delta = generateGaussian(data) * rate.sigma;
            value = value + delta;
            if (value < minValue) {
                value = minValue;
            } else if (value > maxValue) {
                value = maxValue;
            }
            accumulatedMutations += std::abs(delta);
        }
        ++cellTypePropertiesCount;
    };

    auto mutateIntegral = [&](auto& value, long long minValue, long long maxValue) {
        using ValueType = std::decay_t<decltype(value)>;
        if (rate.eventProbability > 0 && data.primaryNumberGen.random() < rate.eventProbability && rate.sigma > 0) {
            auto currentValue = static_cast<long long>(value);
            auto delta = static_cast<long long>(std::llround(generateGaussian(data) * rate.sigma));
            currentValue += delta;
            if (currentValue < minValue) {
                currentValue = minValue;
            } else if (currentValue > maxValue) {
                currentValue = maxValue;
            }
            value = static_cast<ValueType>(currentValue);
            accumulatedMutations += static_cast<float>(std::abs(static_cast<double>(delta)));
        }
        ++cellTypePropertiesCount;
    };

    auto mutateBoolField = [&](bool& value) {
        if (rate.eventProbability > 0 && data.primaryNumberGen.random() < rate.eventProbability && data.primaryNumberGen.random() < rate.probability) {
            value = !value;
            accumulatedMutations += 1.0f;
        }
        ++cellTypePropertiesCount;
    };

    auto mutateEnumField = [&](auto& value, int count) {
        using ValueType = std::decay_t<decltype(value)>;
        if (rate.eventProbability > 0 && count > 1 && data.primaryNumberGen.random() < rate.eventProbability
            && data.primaryNumberGen.random() < rate.probability) {
            auto currentValue = static_cast<int>(value);
            auto newValue = data.primaryNumberGen.random(count - 2);
            if (newValue >= currentValue) {
                ++newValue;
            }
            value = static_cast<ValueType>(newValue);
            accumulatedMutations += 1.0f;
        }
        ++cellTypePropertiesCount;
    };

    for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
        auto& gene = genome->genes[geneIndex];
        for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
            auto& node = gene.nodes[nodeIndex];

            switch (node.cellType) {
            case CellType_Base:
                break;
            case CellType_Depot:
                mutateFloat(node.cellTypeData.depot.storageLimit, 0.0f, 1000.0f);
                mutateFloat(node.cellTypeData.depot.initialStoredUsableEnergy, 0.0f, node.cellTypeData.depot.storageLimit);
                break;
            case CellType_Sensor:
                mutateBoolField(node.cellTypeData.sensor.autoTrigger);
                mutateBoolField(node.cellTypeData.sensor.tagForAttackers);
                mutateIntegral(node.cellTypeData.sensor.minRange, 0, 512);
                mutateIntegral(node.cellTypeData.sensor.maxRange, 0, 512);
                switch (node.cellTypeData.sensor.mode) {
                case SensorMode_Telemetry:
                    break;
                case SensorMode_DetectEnergy:
                    mutateFloat(node.cellTypeData.sensor.modeData.detectEnergy.minDensity, 0.0f, 1.0e30f);
                    break;
                case SensorMode_DetectSolid:
                    break;
                case SensorMode_DetectFreeCell:
                    mutateFloat(node.cellTypeData.sensor.modeData.detectFreeCell.minDensity, 0.0f, 1.0f);
                    mutateIntegral(node.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors, 0, (1 << MAX_COLORS) - 1);
                    break;
                case SensorMode_DetectCreature:
                    mutateIntegral(node.cellTypeData.sensor.modeData.detectCreature.minNumCells, 0, INT_MAX);
                    mutateIntegral(node.cellTypeData.sensor.modeData.detectCreature.maxNumCells, 0, INT_MAX);
                    mutateIntegral(node.cellTypeData.sensor.modeData.detectCreature.restrictToColors, 0, (1 << MAX_COLORS) - 1);
                    mutateEnumField(node.cellTypeData.sensor.modeData.detectCreature.restrictToLineage, LineageRestriction_Count);
                    break;
                }
                break;
            case CellType_Generator:
                mutateBoolField(node.cellTypeData.generator.additive);
                mutateFloat(node.cellTypeData.generator.minValue, -2.0f, 2.0f);
                mutateFloat(node.cellTypeData.generator.maxValue, -2.0f, 2.0f);
                mutateIntegral(node.cellTypeData.generator.timeOffset, 0, INT_MAX);
                switch (node.cellTypeData.generator.mode) {
                case GeneratorMode_SquareSignal:
                    mutateIntegral(node.cellTypeData.generator.modeData.squareSignal.period, 1, INT_MAX);
                    break;
                case GeneratorMode_SawtoothSignal:
                    mutateIntegral(node.cellTypeData.generator.modeData.sawtoothSignal.period, 1, INT_MAX);
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
                    mutateIntegral(node.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors, 0, (1 << MAX_COLORS) - 1);
                    break;
                case AttackerMode_Creature:
                    break;
                }
                break;
            case CellType_Injector:
                mutateIntegral(node.cellTypeData.injector.geneIndex, 0, INT_MAX);
                break;
            case CellType_Muscle:
                switch (node.cellTypeData.muscle.mode) {
                case MuscleMode_AutoBending:
                    mutateFloat(node.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation, 0.0f, 1.0f);
                    mutateFloat(node.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio, 0.0f, 1.0f);
                    break;
                case MuscleMode_ManualBending:
                    mutateFloat(node.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation, 0.0f, 1.0f);
                    mutateFloat(node.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio, 0.0f, 1.0f);
                    break;
                case MuscleMode_AngleBending:
                    mutateFloat(node.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation, 0.0f, 1.0f);
                    mutateFloat(node.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio, 0.0f, 1.0f);
                    break;
                case MuscleMode_AutoCrawling:
                    mutateFloat(node.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation, 0.0f, 1.0f);
                    mutateFloat(node.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio, 0.0f, 1.0f);
                    break;
                case MuscleMode_ManualCrawling:
                    mutateFloat(node.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation, 0.0f, 1.0f);
                    mutateFloat(node.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio, 0.0f, 1.0f);
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
                    mutateIntegral(node.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors, 0, (1 << MAX_COLORS) - 1);
                    break;
                case ReconnectorMode_Creature:
                    mutateIntegral(node.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells, 0, INT_MAX);
                    mutateIntegral(node.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells, 0, INT_MAX);
                    mutateIntegral(node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors, 0, (1 << MAX_COLORS) - 1);
                    mutateEnumField(node.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage, LineageRestriction_Count);
                    break;
                }
                break;
            case CellType_Detonator:
                mutateIntegral(node.cellTypeData.detonator.countdown, 1, INT_MAX);
                break;
            case CellType_Digestor:
                mutateFloat(node.cellTypeData.digestor.rawEnergyConductivity, 0.0f, 1.0f);
                break;
            case CellType_Memory:
                mutateIntegral(node.cellTypeData.memory.channelBitMask, 0, 0xffff);
                switch (node.cellTypeData.memory.mode) {
                case MemoryMode_SignalDelay:
                    mutateIntegral(node.cellTypeData.memory.modeData.signalDelay.delay, 0, MAX_CELL_MEMORY_ENTRIES);
                    break;
                case MemoryMode_SignalRecorder:
                    mutateBoolField(node.cellTypeData.memory.modeData.signalRecorder.readOnly);
                    mutateIntegral(
                        node.cellTypeData.memory.modeData.signalRecorder.numWrittenSignalEntries,
                        0,
                        static_cast<int>(node.cellTypeData.memory.numSignalEntries));
                    break;
                case MemoryMode_SignalStorage:
                    mutateBoolField(node.cellTypeData.memory.modeData.signalStorage.readOnly);
                    break;
                case MemoryMode_SignalIntegrator:
                    mutateFloat(node.cellTypeData.memory.modeData.signalIntegrator.newSignalWeight, 0.0f, 1.0f);
                    break;
                }
                break;
            case CellType_Communicator:
                switch (node.cellTypeData.communicator.mode) {
                case CommunicatorMode_Sender:
                    mutateIntegral(node.cellTypeData.communicator.modeData.sender.range, 0, 20);
                    mutateIntegral(node.cellTypeData.communicator.modeData.sender.maxTimesSent, 0, INT_MAX);
                    break;
                case CommunicatorMode_Receiver:
                    mutateIntegral(node.cellTypeData.communicator.modeData.receiver.restrictToColors, 0, (1 << MAX_COLORS) - 1);
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
            mutateFloat(genome->mutationRates.cellTypePropertiesMutation.eventProbability);
            mutateFloat(genome->mutationRates.cellTypePropertiesMutation.sigma);
            mutateFloat(genome->mutationRates.cellTypePropertiesMutation.probability);
        }
    }
}

__inline__ __device__ void
MutationProcessor::updateAccumulatedMutationsAndLineageId(SimulationData& data, Genome* genome, float& accumulatedMutations, int cellTypePropertiesCount)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();
    if (laneId == 0) {
        auto numberOfNodes = getNumberOfNodes(genome);
        auto denominator = numberOfNodes > 0 ? toFloat(numberOfNodes) : 1.0f;

        // Normalization
        denominator *= NEURONS_PER_CELL * NEURONS_PER_CELL  // For weight mutation
            + NEURONS_PER_CELL                              // For bias mutation
            + NEURONS_PER_CELL                              // For ActFn mutation
            + MAX_OBJECT_CONNECTIONS                        // For connection mutation
            ;
        denominator *= 2;                                   // 2 mutations for each type
        denominator += cellTypePropertiesCount;

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
