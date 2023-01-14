#pragma once

#include "EngineInterface/CellFunctionEnums.h"

#include "CellConnectionProcessor.cuh"
#include "GenomeDecoder.cuh"

class MutationProcessor
{
public:
    __inline__ __device__ static void applyRandomMutation(SimulationData& data, Cell* cell);

    __inline__ __device__ static void mutateData(SimulationData& data, Cell* cell);
    __inline__ __device__ static void mutateNeuronData(SimulationData& data, Cell* cell);
    __inline__ __device__ static void mutateCellFunction(SimulationData& data, Cell* cell);
    __inline__ __device__ static void mutateInsertion(SimulationData& data, Cell* cell);

private:
    __inline__ __device__ static int getRandomGenomeNodeIndex(SimulationData& data, uint8_t* genome, int genomeSize);
    __inline__ __device__ static void setRandomCellFunctionData(
        SimulationData& data,
        uint8_t* genomeData,
        uint64_t targetGenomeSize,
        int offset,
        CellFunction const& cellFunction,
        bool makeSelfCopy,
        int subGenomeSize);

    
    //internal constants
    static int constexpr CellFunctionMutationMaxGenomeSize = 200;
    static int constexpr CellBasicBytes = 8;
    static int constexpr NeuronBytes = 72;
    static int constexpr TransmitterBytes = 1;
    static int constexpr ConstructorFixedBytes = 8;
    static int constexpr SensorBytes = 4;
    static int constexpr NerveBytes = 2;
    static int constexpr AttackerBytes = 1;
    static int constexpr InjectorFixedBytes = 0;
    static int constexpr MuscleBytes = 1;
    __inline__ __device__ static int getNumGenomeCells(uint8_t* genome, int genomeSize);
    __inline__ __device__ static int findStartNodeIndex(uint8_t* genome, int genomeSize, int refIndex);
    __inline__ __device__ static int convertCellIndexToNodeIndex(uint8_t* genome, int genomeSize, int cellIndex);
    __inline__ __device__ static int convertNodeIndexToCellIndex(uint8_t* genome, int genomeSize, int nodeIndex);
    __inline__ __device__ static int getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int nodeIndex);
    __inline__ __device__ static int getNextCellFunctionType(uint8_t* genome, int genomeSize, int nodeIndex);
    __inline__ __device__ static int getNextSubGenomeSize(uint8_t* genome, int genomeSize, int nodeIndex);  //prerequisites: (constructor or injector) and !makeSelfCopy
    __inline__ __device__
        static int getCellFunctionDataSize(
        CellFunction cellFunction,
        bool makeSelfCopy,
        int genomeSize);  //genomeSize only relevant for cellFunction = constructor or injector

    __inline__ __device__ static void writeByte(uint8_t* genome, int genomeSize, int byteIndex, uint8_t byte);
    __inline__ __device__ static uint8_t readByte(uint8_t* genome, int genomeSize, int byteIndex);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void MutationProcessor::applyRandomMutation(SimulationData& data, Cell* cell)
{
    auto cellFunctionConstructorMutationNeuronProbability =
        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionConstructorMutationNeuronDataProbability, data, cell->absPos);
    auto cellFunctionConstructorMutationDataProbability =
        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionConstructorMutationDataProbability, data, cell->absPos);
    auto cellFunctionConstructorMutationCellFunctionProbability =
        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionConstructorMutationCellFunctionProbability, data, cell->absPos);
    auto cellFunctionConstructorMutationInsertionProbability =
        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionConstructorMutationInsertionProbability, data, cell->absPos);

    if (data.numberGen1.random() < cellFunctionConstructorMutationNeuronProbability) {
        mutateNeuronData(data, cell);
    }
    if (data.numberGen1.random() < cellFunctionConstructorMutationDataProbability) {
        mutateData(data, cell);
    }
    if (data.numberGen1.random() < cellFunctionConstructorMutationCellFunctionProbability) {
        mutateCellFunction(data, cell);
    }
    if (data.numberGen1.random() < cellFunctionConstructorMutationInsertionProbability) {
        mutateInsertion(data, cell);
    }
}

__inline__ __device__ void MutationProcessor::mutateData(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize == 0) {
        return;
    }
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize);

    //basic property mutation
    if (data.numberGen1.randomBool()) {
        auto randomDelta = data.numberGen1.random(CellBasicBytes - 1);
        if (randomDelta == 0 || randomDelta == 5) { //no cell function type and color changes
            return;
        }
        writeByte(genome, genomeSize, nodeIndex + randomDelta, data.numberGen1.randomByte());
    }

    //cell function specific mutation
    else {
        auto nextCellFunctionGenomeBytes = getNextCellFunctionDataSize(genome, genomeSize, nodeIndex);
        auto type = getNextCellFunctionType(genome, genomeSize, nodeIndex);
        if (type == CellFunction_Constructor || type == CellFunction_Injector) {
            nextCellFunctionGenomeBytes = type == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
        }
        if (nextCellFunctionGenomeBytes > 0) {
            auto randomDelta = data.numberGen1.random(nextCellFunctionGenomeBytes - 1);
            writeByte(genome, genomeSize, nodeIndex + CellBasicBytes + randomDelta, data.numberGen1.randomByte());
        }
    }
}

__inline__ __device__ void MutationProcessor::mutateNeuronData(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize == 0) {
        return;
    }
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize);

    auto type = getNextCellFunctionType(genome, genomeSize, nodeIndex);
    if (type == CellFunction_Neuron) {
        auto delta = data.numberGen1.random(NeuronBytes - 1);
        writeByte(genome, genomeSize, nodeIndex + CellBasicBytes + delta, data.numberGen1.randomByte());
    }
}

__inline__ __device__ void MutationProcessor::mutateCellFunction(SimulationData& data, Cell* cell)
{
    //auto& constructor = cell->cellFunctionData.constructor;
    //auto numGenomeCells = getNumGenomeCells(constructor);
    //if (numGenomeCells == 0) {
    //    return;
    //}
    //auto origCurrentCellIndex = convertNodeIndexToCellIndex(constructor, constructor.currentGenomePos);

    //auto mutationCellIndex = data.numberGen1.random(numGenomeCells - 1);
    //auto mutationByteIndex = convertCellIndexToNodeIndex(constructor, mutationCellIndex);

    //auto origCellFunction = getNextCellFunctionType(constructor, mutationByteIndex);
    //if (origCellFunction == CellFunction_Constructor || origCellFunction == CellFunction_Injector) {
    //    return;
    //}
    //auto newCellFunction = data.numberGen1.random(CellFunction_Count - 1);
    //auto makeSelfCopy = false;
    //auto subGenomeSize = 0;
    //if (newCellFunction == CellFunction_Constructor || newCellFunction == CellFunction_Injector) {
    //    return;
    //    //subGenomeSize = data.numberGen1.random(CellFunctionMutationMaxGenomeSize);
    //    //makeSelfCopy = data.numberGen1.randomBool();
    //}
    //auto newCellFunctionSize = getCellFunctionDataSize(newCellFunction, makeSelfCopy, subGenomeSize);
    //auto oldCellFunctionSize = getNextCellFunctionDataSize(constructor, mutationByteIndex);

    //auto targetGenomeSize =
    //    max(min(toInt(constructor.genomeSize) + newCellFunctionSize - oldCellFunctionSize, MAX_GENOME_BYTES),
    //        mutationByteIndex + CellBasicBytes + newCellFunctionSize);
    //auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);

    //for (int i = 0; i < mutationByteIndex + CellBasicBytes; ++i) {
    //    targetGenome[i % targetGenomeSize] = constructor.genome[i % constructor.genomeSize];
    //}
    //targetGenome[mutationByteIndex] = newCellFunction;
    //setRandomCellFunctionData(data, targetGenome, targetGenomeSize, mutationByteIndex + CellBasicBytes, newCellFunction, makeSelfCopy, subGenomeSize);

    //for (int i = mutationByteIndex + CellBasicBytes + oldCellFunctionSize; i < constructor.genomeSize; ++i) {
    //    targetGenome[(i + newCellFunctionSize - oldCellFunctionSize) % targetGenomeSize] = constructor.genome[i];
    //}
    //constructor.genomeSize = targetGenomeSize;
    //constructor.genome = targetGenome;
    //constructor.currentGenomePos = origCurrentCellIndex == numGenomeCells ? targetGenomeSize : convertCellIndexToNodeIndex(constructor, origCurrentCellIndex);
}

__inline__ __device__ void MutationProcessor::mutateInsertion(SimulationData& data, Cell* cell)
{
    //auto& constructor = cell->cellFunctionData.constructor;
    //auto numGenomeCells = getNumGenomeCells(constructor);
    //if (numGenomeCells == 0) {
    //    return;
    //}
    //auto origCurrentCellIndex = convertNodeIndexToCellIndex(constructor, constructor.currentGenomePos);

    //auto mutationCellIndex = data.numberGen1.random(numGenomeCells - 1);
    //auto mutationByteIndex = convertCellIndexToNodeIndex(constructor, mutationCellIndex);

    //auto newCellFunction = data.numberGen1.random(CellFunction_Count - 1);
    //auto makeSelfCopy = false;
    //auto subGenomeSize = 0;
    //if (newCellFunction == CellFunction_Constructor || newCellFunction == CellFunction_Injector) {
    //    subGenomeSize = data.numberGen1.random(CellFunctionMutationMaxGenomeSize);
    //    makeSelfCopy = data.numberGen1.randomBool();
    //}
    //auto newCellFunctionSize = getCellFunctionDataSize(newCellFunction, makeSelfCopy, subGenomeSize);

    //auto targetGenomeSize = min(toInt(constructor.genomeSize) + CellBasicBytes + newCellFunctionSize, MAX_GENOME_BYTES);
    //auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);

    //for (int i = 0; i < mutationByteIndex; ++i) {
    //    targetGenome[i % targetGenomeSize] = constructor.genome[i % constructor.genomeSize];
    //}
    //targetGenome[mutationByteIndex] = newCellFunction;
    //data.numberGen1.randomBytes(targetGenome + mutationByteIndex + 1, CellBasicBytes - 1);
    //setRandomCellFunctionData(data, targetGenome, targetGenomeSize, mutationByteIndex + CellBasicBytes, newCellFunction, makeSelfCopy, subGenomeSize);

    //for (int i = mutationByteIndex; i < constructor.genomeSize; ++i) {
    //    targetGenome[(i + CellBasicBytes +  newCellFunctionSize) % targetGenomeSize] = constructor.genome[i];
    //}
    //constructor.genomeSize = targetGenomeSize;
    //constructor.genome = targetGenome;
    //constructor.currentGenomePos = convertCellIndexToNodeIndex(constructor, origCurrentCellIndex);
}

__inline__ __device__ int MutationProcessor::getRandomGenomeNodeIndex(SimulationData& data, uint8_t* genome, int genomeSize)
{
    if (genomeSize == 0) {
        return 0;
    }
    auto randomRefIndex = data.numberGen1.random(genomeSize - 1);

    auto origGenomeSize = genomeSize;
    int result = 0;
    for (int depth = 0; depth < 20; ++depth) {
        auto nodeIndex = findStartNodeIndex(genome, genomeSize, randomRefIndex);
        result += nodeIndex;
        auto cellFunction = getNextCellFunctionType(genome, genomeSize, nodeIndex);

        if (cellFunction == CellFunction_Constructor || cellFunction == CellFunction_Injector) {
            auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
            auto makeSelfCopy = GenomeDecoder::convertByteToBool(readByte(genome, genomeSize, nodeIndex + CellBasicBytes + cellFunctionFixedBytes));
            if (makeSelfCopy) {
                break;
            } else {
                auto subGenomeStartIndex = nodeIndex + CellBasicBytes + cellFunctionFixedBytes + 3;
                auto subGenomeSize = getNextSubGenomeSize(genome, genomeSize, nodeIndex);
                if (subGenomeSize == 0) {
                    if (data.numberGen1.randomBool()) {
                        auto delta = CellBasicBytes + cellFunctionFixedBytes + 3;
                        if (result + delta >= origGenomeSize) {
                            break;
                        }
                        result += delta;
                    }
                    break;
                }
                genomeSize = subGenomeSize;
                genome = genome + subGenomeStartIndex;
                randomRefIndex -= subGenomeStartIndex;
                result += CellBasicBytes + cellFunctionFixedBytes + 3;
            }
        } else {
            break;
        }
    }
    return result;
}

__inline__ __device__ void MutationProcessor::setRandomCellFunctionData(
    SimulationData& data,
    uint8_t* genomeData,
    uint64_t targetGenomeSize,
    int offset,
    CellFunction const& cellFunction,
    bool makeSelfCopy,
    int subGenomeSize)
{
    auto newCellFunctionSize = getCellFunctionDataSize(cellFunction, makeSelfCopy, subGenomeSize);
    data.numberGen1.randomBytes(genomeData + offset, newCellFunctionSize);
    if (cellFunction == CellFunction_Constructor || cellFunction == CellFunction_Injector) {
        auto nodeIndex =
            cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
        writeByte(genomeData, targetGenomeSize, nodeIndex + offset, makeSelfCopy ? 1 : 0);
        if (!makeSelfCopy) {
            writeByte(genomeData, targetGenomeSize, nodeIndex + offset + 1, static_cast<uint8_t>(subGenomeSize & 0xff));
            writeByte(genomeData, targetGenomeSize, nodeIndex + offset + 2, static_cast<uint8_t>((subGenomeSize >> 8) & 0xff));
        }
    }
}

__inline__ __device__ int MutationProcessor::getNumGenomeCells(uint8_t* genome, int genomeSize)
{
    int result = 0;
    int currentNodeIndex = 0;
    for (; result < genomeSize && currentNodeIndex < genomeSize; ++result) {
        currentNodeIndex += CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeIndex);
    }

    return result;
}

__inline__ __device__ int MutationProcessor::findStartNodeIndex(uint8_t* genome, int genomeSize, int refIndex)
{
    int currentNodeIndex = 0;
    for (; currentNodeIndex <= refIndex;) {
        auto prevCurrentNodeIndex = currentNodeIndex;
        currentNodeIndex += CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeIndex);
        if (currentNodeIndex > refIndex) {
            return prevCurrentNodeIndex;
        }
    }
    return 0;
}

__inline__ __device__ int MutationProcessor::convertCellIndexToNodeIndex(uint8_t* genome, int genomeSize, int cellIndex)
{
    int currentNodeIndex = 0;
    for (int currentCellIndex = 0; currentCellIndex < cellIndex; ++currentCellIndex) {
        if (currentNodeIndex >= genomeSize) {
            break;
        }
        currentNodeIndex += CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeIndex);
    }

    return currentNodeIndex % genomeSize;
}

__inline__ __device__ int MutationProcessor::convertNodeIndexToCellIndex(uint8_t* genome, int genomeSize, int nodeIndex)
{
    int currentNodeIndex = 0;
    for (int currentCellIndex = 0; currentNodeIndex <= nodeIndex; ++currentCellIndex) {
        if (currentNodeIndex == nodeIndex) {
            return currentCellIndex;
        }
        currentNodeIndex += CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeIndex);
    }
    return 0;
}

__inline__ __device__ int MutationProcessor::getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int nodeIndex)
{
    auto cellFunction = getNextCellFunctionType(genome, genomeSize, nodeIndex);
    switch (cellFunction) {
    case CellFunction_Neuron:
        return NeuronBytes;
    case CellFunction_Transmitter:
        return TransmitterBytes;
    case CellFunction_Constructor: {
        auto isMakeCopy = GenomeDecoder::convertByteToBool(readByte(genome, genomeSize, nodeIndex + CellBasicBytes + ConstructorFixedBytes));
        if (isMakeCopy) {
            return ConstructorFixedBytes + 1;
        } else {
            return ConstructorFixedBytes + 3 + getNextSubGenomeSize(genome, genomeSize, nodeIndex);
        }
    }
    case CellFunction_Sensor:
        return SensorBytes;
    case CellFunction_Nerve:
        return NerveBytes;
    case CellFunction_Attacker:
        return AttackerBytes;
    case CellFunction_Injector: {
        auto isMakeCopy = GenomeDecoder::convertByteToBool(readByte(genome, genomeSize, nodeIndex + CellBasicBytes + InjectorFixedBytes));
        if (isMakeCopy) {
            return InjectorFixedBytes + 1;
        } else {
            return InjectorFixedBytes + 3 + getNextSubGenomeSize(genome, genomeSize, nodeIndex);
        }
    }
    case CellFunction_Muscle:
        return MuscleBytes;
    default:
        return 0;
    }
}

__inline__ __device__ int MutationProcessor::getNextCellFunctionType(uint8_t* genome, int genomeSize, int nodeIndex)
{
    return readByte(genome, genomeSize, nodeIndex) % CellFunction_Count;
}

__inline__ __device__ int MutationProcessor::getNextSubGenomeSize(uint8_t* genome, int genomeSize, int nodeIndex)
{
    auto cellFunction = getNextCellFunctionType(genome, genomeSize, nodeIndex);
    auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
    auto subGenomeSizeIndex = nodeIndex + CellBasicBytes + cellFunctionFixedBytes + 1;
    return max(
        min(GenomeDecoder::convertBytesToWord(readByte(genome, genomeSize, subGenomeSizeIndex), readByte(genome, genomeSize, subGenomeSizeIndex + 1)),
            genomeSize - (subGenomeSizeIndex + 2)),
        0);
}

__inline__ __device__ int MutationProcessor::getCellFunctionDataSize(CellFunction cellFunction, bool makeSelfCopy, int genomeSize)
{
    switch (cellFunction) {
    case CellFunction_Neuron:
        return NeuronBytes;
    case CellFunction_Transmitter:
        return TransmitterBytes;
    case CellFunction_Constructor: {
        return makeSelfCopy ? ConstructorFixedBytes + 1 : ConstructorFixedBytes + 3 + genomeSize;
    }
    case CellFunction_Sensor:
        return SensorBytes;
    case CellFunction_Nerve:
        return NerveBytes;
    case CellFunction_Attacker:
        return AttackerBytes;
    case CellFunction_Injector: {
        return makeSelfCopy ? InjectorFixedBytes + 1 : InjectorFixedBytes + 3 + genomeSize;
    }
    case CellFunction_Muscle:
        return MuscleBytes;
    default:
        return 0;
    }
}

__inline__ __device__ void MutationProcessor::writeByte(uint8_t* genome, int genomeSize, int byteIndex, uint8_t byte)
{
    genome[byteIndex % genomeSize] = byte;
}

__inline__ __device__ uint8_t MutationProcessor::readByte(uint8_t* genome, int genomeSize, int byteIndex)
{
    return genome[byteIndex % genomeSize];
}
