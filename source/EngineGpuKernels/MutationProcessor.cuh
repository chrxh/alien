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
    __inline__ __device__ static void mutateDeletion(SimulationData& data, Cell* cell);

private:
    static auto constexpr MAX_SUBGENOME_RECURSION_DEPTH = 20;

    __inline__ __device__ static int getRandomGenomeNodeIndex(
        SimulationData& data,
        uint8_t* genome,
        int genomeSize,
        bool considerZeroSubGenomes,
        int* subGenomesSizeIndices = nullptr,
        int* numSubGenomesSizeIndices = nullptr);
    __inline__ __device__ static void setRandomCellFunctionData(
        SimulationData& data,
        uint8_t* genomeData,
        int byteIndex,
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
    __inline__ __device__ static int getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int nodeIndex);
    __inline__ __device__ static int getNextCellFunctionType(uint8_t* genome, int nodeIndex);
    __inline__ __device__ static int getNextCellColor(uint8_t* genome, int nodeIndex);
    __inline__ __device__ static void setNextCellFunctionType(uint8_t* genome, int nodeIndex, CellFunction cellFunction);
    __inline__ __device__ static void setNextCellColor(uint8_t* genome, int nodeIndex, int color);
    __inline__ __device__
        static int getNextSubGenomeSize(uint8_t* genome, int genomeSize, int nodeIndex);  //prerequisites: (constructor or injector) and !makeSelfCopy
    __inline__ __device__
        static int getCellFunctionDataSize(
        CellFunction cellFunction,
        bool makeSelfCopy,
        int genomeSize);  //genomeSize only relevant for cellFunction = constructor or injector

    __inline__ __device__ static int readWord(uint8_t* genome, int byteIndex);
    __inline__ __device__ static void writeWord(uint8_t* genome, int byteIndex, int word);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void MutationProcessor::applyRandomMutation(SimulationData& data, Cell* cell)
{
    auto cellFunctionConstructorMutationNeuronProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationNeuronDataProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationNeuronDataProbability,
        data,
        cell->absPos);
    auto cellFunctionConstructorMutationDataProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationDataProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationDataProbability,
        data,
        cell->absPos);
    auto cellFunctionConstructorMutationCellFunctionProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationCellFunctionProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationCellFunctionProbability,
        data,
        cell->absPos);
    auto cellFunctionConstructorMutationInsertionProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationInsertionProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationInsertionProbability,
        data,
        cell->absPos);

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
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false);

    //basic property mutation
    if (data.numberGen1.randomBool()) {
        auto randomDelta = data.numberGen1.random(CellBasicBytes - 1);
        if (randomDelta == 0 || randomDelta == 5) { //no cell function type and color changes
            return;
        }
        genome[nodeIndex + randomDelta] = data.numberGen1.randomByte();
    }

    //cell function specific mutation
    else {
        auto nextCellFunctionGenomeBytes = getNextCellFunctionDataSize(genome, genomeSize, nodeIndex);
        auto type = getNextCellFunctionType(genome, nodeIndex);
        if (type == CellFunction_Constructor || type == CellFunction_Injector) {
            nextCellFunctionGenomeBytes = type == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
        }
        if (nextCellFunctionGenomeBytes > 0) {
            auto randomDelta = data.numberGen1.random(nextCellFunctionGenomeBytes - 1);
            genome[nodeIndex + CellBasicBytes + randomDelta] = data.numberGen1.randomByte();
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
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false);

    auto type = getNextCellFunctionType(genome, nodeIndex);
    if (type == CellFunction_Neuron) {
        auto delta = data.numberGen1.random(NeuronBytes - 1);
        genome[nodeIndex + CellBasicBytes + delta] = data.numberGen1.randomByte();
    }
}

__inline__ __device__ void MutationProcessor::mutateCellFunction(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize == 0) {
        return;
    }
    int subGenomesSizeIndices[MAX_SUBGENOME_RECURSION_DEPTH];
    int numSubGenomesSizeIndices;
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto newCellFunction = data.numberGen1.random(CellFunction_Count - 1);
    auto makeSelfCopy = false;
    //data.numberGen1.randomBool();  //only used if newCellFunction == Constructor or Injector

    auto newCellFunctionSize = getCellFunctionDataSize(newCellFunction, makeSelfCopy, 0);
    auto origCellFunctionSize = getNextCellFunctionDataSize(genome, genomeSize, nodeIndex);
    auto sizeDelta = newCellFunctionSize - origCellFunctionSize;

    auto targetGenomeSize = genomeSize + sizeDelta;
    if (targetGenomeSize > MAX_GENOME_BYTES) {
        return;
    }
    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);
    for (int i = 0; i < nodeIndex + CellBasicBytes; ++i) {
        targetGenome[i] = genome[i];
    }
    setNextCellFunctionType(targetGenome, nodeIndex, newCellFunction);
    setRandomCellFunctionData(data, targetGenome, nodeIndex + CellBasicBytes, newCellFunction, makeSelfCopy, 0);

    for (int i = nodeIndex + CellBasicBytes + origCellFunctionSize; i < genomeSize; ++i) {
        targetGenome[i + sizeDelta] = genome[i];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = readWord(genome, subGenomesSizeIndices[i]);
        writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize + sizeDelta);
    }
    if (constructor.currentGenomePos > nodeIndex) {
        constructor.currentGenomePos += sizeDelta;
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
}

__inline__ __device__ void MutationProcessor::mutateInsertion(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);

    int subGenomesSizeIndices[MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices;
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, true, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto newColor = cell->color;
    if (nodeIndex < genomeSize) {
        newColor = getNextCellColor(genome, nodeIndex);
    }
    auto newCellFunction = data.numberGen1.random(CellFunction_Count - 1);
    auto makeSelfCopy = false;

    auto newCellFunctionSize = getCellFunctionDataSize(newCellFunction, makeSelfCopy, 0);
    auto sizeDelta = newCellFunctionSize + CellBasicBytes;

    auto targetGenomeSize = genomeSize + sizeDelta;
    if (targetGenomeSize > MAX_GENOME_BYTES) {
        return;
    }
    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);
    for (int i = 0; i < nodeIndex; ++i) {
        targetGenome[i] = genome[i];
    }
    data.numberGen1.randomBytes(targetGenome + nodeIndex, CellBasicBytes);
    setNextCellFunctionType(targetGenome, nodeIndex, newCellFunction);
    setNextCellColor(targetGenome, nodeIndex, newColor);
    setRandomCellFunctionData(data, targetGenome, nodeIndex + CellBasicBytes, newCellFunction, makeSelfCopy, 0);

    for (int i = nodeIndex; i < genomeSize; ++i) {
        targetGenome[i + sizeDelta] = genome[i];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = readWord(genome, subGenomesSizeIndices[i]);
        writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize + sizeDelta);
    }
    if (constructor.currentGenomePos > nodeIndex || constructor.currentGenomePos == constructor.genomeSize) {
        constructor.currentGenomePos += sizeDelta;
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
}

__inline__ __device__ void MutationProcessor::mutateDeletion(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize == 0) {
        return;
    }

    int subGenomesSizeIndices[MAX_SUBGENOME_RECURSION_DEPTH];
    int numSubGenomesSizeIndices;
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto origCellFunctionSize = getNextCellFunctionDataSize(genome, genomeSize, nodeIndex);
    auto deleteSize = origCellFunctionSize + CellBasicBytes;

    auto targetGenomeSize = genomeSize - deleteSize;
    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);
    for (int i = 0; i < nodeIndex; ++i) {
        targetGenome[i] = genome[i];
    }
    data.numberGen1.randomBytes(targetGenome + nodeIndex, CellBasicBytes);

    for (int i = nodeIndex; i < genomeSize; ++i) {
        targetGenome[i] = genome[i + deleteSize];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = readWord(genome, subGenomesSizeIndices[i]);
        writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize - deleteSize);
    }
    if (constructor.currentGenomePos > nodeIndex) {
        constructor.currentGenomePos -= deleteSize;
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
}

__inline__ __device__ int MutationProcessor::getRandomGenomeNodeIndex(
    SimulationData& data,
    uint8_t* genome,
    int genomeSize,
    bool considerZeroSubGenomes,
    int* subGenomesSizeIndices,
    int* numSubGenomesSizeIndices)
{
    if (numSubGenomesSizeIndices) {
        *numSubGenomesSizeIndices = 0;
    }
    if (genomeSize == 0) {
        return 0;
    }
    auto randomRefIndex = data.numberGen1.random(genomeSize - 1);

    int result = 0;
    for (int depth = 0; depth < MAX_SUBGENOME_RECURSION_DEPTH; ++depth) {
        auto nodeIndex = findStartNodeIndex(genome, genomeSize, randomRefIndex);
        result += nodeIndex;
        auto cellFunction = getNextCellFunctionType(genome, nodeIndex);

        if (cellFunction == CellFunction_Constructor || cellFunction == CellFunction_Injector) {
            auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
            auto makeSelfCopy = GenomeDecoder::convertByteToBool(genome[nodeIndex + CellBasicBytes + cellFunctionFixedBytes]);
            if (makeSelfCopy) {
                break;
            } else {
                if (numSubGenomesSizeIndices) {
                    subGenomesSizeIndices[*numSubGenomesSizeIndices] = result + CellBasicBytes + cellFunctionFixedBytes + 1;
                    ++(*numSubGenomesSizeIndices);
                }
                auto subGenomeStartIndex = nodeIndex + CellBasicBytes + cellFunctionFixedBytes + 3;
                auto subGenomeSize = getNextSubGenomeSize(genome, genomeSize, nodeIndex);
                if (subGenomeSize == 0) {
                    if (considerZeroSubGenomes && data.numberGen1.randomBool()) {
                        result += CellBasicBytes + cellFunctionFixedBytes + 3;
                    } else {
                        if (numSubGenomesSizeIndices) {
                            --(*numSubGenomesSizeIndices);
                        }
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
    int byteIndex,
    CellFunction const& cellFunction,
    bool makeSelfCopy,
    int subGenomeSize)
{
    auto newCellFunctionSize = getCellFunctionDataSize(cellFunction, makeSelfCopy, subGenomeSize);
    data.numberGen1.randomBytes(genomeData + byteIndex, newCellFunctionSize);
    if (cellFunction == CellFunction_Constructor || cellFunction == CellFunction_Injector) {
        auto cellFunctionFixedBytes =
            cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
        genomeData[byteIndex + cellFunctionFixedBytes] = makeSelfCopy ? 1 : 0;
        if (!makeSelfCopy) {
            writeWord(genomeData, byteIndex + cellFunctionFixedBytes + 1, subGenomeSize);
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

__inline__ __device__ int MutationProcessor::getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int nodeIndex)
{
    auto cellFunction = getNextCellFunctionType(genome, nodeIndex);
    switch (cellFunction) {
    case CellFunction_Neuron:
        return NeuronBytes;
    case CellFunction_Transmitter:
        return TransmitterBytes;
    case CellFunction_Constructor: {
        auto isMakeCopy = GenomeDecoder::convertByteToBool(genome[nodeIndex + CellBasicBytes + ConstructorFixedBytes]);
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
        auto isMakeCopy = GenomeDecoder::convertByteToBool(genome[nodeIndex + CellBasicBytes + InjectorFixedBytes]);
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

__inline__ __device__ int MutationProcessor::getNextCellFunctionType(uint8_t* genome, int nodeIndex)
{
    return genome[nodeIndex] % CellFunction_Count;
}

__inline__ __device__ int MutationProcessor::getNextCellColor(uint8_t* genome, int nodeIndex)
{
    return genome[nodeIndex + 5] % MAX_COLORS;
}

__inline__ __device__ void MutationProcessor::setNextCellFunctionType(uint8_t* genome, int nodeIndex, CellFunction cellFunction)
{
    genome[nodeIndex] = static_cast<uint8_t>(cellFunction);
}

__inline__ __device__ void MutationProcessor::setNextCellColor(uint8_t* genome, int nodeIndex, int color)
{
    genome[nodeIndex + 5] = color;
}

__inline__ __device__ int MutationProcessor::getNextSubGenomeSize(uint8_t* genome, int genomeSize, int nodeIndex)
{
    auto cellFunction = getNextCellFunctionType(genome, nodeIndex);
    auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
    auto subGenomeSizeIndex = nodeIndex + CellBasicBytes + cellFunctionFixedBytes + 1;
    return max(
        min(GenomeDecoder::convertBytesToWord(genome[subGenomeSizeIndex], genome[subGenomeSizeIndex + 1]),
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

__inline__ __device__ int MutationProcessor::readWord(uint8_t* genome, int byteIndex)
{
    return GenomeDecoder::convertBytesToWord(genome[byteIndex], genome[byteIndex + 1]);
}

__inline__ __device__ void MutationProcessor::writeWord(uint8_t* genome, int byteIndex, int word)
{
    GenomeDecoder::convertWordToBytes(word, genome[byteIndex], genome[byteIndex + 1]);
}
