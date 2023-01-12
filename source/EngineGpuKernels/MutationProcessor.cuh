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

private:
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
    __inline__ __device__ static int getNumGenomeCells(ConstructorFunction const& constructor);
    __inline__ __device__ static int convertCellIndexToByteIndex(ConstructorFunction const& constructor, int cellIndex);
    __inline__ __device__ static int convertByteIndexToCellIndex(ConstructorFunction const& constructor, int byteIndex);
    __inline__ __device__ static int getNextCellFunctionGenomeBytes(ConstructorFunction const& constructor, int byteIndex);
    __inline__ __device__ static int getNextCellFunctionType(ConstructorFunction const& constructor, int byteIndex);
    __inline__ __device__ static bool getNextCellFunctionMakeSelfCopy(ConstructorFunction const& constructor, int byteIndex);
    __inline__ __device__ static int getCellFunctionDataSize(
        CellFunction cellFunction,
        bool makeSelfCopy,
        int genomeSize);  //genomeSize only relevant for cellFunction = constructor or injector
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

    if (data.numberGen1.random() < cellFunctionConstructorMutationNeuronProbability) {
        mutateNeuronData(data, cell);
    }
    if (data.numberGen1.random() < cellFunctionConstructorMutationDataProbability) {
        mutateData(data, cell);
    }
    if (data.numberGen1.random() < cellFunctionConstructorMutationCellFunctionProbability) {
        mutateCellFunction(data, cell);
    }
    //insert mutation

    //delete mutation

    //if (data.numberGen1.random() < 0.0002f) {
    //    if (constructor.genomeSize > 0) {
    //        int index = data.numberGen1.random(toInt(constructor.genomeSize - 1));
    //        constructor.genome[index] = data.numberGen1.randomByte();
    //    }
    //}
    //if (data.numberGen1.random() < 0.0005f && data.numberGen2.random() < 0.001) {

    //    auto newGenomeSize = min(MAX_GENOME_BYTES, toInt(constructor.genomeSize) + data.numberGen1.random(100));
    //    auto newGenome = data.objects.auxiliaryData.getAlignedSubArray(newGenomeSize);
    //    for (int i = 0; i < constructor.genomeSize; ++i) {
    //        newGenome[i] = constructor.genome[i];
    //    }
    //    for (int i = constructor.genomeSize; i < newGenomeSize; ++i) {
    //        newGenome[i] = data.numberGen1.randomByte();
    //    }
    //    constructor.genome = newGenome;
    //    constructor.genomeSize = newGenomeSize;
    //}
    //if (data.numberGen1.random() < 0.0005f && data.numberGen2.random() < 0.001) {

    //    constructor.genomeSize = max(0, toInt(constructor.genomeSize) - data.numberGen1.random(100));
    //    constructor.currentGenomePos = min(constructor.genomeSize, constructor.currentGenomePos);
    //}
}

__inline__ __device__ void MutationProcessor::mutateData(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto numGenomeCells = getNumGenomeCells(constructor);
    if (numGenomeCells == 0) {
        return;
    }
    auto cellIndex = data.numberGen1.random(numGenomeCells - 1);
    auto byteIndex = convertCellIndexToByteIndex(constructor, cellIndex);

    //basic property mutation
    if (data.numberGen1.randomBool()) {
        auto delta = data.numberGen1.random(CellBasicBytes - 2) + 1;  //+1 since cell function type should not be changed here
        byteIndex = (byteIndex + delta) % constructor.genomeSize;
        constructor.genome[byteIndex] = data.numberGen1.randomByte();
    }

    //cell function specific mutation
    else {
        auto nextCellFunctionGenomeBytes = getNextCellFunctionGenomeBytes(constructor, byteIndex);
        if (nextCellFunctionGenomeBytes > 0) {
            auto delta = data.numberGen1.random(nextCellFunctionGenomeBytes - 1);
            auto type = getNextCellFunctionType(constructor, byteIndex);
            //do not override makeSelfCopy flag and length bytes (at relative position is CellBasicBytes + ConstructorFixedBytes)!
            if (type == CellFunction_Constructor) {
                if (delta == ConstructorFixedBytes) {
                    return;
                }
                auto makeSelfCopy = getNextCellFunctionMakeSelfCopy(constructor, byteIndex);
                if (!makeSelfCopy && (delta == ConstructorFixedBytes + 1 || delta == ConstructorFixedBytes + 2)) {
                    return;
                }
            }
            if (type == CellFunction_Injector) {
                if (delta == InjectorFixedBytes) {
                    return;
                }
                auto makeSelfCopy = getNextCellFunctionMakeSelfCopy(constructor, byteIndex);
                if (!makeSelfCopy && (delta == InjectorFixedBytes + 1 || delta == InjectorFixedBytes + 2)) {
                    return;
                }
            }
            byteIndex = (byteIndex + CellBasicBytes + delta) % constructor.genomeSize;
            constructor.genome[byteIndex] = data.numberGen1.randomByte();
        }
    }
}

__inline__ __device__ void MutationProcessor::mutateNeuronData(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto numGenomeCells = getNumGenomeCells(constructor);
    if (numGenomeCells == 0) {
        return;
    }
    auto cellIndex = data.numberGen1.random(numGenomeCells - 1);
    auto byteIndex = convertCellIndexToByteIndex(constructor, cellIndex);

    auto type = getNextCellFunctionType(constructor, byteIndex);
    if (type == CellFunction_Neuron) {
        auto delta = data.numberGen1.random(NeuronBytes - 1);
        byteIndex = (byteIndex + CellBasicBytes + delta) % constructor.genomeSize;
        constructor.genome[byteIndex] = data.numberGen1.randomByte();
    }
}

__inline__ __device__ void MutationProcessor::mutateCellFunction(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto numGenomeCells = getNumGenomeCells(constructor);
    if (numGenomeCells == 0) {
        return;
    }
    auto origCurrentCellIndex = convertByteIndexToCellIndex(constructor, constructor.currentGenomePos);

    auto mutationCellIndex = data.numberGen1.random(numGenomeCells - 1);
    auto mutationByteIndex = convertCellIndexToByteIndex(constructor, mutationCellIndex);

    auto newCellFunction = data.numberGen1.random(CellFunction_Count - 1);
    auto makeSelfCopy = false;
    auto subGenomeSize = 0;
    if (newCellFunction == CellFunction_Constructor || newCellFunction == CellFunction_Injector) {
        subGenomeSize = data.numberGen1.random(CellFunctionMutationMaxGenomeSize);
        makeSelfCopy = data.numberGen1.randomBool();
    }
    auto newCellFunctionSize = getCellFunctionDataSize(newCellFunction, makeSelfCopy, subGenomeSize);
    auto oldCellFunctionSize = getNextCellFunctionGenomeBytes(constructor, mutationByteIndex);

    auto targetGenomeSize = max(min(toInt(constructor.genomeSize) + newCellFunctionSize - oldCellFunctionSize, MAX_GENOME_BYTES), 0);
    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);

    for (int i = 0; i < mutationByteIndex + CellBasicBytes; ++i) {
        targetGenome[i % targetGenomeSize] = constructor.genome[i % constructor.genomeSize];
    }
    targetGenome[mutationByteIndex] = newCellFunction;
    setRandomCellFunctionData(data, targetGenome, targetGenomeSize, mutationByteIndex + CellBasicBytes, newCellFunction, makeSelfCopy, subGenomeSize);

    for (int i = mutationByteIndex + CellBasicBytes + oldCellFunctionSize; i < constructor.genomeSize; ++i) {
        targetGenome[(i + newCellFunctionSize - oldCellFunctionSize) % targetGenomeSize] = constructor.genome[i];
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
    constructor.currentGenomePos = origCurrentCellIndex == numGenomeCells ? targetGenomeSize : convertCellIndexToByteIndex(constructor, origCurrentCellIndex);
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
        auto genomeByteIndex =
            cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
        genomeData[(genomeByteIndex + offset) % targetGenomeSize] = makeSelfCopy ? 1 : 0;
        if (!makeSelfCopy) {
            genomeData[(genomeByteIndex + offset + 1) % targetGenomeSize] = static_cast<uint8_t>(subGenomeSize & 0xff);
            genomeData[(genomeByteIndex + offset + 2) % targetGenomeSize] = static_cast<uint8_t>((subGenomeSize >> 8) & 0xff);
        }
    }
}

__inline__ __device__ int MutationProcessor::getNumGenomeCells(ConstructorFunction const& constructor)
{
    int result = 0;
    int currentByteIndex = 0;
    for (; result < constructor.genomeSize; ++result) {
        if (currentByteIndex >= constructor.genomeSize) {
            break;
        }
        currentByteIndex += CellBasicBytes + getNextCellFunctionGenomeBytes(constructor, currentByteIndex);
    }

    return result;
}

__inline__ __device__ int MutationProcessor::convertCellIndexToByteIndex(ConstructorFunction const& constructor, int cellIndex)
{
    int currentByteIndex = 0;
    for (int currentCellIndex = 0; currentCellIndex < cellIndex; ++currentCellIndex) {
        if (currentByteIndex >= constructor.genomeSize) {
            break;
        }
        currentByteIndex += CellBasicBytes + getNextCellFunctionGenomeBytes(constructor, currentByteIndex);
    }

    return currentByteIndex % constructor.genomeSize;
}

__inline__ __device__ int MutationProcessor::convertByteIndexToCellIndex(ConstructorFunction const& constructor, int byteIndex)
{
    int currentByteIndex = 0;
    for (int currentCellIndex = 0; currentByteIndex <= byteIndex; ++currentCellIndex) {
        if (currentByteIndex == byteIndex) {
            return currentCellIndex;
        }
        currentByteIndex += CellBasicBytes + getNextCellFunctionGenomeBytes(constructor, currentByteIndex);
    }
    return 0;
}

__inline__ __device__ int MutationProcessor::getNextCellFunctionGenomeBytes(ConstructorFunction const& constructor, int byteIndex)
{
    auto cellFunction = getNextCellFunctionType(constructor, byteIndex);
    switch (cellFunction) {
    case CellFunction_Neuron:
        return NeuronBytes;
    case CellFunction_Transmitter:
        return TransmitterBytes;
    case CellFunction_Constructor: {
        auto makeCopyIndex = byteIndex + CellBasicBytes + ConstructorFixedBytes;
        auto isMakeCopy = GenomeDecoder::convertByteToBool(constructor.genome[makeCopyIndex % constructor.genomeSize]);
        if (isMakeCopy) {
            return ConstructorFixedBytes + 1;
        } else {
            auto genomeSizeIndex = byteIndex + CellBasicBytes + ConstructorFixedBytes + 1;
            auto genomeSize = GenomeDecoder::convertBytesToWord(
                constructor.genome[genomeSizeIndex % constructor.genomeSize], constructor.genome[(genomeSizeIndex + 1) % constructor.genomeSize]);
            return ConstructorFixedBytes + 3 + genomeSize;
        }
    }
    case CellFunction_Sensor:
        return SensorBytes;
    case CellFunction_Nerve:
        return NerveBytes;
    case CellFunction_Attacker:
        return AttackerBytes;
    case CellFunction_Injector: {
        auto makeCopyIndex = byteIndex + CellBasicBytes + InjectorFixedBytes;
        auto isMakeCopy = GenomeDecoder::convertByteToBool(constructor.genome[makeCopyIndex % constructor.genomeSize]);
        if (isMakeCopy) {
            return InjectorFixedBytes + 1;
        } else {
            auto genomeSizeIndex = byteIndex + CellBasicBytes + InjectorFixedBytes + 1;
            auto genomeSize = GenomeDecoder::convertBytesToWord(
                constructor.genome[genomeSizeIndex % constructor.genomeSize], constructor.genome[(genomeSizeIndex + 1) % constructor.genomeSize]);
            return InjectorFixedBytes + 3 + genomeSize;
        }
    }
    case CellFunction_Muscle:
        return MuscleBytes;
    default:
        return 0;
    }
}

__inline__ __device__ int MutationProcessor::getNextCellFunctionType(ConstructorFunction const& constructor, int byteIndex)
{
    return constructor.genome[byteIndex % constructor.genomeSize] % CellFunction_Count;
}

__inline__ __device__ bool MutationProcessor::getNextCellFunctionMakeSelfCopy(ConstructorFunction const& constructor, int byteIndex)
{
    switch (getNextCellFunctionType(constructor, byteIndex)) {
    case CellFunction_Constructor:
        return GenomeDecoder::convertByteToBool(constructor.genome[(byteIndex + CellBasicBytes + ConstructorFixedBytes) % constructor.genomeSize]);
    case CellFunction_Injector:
        return GenomeDecoder::convertByteToBool(constructor.genome[(byteIndex + CellBasicBytes + InjectorFixedBytes) % constructor.genomeSize]);
    default:
        return false;
    }
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
