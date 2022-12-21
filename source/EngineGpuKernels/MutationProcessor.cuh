#pragma once

#include "EngineInterface/Enums.h"

#include "CellConnectionProcessor.cuh"
#include "GenomeDecoder.cuh"

class MutationProcessor
{
public:
    __inline__ __device__ static void applyRandomMutation(SimulationData& data, Cell* cell);

    __inline__ __device__ static void mutateData(SimulationData& data, Cell* cell);
    __inline__ __device__ static void mutateNeuronData(SimulationData& data, Cell* cell);

private:
    
    //internal constants
    static int constexpr CellFunctionMutationMaxSizeDelta = 100;
    static int constexpr CellFunctionMutationMaxGenomeSize = 50;
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
    __inline__ __device__ static int getGenomeByteIndex(ConstructorFunction const& constructor, int cellIndex);
    __inline__ __device__ static int getGenomeCellIndex(ConstructorFunction const& constructor, int byteIndex);
    __inline__ __device__ static int getNextCellFunctionGenomeBytes(ConstructorFunction const& constructor, int genomePos);
    __inline__ __device__ static int getNextCellFunctionType(ConstructorFunction const& constructor, int genomePos);
    __inline__ __device__ static bool getNextCellFunctionMakeSelfCopy(ConstructorFunction const& constructor, int genomePos);
    __inline__ __device__ static int getCellFunctionDataSize(
        Enums::CellFunction cellFunction,
        bool makeSelfCopy,
        int genomeSize);  //genomeSize only relevant for cellFunction = constructor or injector
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void MutationProcessor::mutateData(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto numGenomeCells = getNumGenomeCells(constructor);
    if (numGenomeCells == 0) {
        return;
    }
    auto cellIndex = data.numberGen1.random(numGenomeCells - 1);
    auto genomePos = getGenomeByteIndex(constructor, cellIndex);

    //basic property mutation
    if (data.numberGen1.randomBool()) {
        auto delta = data.numberGen1.random(CellBasicBytes - 2) + 1;  //+1 since cell function type should not be changed here
        genomePos = (genomePos + delta) % constructor.genomeSize;
        constructor.genome[genomePos] = data.numberGen1.randomByte();
    }

    //cell function specific mutation
    else {
        auto nextCellFunctionGenomeBytes = getNextCellFunctionGenomeBytes(constructor, genomePos);
        if (nextCellFunctionGenomeBytes > 0) {
            auto delta = data.numberGen1.random(nextCellFunctionGenomeBytes - 1);
            auto type = getNextCellFunctionType(constructor, genomePos);
            //do not override makeSelfCopy flag and length bytes (at relative position is CellBasicBytes + ConstructorFixedBytes)!
            if (type == Enums::CellFunction_Constructor) {
                if (delta == ConstructorFixedBytes) {
                    return;
                }
                auto makeSelfCopy = getNextCellFunctionMakeSelfCopy(constructor, genomePos);
                if (!makeSelfCopy && (delta == ConstructorFixedBytes + 1 || delta == ConstructorFixedBytes + 2)) {
                    return;
                }
            }
            if (type == Enums::CellFunction_Injector) {
                if (delta == InjectorFixedBytes) {
                    return;
                }
                auto makeSelfCopy = getNextCellFunctionMakeSelfCopy(constructor, genomePos);
                if (!makeSelfCopy && (delta == InjectorFixedBytes + 1 || delta == InjectorFixedBytes + 2)) {
                    return;
                }
            }
            genomePos = (genomePos + CellBasicBytes + delta) % constructor.genomeSize;
            constructor.genome[genomePos] = data.numberGen1.randomByte();
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
    auto genomePos = getGenomeByteIndex(constructor, cellIndex);

    auto type = getNextCellFunctionType(constructor, genomePos);
    if (type == Enums::CellFunction_Neuron) {
        auto delta = data.numberGen1.random(NeuronBytes - 1);
        genomePos = (genomePos + CellBasicBytes + delta) % constructor.genomeSize;
        constructor.genome[genomePos] = data.numberGen1.randomByte();
    }
}

__inline__ __device__ void MutationProcessor::applyRandomMutation(SimulationData& data, Cell* cell)
{
    auto cellFunctionConstructorMutationNeuronProbability =
        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionConstructorMutationNeuronDataProbability, data, cell->absPos);
    auto cellFunctionConstructorMutationDataProbability =
        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionConstructorMutationDataProbability, data, cell->absPos);

    if (data.numberGen1.random() < cellFunctionConstructorMutationNeuronProbability) {
        mutateNeuronData(data, cell);
    }

    if (data.numberGen1.random() < cellFunctionConstructorMutationDataProbability) {
        mutateData(data, cell);
    }

    //cell function changing mutation
    //if (data.numberGen1.random() < 0.0002f) {
    //    auto numCellIndices = getNumGenomeCells(constructor);
    //    if (numCellIndices == 0) {
    //        return;
    //    }
    //    auto mutationCellIndex = data.numberGen1.random(numCellIndices - 1);
    //    auto sourceGenomePos = getGenomeByteIndex(constructor, mutationCellIndex);
    //    auto sourceRemainingGenomePos = sourceGenomePos + CellBasicBytes + getNextCellFunctionGenomeBytes(constructor, sourceGenomePos);

    //    auto targetGenomeSize = constructor.genomeSize + CellFunctionMutationMaxSizeDelta;
    //    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);

    //    for (int pos = 0; pos < sourceGenomePos + CellBasicBytes; ++pos) {
    //        targetGenome[pos] = constructor.genome[pos];
    //    }

    //    targetGenome[sourceGenomePos] = data.numberGen1.random(Enums::CellFunction_Count - 1);
    //    auto cellFunctionDataSize =
    //        getCellFunctionDataSize(targetGenome[sourceGenomePos], data.numberGen1.randomByte(), data.numberGen1.random(CellFunctionMutationMaxGenomeSize));
    //    for (int pos = sourceGenomePos + CellBasicBytes; pos < sourceGenomePos + CellBasicBytes + cellFunctionDataSize; ++pos) {
    //        targetGenome[pos] = data.numberGen1.randomByte();
    //    }

    //    auto targetPos = sourceGenomePos + CellBasicBytes + cellFunctionDataSize;
    //    if (sourceRemainingGenomePos > sourceGenomePos) {
    //        auto sourcePos = sourceRemainingGenomePos;
    //        for (; sourcePos < constructor.genomeSize; ++sourcePos, ++targetPos) {
    //            targetGenome[targetPos] = constructor.genome[sourcePos];
    //        }
    //    }
    //    auto cellIndex = getGenomeCellIndex(constructor, mutationCellIndex);
    //    constructor.genome = targetGenome;
    //    constructor.genomeSize = targetPos;
    //    constructor.currentGenomePos = getGenomeByteIndex(constructor, cellIndex);
    //}

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

__inline__ __device__ int MutationProcessor::getGenomeByteIndex(ConstructorFunction const& constructor, int cellIndex)
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

__inline__ __device__ int MutationProcessor::getGenomeCellIndex(ConstructorFunction const& constructor, int byteIndex)
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

__inline__ __device__ int MutationProcessor::getNextCellFunctionGenomeBytes(ConstructorFunction const& constructor, int genomePos)
{
    auto cellFunction = getNextCellFunctionType(constructor, genomePos);
    switch (cellFunction) {
    case Enums::CellFunction_Neuron:
        return NeuronBytes;
    case Enums::CellFunction_Transmitter:
        return TransmitterBytes;
    case Enums::CellFunction_Constructor: {
        auto makeCopyIndex = genomePos + CellBasicBytes + ConstructorFixedBytes;
        auto isMakeCopy = GenomeDecoder::convertByteToBool(constructor.genome[makeCopyIndex % constructor.genomeSize]);
        if (isMakeCopy) {
            return ConstructorFixedBytes + 1;
        } else {
            auto genomeSizeIndex = genomePos + CellBasicBytes + ConstructorFixedBytes + 1;
            auto genomeSize = GenomeDecoder::convertBytesToWord(
                constructor.genome[genomeSizeIndex % constructor.genomeSize], constructor.genome[(genomeSizeIndex + 1) % constructor.genomeSize]);
            return ConstructorFixedBytes + 3 + genomeSize;
        }
    }
    case Enums::CellFunction_Sensor:
        return SensorBytes;
    case Enums::CellFunction_Nerve:
        return NerveBytes;
    case Enums::CellFunction_Attacker:
        return AttackerBytes;
    case Enums::CellFunction_Injector: {
        auto makeCopyIndex = genomePos + CellBasicBytes + InjectorFixedBytes;
        auto isMakeCopy = GenomeDecoder::convertByteToBool(constructor.genome[makeCopyIndex % constructor.genomeSize]);
        if (isMakeCopy) {
            return InjectorFixedBytes + 1;
        } else {
            auto genomeSizeIndex = genomePos + CellBasicBytes + InjectorFixedBytes + 1;
            auto genomeSize = GenomeDecoder::convertBytesToWord(
                constructor.genome[genomeSizeIndex % constructor.genomeSize], constructor.genome[(genomeSizeIndex + 1) % constructor.genomeSize]);
            return InjectorFixedBytes + 3 + genomeSize;
        }
    }
    case Enums::CellFunction_Muscle:
        return MuscleBytes;
    default:
        return 0;
    }
}

__inline__ __device__ int MutationProcessor::getNextCellFunctionType(ConstructorFunction const& constructor, int genomePos)
{
    return constructor.genome[genomePos] % Enums::CellFunction_Count;
}

__inline__ __device__ bool MutationProcessor::getNextCellFunctionMakeSelfCopy(ConstructorFunction const& constructor, int genomePos)
{
    switch (getNextCellFunctionType(constructor, genomePos)) {
    case Enums::CellFunction_Constructor:
        return GenomeDecoder::convertByteToBool(constructor.genome[(genomePos + CellBasicBytes + ConstructorFixedBytes) % constructor.genomeSize]);
    case Enums::CellFunction_Injector:
        return GenomeDecoder::convertByteToBool(constructor.genome[(genomePos + CellBasicBytes + InjectorFixedBytes) % constructor.genomeSize]);
    default:
        return false;
    }
}

__inline__ __device__ int MutationProcessor::getCellFunctionDataSize(Enums::CellFunction cellFunction, bool makeSelfCopy, int genomeSize)
{
    switch (cellFunction) {
    case Enums::CellFunction_Neuron:
        return NeuronBytes;
    case Enums::CellFunction_Transmitter:
        return TransmitterBytes;
    case Enums::CellFunction_Constructor: {
        return makeSelfCopy ? ConstructorFixedBytes + 1 : ConstructorFixedBytes + 3 + genomeSize;
    }
    case Enums::CellFunction_Sensor:
        return SensorBytes;
    case Enums::CellFunction_Nerve:
        return NerveBytes;
    case Enums::CellFunction_Attacker:
        return AttackerBytes;
    case Enums::CellFunction_Injector: {
        return makeSelfCopy ? InjectorFixedBytes + 1 : InjectorFixedBytes + 3 + genomeSize;
    }
    case Enums::CellFunction_Muscle:
        return MuscleBytes;
    default:
        return 0;
    }
}
