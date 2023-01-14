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
    __inline__ __device__ static int getRandomGenomeByteIndex(SimulationData& data, uint8_t* genome, int genomeSize);
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
    __inline__ __device__ static int findStartByteIndex(uint8_t* genome, int genomeSize, int refIndex);
    __inline__ __device__ static int convertCellIndexToByteIndex(uint8_t* genome, int genomeSize, int cellIndex);
    __inline__ __device__ static int convertByteIndexToCellIndex(uint8_t* genome, int genomeSize, int byteIndex);
    __inline__ __device__ static int getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int byteIndex);
    __inline__ __device__ static int getNextCellFunctionType(uint8_t* genome, int genomeSize, int byteIndex);
    __inline__ __device__ static int getNextSubGenomeSize(uint8_t* genome, int genomeSize, int byteIndex);  //prerequisites: (constructor or injector) and !makeSelfCopy
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
    auto byteIndex = getRandomGenomeByteIndex(data, genome, genomeSize);

    //basic property mutation
    if (data.numberGen1.randomBool()) {
        auto randomDelta = data.numberGen1.random(CellBasicBytes - 1);
        if (randomDelta == 0 || randomDelta == 5) { //no cell function type and color changes
            return;
        }
        writeByte(genome, genomeSize, byteIndex + randomDelta, data.numberGen1.randomByte());
    }

    //cell function specific mutation
    else {
        auto nextCellFunctionGenomeBytes = getNextCellFunctionDataSize(genome, genomeSize, byteIndex);
        auto type = getNextCellFunctionType(genome, genomeSize, byteIndex);
        if (type == CellFunction_Constructor || type == CellFunction_Injector) {
            nextCellFunctionGenomeBytes = type == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
        }
        if (nextCellFunctionGenomeBytes > 0) {
            auto randomDelta = data.numberGen1.random(nextCellFunctionGenomeBytes - 1);
            writeByte(genome, genomeSize, byteIndex + CellBasicBytes + randomDelta, data.numberGen1.randomByte());
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
    auto byteIndex = getRandomGenomeByteIndex(data, genome, genomeSize);

    auto type = getNextCellFunctionType(genome, genomeSize, byteIndex);
    if (type == CellFunction_Neuron) {
        auto delta = data.numberGen1.random(NeuronBytes - 1);
        writeByte(genome, genomeSize, byteIndex + CellBasicBytes + delta, data.numberGen1.randomByte());
    }
}

__inline__ __device__ void MutationProcessor::mutateCellFunction(SimulationData& data, Cell* cell)
{
    //auto& constructor = cell->cellFunctionData.constructor;
    //auto numGenomeCells = getNumGenomeCells(constructor);
    //if (numGenomeCells == 0) {
    //    return;
    //}
    //auto origCurrentCellIndex = convertByteIndexToCellIndex(constructor, constructor.currentGenomePos);

    //auto mutationCellIndex = data.numberGen1.random(numGenomeCells - 1);
    //auto mutationByteIndex = convertCellIndexToByteIndex(constructor, mutationCellIndex);

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
    //constructor.currentGenomePos = origCurrentCellIndex == numGenomeCells ? targetGenomeSize : convertCellIndexToByteIndex(constructor, origCurrentCellIndex);
}

__inline__ __device__ void MutationProcessor::mutateInsertion(SimulationData& data, Cell* cell)
{
    //auto& constructor = cell->cellFunctionData.constructor;
    //auto numGenomeCells = getNumGenomeCells(constructor);
    //if (numGenomeCells == 0) {
    //    return;
    //}
    //auto origCurrentCellIndex = convertByteIndexToCellIndex(constructor, constructor.currentGenomePos);

    //auto mutationCellIndex = data.numberGen1.random(numGenomeCells - 1);
    //auto mutationByteIndex = convertCellIndexToByteIndex(constructor, mutationCellIndex);

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
    //constructor.currentGenomePos = convertCellIndexToByteIndex(constructor, origCurrentCellIndex);
}

__inline__ __device__ int MutationProcessor::getRandomGenomeByteIndex(SimulationData& data, uint8_t* genome, int genomeSize)
{
    if (genomeSize == 0) {
        return 0;
    }
    auto randomRefIndex = data.numberGen1.random(genomeSize - 1);

    auto origGenomeSize = genomeSize;
    int result = 0;
    for (int depth = 0; depth < 20; ++depth) {
        auto byteIndex = findStartByteIndex(genome, genomeSize, randomRefIndex);
        result += byteIndex;
        auto cellFunction = getNextCellFunctionType(genome, genomeSize, byteIndex);

        if (cellFunction == CellFunction_Constructor || cellFunction == CellFunction_Injector) {
            auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
            auto makeSelfCopy = GenomeDecoder::convertByteToBool(readByte(genome, genomeSize, byteIndex + CellBasicBytes + cellFunctionFixedBytes));
            if (makeSelfCopy) {
                break;
            } else {
                auto subGenomeStartIndex = byteIndex + CellBasicBytes + cellFunctionFixedBytes + 3;
                auto subGenomeSize = getNextSubGenomeSize(genome, genomeSize, byteIndex);
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
        auto genomeByteIndex =
            cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
        genomeData[(genomeByteIndex + offset) % targetGenomeSize] = makeSelfCopy ? 1 : 0;
        if (!makeSelfCopy) {
            genomeData[(genomeByteIndex + offset + 1) % targetGenomeSize] = static_cast<uint8_t>(subGenomeSize & 0xff);
            genomeData[(genomeByteIndex + offset + 2) % targetGenomeSize] = static_cast<uint8_t>((subGenomeSize >> 8) & 0xff);
        }
    }
}

__inline__ __device__ int MutationProcessor::getNumGenomeCells(uint8_t* genome, int genomeSize)
{
    int result = 0;
    int currentByteIndex = 0;
    for (; result < genomeSize && currentByteIndex < genomeSize; ++result) {
        currentByteIndex += CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentByteIndex);
    }

    return result;
}

__inline__ __device__ int MutationProcessor::findStartByteIndex(uint8_t* genome, int genomeSize, int refIndex)
{
    int currentByteIndex = 0;
    for (; currentByteIndex <= refIndex;) {
        auto prevCurrentByteIndex = currentByteIndex;
        currentByteIndex += CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentByteIndex);
        if (currentByteIndex > refIndex) {
            return prevCurrentByteIndex;
        }
    }
    return 0;
}

__inline__ __device__ int MutationProcessor::convertCellIndexToByteIndex(uint8_t* genome, int genomeSize, int cellIndex)
{
    int currentByteIndex = 0;
    for (int currentCellIndex = 0; currentCellIndex < cellIndex; ++currentCellIndex) {
        if (currentByteIndex >= genomeSize) {
            break;
        }
        currentByteIndex += CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentByteIndex);
    }

    return currentByteIndex % genomeSize;
}

__inline__ __device__ int MutationProcessor::convertByteIndexToCellIndex(uint8_t* genome, int genomeSize, int byteIndex)
{
    int currentByteIndex = 0;
    for (int currentCellIndex = 0; currentByteIndex <= byteIndex; ++currentCellIndex) {
        if (currentByteIndex == byteIndex) {
            return currentCellIndex;
        }
        currentByteIndex += CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentByteIndex);
    }
    return 0;
}

__inline__ __device__ int MutationProcessor::getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int byteIndex)
{
    auto cellFunction = getNextCellFunctionType(genome, genomeSize, byteIndex);
    switch (cellFunction) {
    case CellFunction_Neuron:
        return NeuronBytes;
    case CellFunction_Transmitter:
        return TransmitterBytes;
    case CellFunction_Constructor: {
        auto isMakeCopy = GenomeDecoder::convertByteToBool(readByte(genome, genomeSize, byteIndex + CellBasicBytes + ConstructorFixedBytes));
        if (isMakeCopy) {
            return ConstructorFixedBytes + 1;
        } else {
            return ConstructorFixedBytes + 3 + getNextSubGenomeSize(genome, genomeSize, byteIndex);
        }
    }
    case CellFunction_Sensor:
        return SensorBytes;
    case CellFunction_Nerve:
        return NerveBytes;
    case CellFunction_Attacker:
        return AttackerBytes;
    case CellFunction_Injector: {
        auto isMakeCopy = GenomeDecoder::convertByteToBool(readByte(genome, genomeSize, byteIndex + CellBasicBytes + InjectorFixedBytes));
        if (isMakeCopy) {
            return InjectorFixedBytes + 1;
        } else {
            return InjectorFixedBytes + 3 + getNextSubGenomeSize(genome, genomeSize, byteIndex);
        }
    }
    case CellFunction_Muscle:
        return MuscleBytes;
    default:
        return 0;
    }
}

__inline__ __device__ int MutationProcessor::getNextCellFunctionType(uint8_t* genome, int genomeSize, int byteIndex)
{
    return readByte(genome, genomeSize, byteIndex) % CellFunction_Count;
}

__inline__ __device__ int MutationProcessor::getNextSubGenomeSize(uint8_t* genome, int genomeSize, int byteIndex)
{
    auto cellFunction = getNextCellFunctionType(genome, genomeSize, byteIndex);
    auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? ConstructorFixedBytes : InjectorFixedBytes;
    auto subGenomeSizeIndex = byteIndex + CellBasicBytes + cellFunctionFixedBytes + 1;
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
