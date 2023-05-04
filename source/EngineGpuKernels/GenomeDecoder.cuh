#pragma once

#include "EngineInterface/CellFunctionConstants.h"
#include "Base.cuh"
#include "Cell.cuh"

struct GenomeInfo
{
    ConstructionShape shape;
    bool singleConstruction;
    bool separateConstruction;
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
};

class GenomeDecoder
{
public:
    //automatic increment currentGenomePos
    __inline__ __device__ static bool readBool(ConstructorFunction& constructor);
    __inline__ __device__ static uint8_t readByte(ConstructorFunction& constructor);
    __inline__ __device__ static int readOptionalByte(ConstructorFunction& constructor, int moduloValue);
    __inline__ __device__ static int readWord(ConstructorFunction& constructor);
    __inline__ __device__ static float readFloat(ConstructorFunction& constructor);  //return values from -1 to 1
    __inline__ __device__ static float readEnergy(ConstructorFunction& constructor);  //return values from 36 to 1060
    __inline__ __device__ static float readAngle(ConstructorFunction& constructor);

    __inline__ __device__ static bool isAtFirstNode(ConstructorFunction const& constructor);
    __inline__ __device__ static bool isAtLastNode(ConstructorFunction const& constructor);
    __inline__ __device__ static bool isFinished(ConstructorFunction const& constructor);
    __inline__ __device__ static bool isFinishedSingleConstruction(ConstructorFunction const& constructor);

    template <typename GenomeHolderSource, typename GenomeHolderTarget>
    __inline__ __device__ static void copyGenome(SimulationData& data, GenomeHolderSource& source, GenomeHolderTarget& target);
    __inline__ __device__ static GenomeInfo readGenomeInfo(ConstructorFunction const& constructor);
    __inline__ __device__ static int readWord(uint8_t* genome, int nodeAddress);
    __inline__ __device__ static void writeWord(uint8_t* genome, int address, int word);
    __inline__ __device__ static bool convertByteToBool(uint8_t b);
    __inline__ __device__ static int convertBytesToWord(uint8_t b1, uint8_t b2);
    __inline__ __device__ static void convertWordToBytes(int word, uint8_t& b1, uint8_t& b2);

    __inline__ __device__ static int getRandomGenomeNodeAddress(
        SimulationData& data,
        uint8_t* genome,
        int genomeSize,
        bool considerZeroSubGenomes,
        int* subGenomesSizeIndices = nullptr,
        int* numSubGenomesSizeIndices = nullptr);
    __inline__ __device__ static void setRandomCellFunctionData(
        SimulationData& data,
        uint8_t* genomeData,
        int nodeAddress,
        CellFunction const& cellFunction,
        bool makeSelfCopy,
        int subGenomeSize);

    __inline__ __device__ static int getNumGenomeCells(uint8_t* genome, int genomeSize);
    __inline__ __device__ static int getNodeAddress(uint8_t* genome, int genomeSize, int nodeIndex);
    __inline__ __device__ static int findStartNodeAddress(uint8_t* genome, int genomeSize, int refIndex);
    __inline__ __device__ static int getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int nodeAddress);
    __inline__ __device__ static int getNextCellFunctionType(uint8_t* genome, int nodeAddress);
    __inline__ __device__ static bool isNextCellSelfCopy(uint8_t* genome, int nodeAddress);
    __inline__ __device__ static int getNextCellColor(uint8_t* genome, int nodeAddress);
    __inline__ __device__ static void setNextCellFunctionType(uint8_t* genome, int nodeAddress, CellFunction cellFunction);
    __inline__ __device__ static void setNextCellColor(uint8_t* genome, int nodeAddress, int color);
    __inline__ __device__ static void setNextAngle(uint8_t* genome, int nodeAddress, uint8_t angle);
    __inline__ __device__ static void setNextRequiredConnections(uint8_t* genome, int nodeAddress, uint8_t angle);
    __inline__ __device__ static int
    getNextSubGenomeSize(uint8_t* genome, int genomeSize, int nodeAddress);  //prerequisites: (constructor or injector) and !makeSelfCopy
    __inline__ __device__ static int getCellFunctionDataSize(
        CellFunction cellFunction,
        bool makeSelfCopy,
        int genomeSize);  //genomeSize only relevant for cellFunction = constructor or injector
    __inline__ __device__ static bool hasSelfCopy(uint8_t* genome, int genomeSize);

    static auto constexpr MAX_SUBGENOME_RECURSION_DEPTH = 30;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ bool GenomeDecoder::readBool(ConstructorFunction& constructor)
{
    return convertByteToBool(readByte(constructor));
}

__inline__ __device__ uint8_t GenomeDecoder::readByte(ConstructorFunction& constructor)
{
    if (isFinished(constructor)) {
        return 0;
    }
    uint8_t result = constructor.genome[constructor.currentGenomePos++];
    return result;
}

__inline__ __device__ int GenomeDecoder::readOptionalByte(ConstructorFunction& constructor, int moduloValue)
{
    auto result = static_cast<int>(readByte(constructor));
    result = result > 127 ? -1 : result % moduloValue;
    return result;
}

__inline__ __device__ int GenomeDecoder::readWord(ConstructorFunction& constructor)
{
    auto b1 = readByte(constructor);
    auto b2 = readByte(constructor);
    return convertBytesToWord(b1, b2);
}

__inline__ __device__ float GenomeDecoder::readFloat(ConstructorFunction& constructor)
{
    return static_cast<float>(static_cast<int8_t>(readByte(constructor))) / 128;
}

__inline__ __device__ float GenomeDecoder::readEnergy(ConstructorFunction& constructor)
{
    return static_cast<float>(static_cast<int8_t>(readByte(constructor))) / 128 * 512 + 548;
}

__inline__ __device__ float GenomeDecoder::readAngle(ConstructorFunction& constructor)
{
    return static_cast<float>(static_cast<int8_t>(readByte(constructor))) / 120 * 180;
}

__inline__ __device__ bool GenomeDecoder::isAtFirstNode(ConstructorFunction const& constructor)
{
    return constructor.currentGenomePos <= Const::GenomeInfoSize;
}

__inline__ __device__ bool GenomeDecoder::isAtLastNode(ConstructorFunction const& constructor)
{
    auto nextNodeBytes =
        Const::CellBasicBytes + getNextCellFunctionDataSize(constructor.genome, toInt(constructor.genomeSize), toInt(constructor.currentGenomePos));
    return toInt(constructor.currentGenomePos) + nextNodeBytes >= toInt(constructor.genomeSize);
}

__inline__ __device__ bool GenomeDecoder::isFinished(ConstructorFunction const& constructor)
{
    return constructor.currentGenomePos >= constructor.genomeSize;
}

template <typename GenomeHolderSource, typename GenomeHolderTarget>
__inline__ __device__ void GenomeDecoder::copyGenome(SimulationData& data, GenomeHolderSource& source, GenomeHolderTarget& target)
{
    bool makeGenomeCopy = readBool(source);
    if (!makeGenomeCopy) {
        auto size = readWord(source);
        size = min(size, toInt(source.genomeSize) - toInt(source.currentGenomePos));
        target.genomeSize = size;
        target.genome = data.objects.auxiliaryData.getAlignedSubArray(size);
        //#TODO can be optimized
        for (int i = 0; i < size; ++i) {
            target.genome[i] = readByte(source);
        }
    } else {
        auto size = source.genomeSize;
        target.genomeSize = size;
        target.genome = data.objects.auxiliaryData.getAlignedSubArray(size);
        //#TODO can be optimized
        for (int i = 0; i < size; ++i) {
            target.genome[i] = source.genome[i];
        }
    }
}

__inline__ __device__ bool GenomeDecoder::isFinishedSingleConstruction(ConstructorFunction const& constructor)
{
    auto genomeInfo = readGenomeInfo(constructor);
    return genomeInfo.singleConstruction
        && (constructor.currentGenomePos >= constructor.genomeSize || (constructor.currentGenomePos == 0 && constructor.genomeSize == Const::GenomeInfoSize));
}

__inline__ __device__ GenomeInfo GenomeDecoder::readGenomeInfo(ConstructorFunction const& constructor)
{
    GenomeInfo result;
    if (constructor.genomeSize < Const::GenomeInfoSize) {
        CUDA_THROW_NOT_IMPLEMENTED();
    }
    result.shape = constructor.genome[0] % ConstructionShape_Count;
    result.singleConstruction = GenomeDecoder::convertByteToBool(constructor.genome[1]);
    result.separateConstruction = GenomeDecoder::convertByteToBool(constructor.genome[2]);
    result.angleAlignment = constructor.genome[3] % ConstructorAngleAlignment_Count;
    result.stiffness = toFloat(constructor.genome[4]) / 255;
    return result;
}

__inline__ __device__ int GenomeDecoder::readWord(uint8_t* genome, int address)
{
    return GenomeDecoder::convertBytesToWord(genome[address], genome[address + 1]);
}

__inline__ __device__ void GenomeDecoder::writeWord(uint8_t* genome, int address, int word)
{
    GenomeDecoder::convertWordToBytes(word, genome[address], genome[address + 1]);
}

__inline__ __device__ bool GenomeDecoder::convertByteToBool(uint8_t b)
{
    return static_cast<int8_t>(b) > 0;
}

__inline__ __device__ int GenomeDecoder::convertBytesToWord(uint8_t b1, uint8_t b2)
{
    return static_cast<int>(b1) | (static_cast<int>(b2 << 8));
}

__inline__ __device__ void GenomeDecoder::convertWordToBytes(int word, uint8_t& b1, uint8_t& b2)
{
    b1 = static_cast<uint8_t>(word & 0xff);
    b2 = static_cast<uint8_t>((word >> 8) & 0xff);
}

__inline__ __device__ int GenomeDecoder::getRandomGenomeNodeAddress(
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
        auto nodeAddress = findStartNodeAddress(genome, genomeSize, randomRefIndex);
        result += nodeAddress;
        auto cellFunction = getNextCellFunctionType(genome, nodeAddress);

        if (cellFunction == CellFunction_Constructor || cellFunction == CellFunction_Injector) {
            auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? Const::ConstructorFixedBytes : Const::InjectorFixedBytes;
            auto makeSelfCopy = GenomeDecoder::convertByteToBool(genome[nodeAddress + Const::CellBasicBytes + cellFunctionFixedBytes]);
            if (makeSelfCopy) {
                break;
            } else {
                if (numSubGenomesSizeIndices) {
                    subGenomesSizeIndices[*numSubGenomesSizeIndices] = result + Const::CellBasicBytes + cellFunctionFixedBytes + 1;
                    ++(*numSubGenomesSizeIndices);
                }
                auto subGenomeStartIndex = nodeAddress + Const::CellBasicBytes + cellFunctionFixedBytes + 3;
                auto subGenomeSize = getNextSubGenomeSize(genome, genomeSize, nodeAddress);
                if (subGenomeSize == Const::GenomeInfoSize) {
                    if (considerZeroSubGenomes && data.numberGen1.randomBool()) {
                        result += Const::CellBasicBytes + cellFunctionFixedBytes + 3 + Const::GenomeInfoSize;
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
                result += Const::CellBasicBytes + cellFunctionFixedBytes + 3;
            }
        } else {
            break;
        }
    }
    return result;
}

__inline__ __device__ void GenomeDecoder::setRandomCellFunctionData(
    SimulationData& data,
    uint8_t* genomeData,
    int nodeAddress,
    CellFunction const& cellFunction,
    bool makeSelfCopy,
    int subGenomeSize)
{
    auto newCellFunctionSize = getCellFunctionDataSize(cellFunction, makeSelfCopy, subGenomeSize);
    data.numberGen1.randomBytes(genomeData + nodeAddress, newCellFunctionSize);
    if (cellFunction == CellFunction_Constructor || cellFunction == CellFunction_Injector) {
        auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? Const::ConstructorFixedBytes : Const::InjectorFixedBytes;
        genomeData[nodeAddress + cellFunctionFixedBytes] = makeSelfCopy ? 1 : 0;
        if (!makeSelfCopy) {
            writeWord(genomeData, nodeAddress + cellFunctionFixedBytes + 1, subGenomeSize);
        }
    }
}

__inline__ __device__ int GenomeDecoder::getNumGenomeCells(uint8_t* genome, int genomeSize)
{
    int result = 0;
    int currentNodeAddress = Const::GenomeInfoSize;
    for (; result < genomeSize && currentNodeAddress < genomeSize; ++result) {
        currentNodeAddress += Const::CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeAddress);
    }

    return result;
}

__inline__ __device__ int GenomeDecoder::getNodeAddress(uint8_t* genome, int genomeSize, int nodeIndex)
{
    int currentNodeAddress = Const::GenomeInfoSize;
    for (int currentNodeIndex = 0; currentNodeIndex < nodeIndex; ++currentNodeIndex) {
        if (currentNodeAddress >= genomeSize) {
            break;
        }
        currentNodeAddress += Const::CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeAddress);
    }

    return currentNodeAddress;
}


__inline__ __device__ int GenomeDecoder::findStartNodeAddress(uint8_t* genome, int genomeSize, int refIndex)
{
    int currentNodeAddress = Const::GenomeInfoSize;
    for (; currentNodeAddress <= refIndex;) {
        auto prevCurrentNodeAddress = currentNodeAddress;
        currentNodeAddress += Const::CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeAddress);
        if (currentNodeAddress > refIndex) {
            return prevCurrentNodeAddress;
        }
    }
    return Const::GenomeInfoSize;
}

__inline__ __device__ int GenomeDecoder::getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int nodeAddress)
{
    auto cellFunction = getNextCellFunctionType(genome, nodeAddress);
    switch (cellFunction) {
    case CellFunction_Neuron:
        return Const::NeuronBytes;
    case CellFunction_Transmitter:
        return Const::TransmitterBytes;
    case CellFunction_Constructor: {
        auto isMakeCopy = GenomeDecoder::convertByteToBool(genome[nodeAddress + Const::CellBasicBytes + Const::ConstructorFixedBytes]);
        if (isMakeCopy) {
            return Const::ConstructorFixedBytes + 1;
        } else {
            return Const::ConstructorFixedBytes + 3 + getNextSubGenomeSize(genome, genomeSize, nodeAddress);
        }
    }
    case CellFunction_Sensor:
        return Const::SensorBytes;
    case CellFunction_Nerve:
        return Const::NerveBytes;
    case CellFunction_Attacker:
        return Const::AttackerBytes;
    case CellFunction_Injector: {
        auto isMakeCopy = GenomeDecoder::convertByteToBool(genome[nodeAddress + Const::CellBasicBytes + Const::InjectorFixedBytes]);
        if (isMakeCopy) {
            return Const::InjectorFixedBytes + 1;
        } else {
            return Const::InjectorFixedBytes + 3 + getNextSubGenomeSize(genome, genomeSize, nodeAddress);
        }
    }
    case CellFunction_Muscle:
        return Const::MuscleBytes;
    case CellFunction_Defender:
        return Const::DefenderBytes;
    default:
        return 0;
    }
}

__inline__ __device__ int GenomeDecoder::getNextCellFunctionType(uint8_t* genome, int nodeAddress)
{
    return genome[nodeAddress] % CellFunction_Count;
}

__inline__ __device__ bool GenomeDecoder::isNextCellSelfCopy(uint8_t* genome, int nodeAddress)
{
    switch (getNextCellFunctionType(genome, nodeAddress)) {
    case CellFunction_Constructor:
        return GenomeDecoder::convertByteToBool(genome[nodeAddress + Const::CellBasicBytes + Const::ConstructorFixedBytes]);
    case CellFunction_Injector:
        return GenomeDecoder::convertByteToBool(genome[nodeAddress + Const::CellBasicBytes + Const::InjectorFixedBytes]);
    }
    return false;
}

__inline__ __device__ int GenomeDecoder::getNextCellColor(uint8_t* genome, int nodeAddress)
{
    return genome[nodeAddress + Const::CellColorPos] % MAX_COLORS;
}

__inline__ __device__ void GenomeDecoder::setNextCellFunctionType(uint8_t* genome, int nodeAddress, CellFunction cellFunction)
{
    genome[nodeAddress] = static_cast<uint8_t>(cellFunction);
}

__inline__ __device__ void GenomeDecoder::setNextCellColor(uint8_t* genome, int nodeAddress, int color)
{
    genome[nodeAddress + Const::CellColorPos] = color;
}

__inline__ __device__ void GenomeDecoder::setNextAngle(uint8_t* genome, int nodeAddress, uint8_t angle)
{
    genome[nodeAddress + Const::CellAnglePos] = angle;
}

__inline__ __device__ void GenomeDecoder::setNextRequiredConnections(uint8_t* genome, int nodeAddress, uint8_t angle)
{
    genome[nodeAddress + Const::CellRequiredConnectionsPos] = angle;
}

__inline__ __device__ int GenomeDecoder::getNextSubGenomeSize(uint8_t* genome, int genomeSize, int nodeAddress)
{
    auto cellFunction = getNextCellFunctionType(genome, nodeAddress);
    auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? Const::ConstructorFixedBytes : Const::InjectorFixedBytes;
    auto subGenomeSizeIndex = nodeAddress + Const::CellBasicBytes + cellFunctionFixedBytes + 1;
    return max(min(GenomeDecoder::convertBytesToWord(genome[subGenomeSizeIndex], genome[subGenomeSizeIndex + 1]), genomeSize - (subGenomeSizeIndex + 2)), 0);
}

__inline__ __device__ int GenomeDecoder::getCellFunctionDataSize(CellFunction cellFunction, bool makeSelfCopy, int genomeSize)
{
    switch (cellFunction) {
    case CellFunction_Neuron:
        return Const::NeuronBytes;
    case CellFunction_Transmitter:
        return Const::TransmitterBytes;
    case CellFunction_Constructor: {
        return makeSelfCopy ? Const::ConstructorFixedBytes + 1 : Const::ConstructorFixedBytes + 3 + genomeSize;
    }
    case CellFunction_Sensor:
        return Const::SensorBytes;
    case CellFunction_Nerve:
        return Const::NerveBytes;
    case CellFunction_Attacker:
        return Const::AttackerBytes;
    case CellFunction_Injector: {
        return makeSelfCopy ? Const::InjectorFixedBytes + 1 : Const::InjectorFixedBytes + 3 + genomeSize;
    }
    case CellFunction_Muscle:
        return Const::MuscleBytes;
    case CellFunction_Defender:
        return Const::DefenderBytes;
    default:
        return 0;
    }
}

__inline__ __device__ bool GenomeDecoder::hasSelfCopy(uint8_t* genome, int genomeSize)
{
    int nodeAddress = 0;
    for (; nodeAddress < genomeSize;) {
        if (isNextCellSelfCopy(genome, nodeAddress)) {
            return true;
        }
        nodeAddress += Const::CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, nodeAddress);
    }

    return false;
}
