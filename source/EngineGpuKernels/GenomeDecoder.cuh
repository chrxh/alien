#pragma once

#include "Base.cuh"
#include "Cell.cuh"

struct GenomeInfo
{
    bool singleConstruction;
    bool separateConstruction;
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
};

class GenomeDecoder
{
public:
    __inline__ __device__ static bool isFinished(ConstructorFunction const& constructor);
    __inline__ __device__ static bool isFinishedSingleConstruction(ConstructorFunction const& constructor);
    __inline__ __device__ static bool readBool(ConstructorFunction& constructor);
    __inline__ __device__ static uint8_t readByte(ConstructorFunction& constructor);
    __inline__ __device__ static int readOptionalByte(ConstructorFunction& constructor, int moduloValue);
    __inline__ __device__ static int readWord(ConstructorFunction& constructor);
    __inline__ __device__ static float readFloat(ConstructorFunction& constructor);  //return values from -1 to 1
    __inline__ __device__ static float readEnergy(ConstructorFunction& constructor);  //return values from 36 to 1060
    __inline__ __device__ static float readAngle(ConstructorFunction& constructor);
    template <typename GenomeHolderSource, typename GenomeHolderTarget>
    __inline__ __device__ static void copyGenome(SimulationData& data, GenomeHolderSource& source, GenomeHolderTarget& target);
    __inline__ __device__ static GenomeInfo readGenomeInfo(ConstructorFunction const& constructor);

    __inline__ __device__ static bool convertByteToBool(uint8_t b);
    __inline__ __device__ static int convertBytesToWord(uint8_t b1, uint8_t b2);
    __inline__ __device__ static void convertWordToBytes(int word, uint8_t& b1, uint8_t& b2);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ bool GenomeDecoder::isFinished(ConstructorFunction const& constructor)
{
    return constructor.currentGenomePos >= constructor.genomeSize;
}

__inline__ __device__ bool GenomeDecoder::isFinishedSingleConstruction(ConstructorFunction const& constructor)
{
    auto genomeInfo = readGenomeInfo(constructor);
    return genomeInfo.singleConstruction
        && (constructor.currentGenomePos >= constructor.genomeSize || (constructor.currentGenomePos == 0 && constructor.genomeSize == Const::GenomeInfoSize));
}

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

__inline__ __device__ GenomeInfo GenomeDecoder::readGenomeInfo(ConstructorFunction const& constructor)
{
    GenomeInfo result;
    if (constructor.genomeSize < Const::GenomeInfoSize) {
        CUDA_THROW_NOT_IMPLEMENTED();
    }
    result.singleConstruction = GenomeDecoder::convertByteToBool(constructor.genome[0]);
    result.separateConstruction = GenomeDecoder::convertByteToBool(constructor.genome[1]);
    result.angleAlignment = constructor.genome[2] % ConstructorAngleAlignment_Count;
    result.stiffness = toFloat(constructor.genome[3]) / 255;
    return result;
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
