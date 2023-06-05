#pragma once

#include "EngineInterface/CellFunctionConstants.h"
#include "EngineInterface/GenomeConstants.h"

#include "CellConnectionProcessor.cuh"
#include "GenomeDecoder.cuh"

class MutationProcessor
{
public:
    __inline__ __device__ static void applyRandomMutation(SimulationData& data, Cell* cell);

    __inline__ __device__ static void neuronDataMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void propertiesMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void geometryMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void customGeometryMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void cellFunctionMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void insertMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void deleteMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void translateMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void duplicateMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void colorMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void uniformColorMutation(SimulationData& data, Cell* cell);

private:
    __inline__ __device__ static bool isRandomEvent(SimulationData& data, float probability);
    __inline__ __device__ static int getNewColorFromTransition(SimulationData& data, int origColor);
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
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationDataProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationPropertiesProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationPropertiesProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationGeometryProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationGeometryProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationGeometryProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationCustomGeometryProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationCustomGeometryProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationCustomGeometryProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationCellFunctionProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationCellFunctionProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationCellFunctionProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationInsertionProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationInsertionProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationInsertionProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationDeletionProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationDeletionProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationDeletionProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationTranslationProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationTranslationProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationTranslationProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationDuplicationProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationDuplicationProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationDuplicationProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationColorProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationColorProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationColorProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationUniformColorProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationUniformColorProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationUniformColorProbability,
        data,
        cell->absPos,
        cell->color);

    if (isRandomEvent(data, cellFunctionConstructorMutationNeuronProbability)) {
        neuronDataMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationDataProbability)) {
        propertiesMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationGeometryProbability)) {
        geometryMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationCustomGeometryProbability)) {
        customGeometryMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationCellFunctionProbability)) {
        cellFunctionMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationInsertionProbability)) {
        insertMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationDeletionProbability)) {
        deleteMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationTranslationProbability)) {
        translateMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationDuplicationProbability)) {
        duplicateMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationColorProbability)) {
        colorMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationUniformColorProbability)) {
        uniformColorMutation(data, cell);
    }
}

__inline__ __device__ void MutationProcessor::neuronDataMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeHeaderSize) {
        return;
    }
    auto nodeAddress = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, false);

    auto type = GenomeDecoder::getNextCellFunctionType(genome, nodeAddress);
    if (type == CellFunction_Neuron) {
        auto delta = data.numberGen1.random(Const::NeuronBytes - 1);
        genome[nodeAddress + Const::CellBasicBytes + delta] = data.numberGen1.randomByte();
    }
}

__inline__ __device__ void MutationProcessor::propertiesMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeHeaderSize) {
        return;
    }
    auto nodeAddress = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, false);

    //basic property mutation
    if (data.numberGen1.randomBool()) {
        auto randomDelta = data.numberGen1.random(Const::CellBasicBytes - 1);
        if (randomDelta == 0) {  //no cell function type change
            return;
        }
        if (randomDelta == Const::CellColorPos) {  //no color change
            return;
        }
        if (randomDelta == Const::CellAnglePos || randomDelta == Const::CellRequiredConnectionsPos) {  //no structure change
            return;
        }
        genome[nodeAddress + randomDelta] = data.numberGen1.randomByte();
    }

    //cell function specific mutation
    else {
        auto nextCellFunctionDataSize = GenomeDecoder::getNextCellFunctionDataSize(genome, genomeSize, nodeAddress);
        auto type = GenomeDecoder::getNextCellFunctionType(genome, nodeAddress);
        if (type == CellFunction_Constructor || type == CellFunction_Injector) {
            nextCellFunctionDataSize = type == CellFunction_Constructor ? Const::ConstructorFixedBytes : Const::InjectorFixedBytes;
        }
        if (nextCellFunctionDataSize > 0) {
            auto randomDelta = data.numberGen1.random(nextCellFunctionDataSize - 1);
            genome[nodeAddress + Const::CellBasicBytes + randomDelta] = data.numberGen1.randomByte();
        }
    }
}

__inline__ __device__ void MutationProcessor::geometryMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize < Const::GenomeHeaderSize) {
        return;
    }

    auto subgenome = genome;
    if (genomeSize > Const::GenomeHeaderSize) {
        int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH];
        int numSubGenomesSizeIndices;
        GenomeDecoder::getRandomGenomeNodeAddress(
            data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);  //return value will be discarded

        if (numSubGenomesSizeIndices > 0) {
            subgenome = genome + subGenomesSizeIndices[numSubGenomesSizeIndices - 1] + 2;  //+2 because 2 bytes encode the sub-genome length
        }
    }

    auto delta = data.numberGen1.random(Const::GenomeHeaderSize - 1);
    if (delta == Const::GenomeHeaderSeparationPos) {
        return;
    }
    subgenome[delta] = data.numberGen1.randomByte();
}

__inline__ __device__ void MutationProcessor::customGeometryMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeHeaderSize) {
        return;
    }
    auto nodeAddress = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, false);

    if (data.numberGen1.randomBool()) {
        GenomeDecoder::setNextAngle(genome, nodeAddress, data.numberGen1.randomByte());
    } else {
        GenomeDecoder::setNextRequiredConnections(genome, nodeAddress, data.numberGen1.randomByte());
    }
}

__inline__ __device__ void MutationProcessor::cellFunctionMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeHeaderSize) {
        return;
    }
    int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH];
    int numSubGenomesSizeIndices;
    auto nodeAddress = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto newCellFunction = data.numberGen1.random(CellFunction_Count - 1);
    auto makeSelfCopy = cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication ? data.numberGen1.randomBool() : false;
    if (newCellFunction == CellFunction_Injector) {      //not injection mutation allowed at the moment
        return;
    }
    if ((newCellFunction == CellFunction_Constructor || newCellFunction == CellFunction_Injector) && !makeSelfCopy) {
        if (cudaSimulationParameters.cellFunctionConstructorMutationPreventDepthIncrease
            && GenomeDecoder::getGenomeDepth(genome, genomeSize) <= numSubGenomesSizeIndices) {
            return;
        }
    }

    auto origCellFunction = GenomeDecoder::getNextCellFunctionType(genome, nodeAddress);
    if (origCellFunction == CellFunction_Constructor || origCellFunction == CellFunction_Injector) {
        if (GenomeDecoder::getNextSubGenomeSize(genome, genomeSize, nodeAddress) > Const::GenomeHeaderSize) {
            return;
        }
    }
    auto newCellFunctionSize = GenomeDecoder::getCellFunctionDataSize(newCellFunction, makeSelfCopy, Const::GenomeHeaderSize);
    auto origCellFunctionSize = GenomeDecoder::getNextCellFunctionDataSize(genome, genomeSize, nodeAddress);
    auto sizeDelta = newCellFunctionSize - origCellFunctionSize;

    if (!cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication) {
        if (GenomeDecoder::hasSelfCopy(genome + nodeAddress, Const::CellBasicBytes + origCellFunctionSize)) {
            return;
        }
    }

    auto targetGenomeSize = genomeSize + sizeDelta;
    if (targetGenomeSize > MAX_GENOME_BYTES) {
        return;
    }
    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);
    for (int i = 0; i < nodeAddress + Const::CellBasicBytes; ++i) {
        targetGenome[i] = genome[i];
    }
    GenomeDecoder::setNextCellFunctionType(targetGenome, nodeAddress, newCellFunction);
    GenomeDecoder::setRandomCellFunctionData(data, targetGenome, nodeAddress + Const::CellBasicBytes, newCellFunction, makeSelfCopy, Const::GenomeHeaderSize);

    for (int i = nodeAddress + Const::CellBasicBytes + origCellFunctionSize; i < genomeSize; ++i) {
        targetGenome[i + sizeDelta] = genome[i];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = GenomeDecoder::readWord(genome, subGenomesSizeIndices[i]);
        GenomeDecoder::writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize + sizeDelta);
    }
    if (constructor.genomeReadPosition > nodeAddress) {
        constructor.genomeReadPosition += sizeDelta;
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
}

__inline__ __device__ void MutationProcessor::insertMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);

    int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices;
    auto nodeAddress = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, true, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto newColor = cell->color;
    if (nodeAddress < genomeSize) {
        newColor = GenomeDecoder::getNextCellColor(genome, nodeAddress);
    }
    auto newCellFunction = data.numberGen1.random(CellFunction_Count - 1);
    auto makeSelfCopy = cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication ? data.numberGen1.randomBool() : false;
    if (newCellFunction == CellFunction_Injector) {  //not injection mutation allowed at the moment
        return;
    }
    if ((newCellFunction == CellFunction_Constructor || newCellFunction == CellFunction_Injector) && !makeSelfCopy) {
        if (cudaSimulationParameters.cellFunctionConstructorMutationPreventDepthIncrease
            && GenomeDecoder::getGenomeDepth(genome, genomeSize) <= numSubGenomesSizeIndices) {
            return;
        }
    }

    auto newCellFunctionSize = GenomeDecoder::getCellFunctionDataSize(newCellFunction, makeSelfCopy, Const::GenomeHeaderSize);
    auto sizeDelta = newCellFunctionSize + Const::CellBasicBytes;

    auto targetGenomeSize = genomeSize + sizeDelta;
    if (targetGenomeSize > MAX_GENOME_BYTES) {
        return;
    }
    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);
    for (int i = 0; i < nodeAddress; ++i) {
        targetGenome[i] = genome[i];
    }
    data.numberGen1.randomBytes(targetGenome + nodeAddress, Const::CellBasicBytes);
    GenomeDecoder::setNextCellFunctionType(targetGenome, nodeAddress, newCellFunction);
    GenomeDecoder::setNextCellColor(targetGenome, nodeAddress, newColor);
    GenomeDecoder::setRandomCellFunctionData(data, targetGenome, nodeAddress + Const::CellBasicBytes, newCellFunction, makeSelfCopy, Const::GenomeHeaderSize);

    for (int i = nodeAddress; i < genomeSize; ++i) {
        targetGenome[i + sizeDelta] = genome[i];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = GenomeDecoder::readWord(genome, subGenomesSizeIndices[i]);
        GenomeDecoder::writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize + sizeDelta);
    }
    if (constructor.genomeReadPosition > nodeAddress || constructor.genomeReadPosition == constructor.genomeSize) {
        constructor.genomeReadPosition += sizeDelta;
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
}

__inline__ __device__ void MutationProcessor::deleteMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeHeaderSize) {
        return;
    }

    int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH];
    int numSubGenomesSizeIndices;
    auto nodeAddress = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto origCellFunctionSize = GenomeDecoder::getNextCellFunctionDataSize(genome, genomeSize, nodeAddress);
    auto deleteSize = Const::CellBasicBytes + origCellFunctionSize;

    if (!cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication) {
        if (GenomeDecoder::hasSelfCopy(genome + nodeAddress, deleteSize)) {
            return;
        }
    }

    auto targetGenomeSize = genomeSize - deleteSize;
    for (int i = nodeAddress; i < targetGenomeSize; ++i) {
        genome[i] = genome[i + deleteSize];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = GenomeDecoder::readWord(genome, subGenomesSizeIndices[i]);
        GenomeDecoder::writeWord(genome, subGenomesSizeIndices[i], subGenomeSize - deleteSize);
    }
    if (constructor.genomeReadPosition > nodeAddress || constructor.genomeReadPosition == constructor.genomeSize) {
        constructor.genomeReadPosition -= deleteSize;
    }
    constructor.genomeSize = targetGenomeSize;
}

__inline__ __device__ void MutationProcessor::translateMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeHeaderSize) {
        return;
    }

    //calc source range
    int subGenomesSizeIndices1[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices1;
    auto startSourceIndex = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, false, subGenomesSizeIndices1, &numSubGenomesSizeIndices1);

    int subGenomeSize;
    uint8_t* subGenome;
    if (numSubGenomesSizeIndices1 > 0) {
        auto sizeIndex = subGenomesSizeIndices1[numSubGenomesSizeIndices1 - 1];
        subGenome = genome + sizeIndex + 2;  //after the 2 size bytes the subGenome starts
        subGenomeSize = GenomeDecoder::readWord(genome, sizeIndex);
    } else {
        subGenome = genome;
        subGenomeSize = genomeSize;
    }
    auto numCells = GenomeDecoder::getNumNodes(subGenome, subGenomeSize);
    auto endRelativeCellIndex = data.numberGen1.random(numCells - 1) + 1;
    auto endRelativeNodeAddress = GenomeDecoder::getNodeAddress(subGenome, subGenomeSize, endRelativeCellIndex);
    auto endSourceIndex = toInt(endRelativeNodeAddress + (subGenome - genome));
    if (endSourceIndex <= startSourceIndex) {
        return;
    }
    auto sourceRangeSize = endSourceIndex - startSourceIndex;

    //calc target insertion point
    int subGenomesSizeIndices2[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices2;
    auto startTargetIndex = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, true, subGenomesSizeIndices2, &numSubGenomesSizeIndices2);

    if (startTargetIndex >= startSourceIndex && startTargetIndex <= endSourceIndex) {
        return;
    }
    if (cudaSimulationParameters.cellFunctionConstructorMutationPreventDepthIncrease) {
        auto genomeDepth = GenomeDecoder::getGenomeDepth(genome, genomeSize);
        auto sourceRangeDepth = GenomeDecoder::getGenomeDepth(subGenome, subGenomeSize);
        if (genomeDepth < sourceRangeDepth + numSubGenomesSizeIndices2) {
            return;
        }
    }

    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(genomeSize);
    if (startTargetIndex > endSourceIndex) {

        //copy genome
        for (int i = 0; i < startSourceIndex; ++i) {
            targetGenome[i] = genome[i];
        }
        int delta1 = startTargetIndex - endSourceIndex;
        for (int i = 0; i < delta1; ++i) {
            targetGenome[startSourceIndex + i] = genome[endSourceIndex + i];
        }
        int delta2 = sourceRangeSize;
        for (int i = 0; i < delta2; ++i) {
            targetGenome[startSourceIndex + delta1 + i] = genome[startSourceIndex + i];
        }
        int delta3 = genomeSize - startTargetIndex;
        for (int i = 0; i < delta3; ++i) {
            targetGenome[startSourceIndex + delta1 + delta2 + i] = genome[startTargetIndex + i];
        }

        if (constructor.genomeReadPosition >= startSourceIndex && constructor.genomeReadPosition <= startTargetIndex) {
            constructor.genomeReadPosition = 0;
        }

        //adjust sub genome size fields
        for (int i = 0; i < numSubGenomesSizeIndices1; ++i) {
            auto subGenomeSize = GenomeDecoder::readWord(targetGenome, subGenomesSizeIndices1[i]);
            GenomeDecoder::writeWord(targetGenome, subGenomesSizeIndices1[i], subGenomeSize - sourceRangeSize);
        }
        for (int i = 0; i < numSubGenomesSizeIndices2; ++i) {
            auto address = subGenomesSizeIndices2[i];
            if (address >= startSourceIndex) {
                address -= sourceRangeSize;
            }
            auto subGenomeSize = GenomeDecoder::readWord(targetGenome, address);
            GenomeDecoder::writeWord(targetGenome, address, subGenomeSize + sourceRangeSize);
        }

    } else {

        //copy genome
        for (int i = 0; i < startTargetIndex; ++i) {
            targetGenome[i] = genome[i];
        }
        int delta1 = sourceRangeSize;
        for (int i = 0; i < delta1; ++i) {
            targetGenome[startTargetIndex + i] = genome[startSourceIndex + i];
        }
        int delta2 = startSourceIndex - startTargetIndex;
        for (int i = 0; i < delta2; ++i) {
            targetGenome[startTargetIndex + delta1 + i] = genome[startTargetIndex + i];
        }
        int delta3 = genomeSize - endSourceIndex;
        for (int i = 0; i < delta3; ++i) {
            targetGenome[startTargetIndex + delta1 + delta2 + i] = genome[endSourceIndex + i];
        }
        if (constructor.genomeReadPosition >= startTargetIndex && constructor.genomeReadPosition <= endSourceIndex) {
            constructor.genomeReadPosition = 0;
        }

        //adjust sub genome size fields
        for (int i = 0; i < numSubGenomesSizeIndices1; ++i) {
            auto address = subGenomesSizeIndices1[i];
            if (address >= startTargetIndex) {
                address += sourceRangeSize;
            }
            auto subGenomeSize = GenomeDecoder::readWord(targetGenome, address);
            GenomeDecoder::writeWord(targetGenome, address, subGenomeSize - sourceRangeSize);
        }
        for (int i = 0; i < numSubGenomesSizeIndices2; ++i) {
            auto subGenomeSize = GenomeDecoder::readWord(targetGenome, subGenomesSizeIndices2[i]);
            GenomeDecoder::writeWord(targetGenome, subGenomesSizeIndices2[i], subGenomeSize + sourceRangeSize);
        }
    }

    constructor.genome = targetGenome;
}

__inline__ __device__ void MutationProcessor::duplicateMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeHeaderSize) {
        return;
    }

    int startSourceIndex;
    int endSourceIndex;
    int subGenomeSize;
    uint8_t* subGenome;
    {
        int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH + 1];
        int numSubGenomesSizeIndices;
        startSourceIndex = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);

        if (numSubGenomesSizeIndices > 0) {
            auto sizeIndex = subGenomesSizeIndices[numSubGenomesSizeIndices - 1];
            subGenome = genome + sizeIndex + 2;  //after the 2 size bytes the subGenome starts
            subGenomeSize = GenomeDecoder::readWord(genome, sizeIndex);
        } else {
            subGenome = genome;
            subGenomeSize = genomeSize;
        }
        auto numCells = GenomeDecoder::getNumNodes(subGenome, subGenomeSize);
        auto endRelativeCellIndex = data.numberGen1.random(numCells - 1) + 1;
        auto endRelativeNodeAddress = GenomeDecoder::getNodeAddress(subGenome, subGenomeSize, endRelativeCellIndex);
        endSourceIndex = toInt(endRelativeNodeAddress + (subGenome - genome));
        if (endSourceIndex <= startSourceIndex) {
            return;
        }
    }
    auto sizeDelta = endSourceIndex - startSourceIndex;
    if (!cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication) {
        if (GenomeDecoder::hasSelfCopy(genome + startSourceIndex, sizeDelta)) {
            return;
        }
    }

    int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices;
    auto startTargetIndex = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, true, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto targetGenomeSize = genomeSize + sizeDelta;
    if (targetGenomeSize > MAX_GENOME_BYTES) {
        return;
    }

    if (cudaSimulationParameters.cellFunctionConstructorMutationPreventDepthIncrease) {
        auto genomeDepth = GenomeDecoder::getGenomeDepth(genome, genomeSize);
        auto sourceRangeDepth = GenomeDecoder::getGenomeDepth(subGenome, subGenomeSize);
        if (genomeDepth < sourceRangeDepth + numSubGenomesSizeIndices) {
            return;
        }
    }

    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);
    for (int i = 0; i < startTargetIndex; ++i) {
        targetGenome[i] = genome[i];
    }
    for (int i = 0; i < sizeDelta; ++i) {
        targetGenome[startTargetIndex + i] = genome[startSourceIndex + i];
    }
    for (int i = 0; i < genomeSize - startTargetIndex; ++i) {
        targetGenome[startTargetIndex + sizeDelta + i] = genome[startTargetIndex + i];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = GenomeDecoder::readWord(targetGenome, subGenomesSizeIndices[i]);
        GenomeDecoder::writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize + sizeDelta);
    }
    if (constructor.genomeReadPosition > startTargetIndex || constructor.genomeReadPosition == constructor.genomeSize) {
        constructor.genomeReadPosition += sizeDelta;
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
}

__inline__ __device__ void MutationProcessor::colorMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeHeaderSize) {
        return;
    }

    int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH];
    int numSubGenomesSizeIndices;
    GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);  //return value will be discarded

    int nodeAddress = Const::GenomeHeaderSize;
    auto subgenome = genome;
    int subgenomeSize = genomeSize;
    if (numSubGenomesSizeIndices > 0) {
        subgenome = genome + subGenomesSizeIndices[numSubGenomesSizeIndices - 1] + 2;  //+2 because 2 bytes encode the sub-genome length
        subgenomeSize = GenomeDecoder::readWord(genome, subGenomesSizeIndices[numSubGenomesSizeIndices - 1]);
    }

    auto origColor = GenomeDecoder::getNextCellColor(subgenome, nodeAddress);
    auto newColor = getNewColorFromTransition(data, origColor);

    for (int dummy = 0; nodeAddress < subgenomeSize && dummy < subgenomeSize; ++dummy) {
        GenomeDecoder::setNextCellColor(subgenome, nodeAddress, newColor);
        nodeAddress += Const::CellBasicBytes + GenomeDecoder::getNextCellFunctionDataSize(subgenome, subgenomeSize, nodeAddress);
    }
}

__inline__ __device__ void MutationProcessor::uniformColorMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeHeaderSize) {
        return;
    }

    auto origColor = GenomeDecoder::getNextCellColor(genome, Const::GenomeHeaderSize);
    auto newColor = getNewColorFromTransition(data, origColor);

    GenomeDecoder::executeForEachNodeRecursively(
        genome, genomeSize, [&](int depth, int nodeAddress) { GenomeDecoder::setNextCellColor(genome, nodeAddress, newColor); });
}

__inline__ __device__ bool MutationProcessor::isRandomEvent(SimulationData& data, float probability)
{
    if (probability > 0.001f) {
        return data.numberGen1.random() < probability;
    } else {
        return data.numberGen1.random() < probability * 1000 && data.numberGen2.random() < 0.001f;
    }
}

__inline__ __device__ int MutationProcessor::getNewColorFromTransition(SimulationData& data, int origColor)
{
    int numAllowedColors = 0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        if (cudaSimulationParameters.cellFunctionConstructorMutationColorTransitions[origColor][i]) {
            ++numAllowedColors;
        }
    }
    if (numAllowedColors == 0) {
        return;
    }
    int randomAllowedColorIndex = data.numberGen1.random(numAllowedColors - 1);
    int allowedColorIndex = 0;
    int result = 0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        if (cudaSimulationParameters.cellFunctionConstructorMutationColorTransitions[origColor][i]) {
            if (allowedColorIndex == randomAllowedColorIndex) {
                result = i;
                break;
            }
            ++allowedColorIndex;
        }
    }
    return result;
}

