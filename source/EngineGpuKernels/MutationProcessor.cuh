#pragma once

#include "EngineInterface/CellFunctionEnums.h"
#include "EngineInterface/GenomeConstants.h"

#include "CellConnectionProcessor.cuh"
#include "GenomeDecoder.cuh"

class MutationProcessor
{
public:
    __inline__ __device__ static void applyRandomMutation(SimulationData& data, Cell* cell);

    __inline__ __device__ static void neuronDataMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void propertiesMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void structureMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void cellFunctionMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void insertMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void deleteMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void translateMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void duplicateMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void colorMutation(SimulationData& data, Cell* cell);

private:
    static auto constexpr MAX_SUBGENOME_RECURSION_DEPTH = 30;
    __inline__ __device__ static bool isRandomEvent(SimulationData& data, float probability);

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
    __inline__ __device__ static int getNumGenomeCells(uint8_t* genome, int genomeSize);
    __inline__ __device__ static int getNodeIndex(uint8_t* genome, int genomeSize, int cellIndex);
    __inline__ __device__
        static int findStartNodeIndex(uint8_t* genome, int genomeSize, int refIndex);
    __inline__ __device__ static int getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int nodeIndex);
    __inline__ __device__ static int getNextCellFunctionType(uint8_t* genome, int nodeIndex);
    __inline__ __device__ static bool isNextCellSelfCopy(uint8_t* genome, int nodeIndex);
    __inline__ __device__ static int getNextCellColor(uint8_t* genome, int nodeIndex);
    __inline__ __device__ static void setNextCellFunctionType(uint8_t* genome, int nodeIndex, CellFunction cellFunction);
    __inline__ __device__ static void setNextCellColor(uint8_t* genome, int nodeIndex, int color);
    __inline__ __device__ static void setNextAngle(uint8_t* genome, int nodeIndex, uint8_t angle);
    __inline__ __device__ static void setNextRequiredConnections(uint8_t* genome, int nodeIndex, uint8_t angle);
    __inline__ __device__
        static int getNextSubGenomeSize(uint8_t* genome, int genomeSize, int nodeIndex);  //prerequisites: (constructor or injector) and !makeSelfCopy
    __inline__ __device__
        static int getCellFunctionDataSize(
        CellFunction cellFunction,
        bool makeSelfCopy,
        int genomeSize);  //genomeSize only relevant for cellFunction = constructor or injector
    __inline__ __device__ static bool hasSelfCopy(uint8_t* genome, int genomeSize);

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
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationDataProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationPropertiesProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationPropertiesProbability,
        data,
        cell->absPos,
        cell->color);
    auto cellFunctionConstructorMutationStructureProbability = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionConstructorMutationStructureProbability,
        &SimulationParametersSpotActivatedValues::cellFunctionConstructorMutationStructureProbability,
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

    if (isRandomEvent(data, cellFunctionConstructorMutationNeuronProbability)) {
        neuronDataMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationDataProbability)) {
        propertiesMutation(data, cell);
    }
    if (isRandomEvent(data, cellFunctionConstructorMutationStructureProbability)) {
        structureMutation(data, cell);
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
}

__inline__ __device__ void MutationProcessor::neuronDataMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeInfoSize) {
        return;
    }
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false);

    auto type = getNextCellFunctionType(genome, nodeIndex);
    if (type == CellFunction_Neuron) {
        auto delta = data.numberGen1.random(Const::NeuronBytes - 1);
        genome[nodeIndex + Const::CellBasicBytes + delta] = data.numberGen1.randomByte();
    }
}

__inline__ __device__ void MutationProcessor::propertiesMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeInfoSize) {
        return;
    }
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false);

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
        genome[nodeIndex + randomDelta] = data.numberGen1.randomByte();
    }

    //cell function specific mutation
    else {
        auto nextCellFunctionDataSize = getNextCellFunctionDataSize(genome, genomeSize, nodeIndex);
        auto type = getNextCellFunctionType(genome, nodeIndex);
        if (type == CellFunction_Constructor || type == CellFunction_Injector) {
            nextCellFunctionDataSize = type == CellFunction_Constructor ? Const::ConstructorFixedBytes : Const::InjectorFixedBytes;
        }
        if (nextCellFunctionDataSize > 0) {
            auto randomDelta = data.numberGen1.random(nextCellFunctionDataSize - 1);
            genome[nodeIndex + Const::CellBasicBytes + randomDelta] = data.numberGen1.randomByte();
        }
    }
}

__inline__ __device__ void MutationProcessor::structureMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeInfoSize) {
        return;
    }
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false);

    if (data.numberGen1.randomBool()) {
        setNextAngle(genome, nodeIndex, data.numberGen1.randomByte());
    } else {
        setNextRequiredConnections(genome, nodeIndex, data.numberGen1.randomByte());
    }
}

__inline__ __device__ void MutationProcessor::cellFunctionMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeInfoSize) {
        return;
    }
    int subGenomesSizeIndices[MAX_SUBGENOME_RECURSION_DEPTH];
    int numSubGenomesSizeIndices;
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto newCellFunction = data.numberGen1.random(CellFunction_Count - 1);
    auto makeSelfCopy = cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication ? data.numberGen1.randomBool() : false;

    auto origCellFunction = getNextCellFunctionType(genome, nodeIndex);
    if (origCellFunction == CellFunction_Constructor || origCellFunction == CellFunction_Injector) {
        if (getNextSubGenomeSize(genome, genomeSize, nodeIndex) > Const::GenomeInfoSize) {
            return;
        }
    }
    auto newCellFunctionSize = getCellFunctionDataSize(newCellFunction, makeSelfCopy, Const::GenomeInfoSize);
    auto origCellFunctionSize = getNextCellFunctionDataSize(genome, genomeSize, nodeIndex);
    auto sizeDelta = newCellFunctionSize - origCellFunctionSize;

    if (!cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication) {
        if (hasSelfCopy(genome + nodeIndex, Const::CellBasicBytes + origCellFunctionSize)) {
            return;
        }
    }

    auto targetGenomeSize = genomeSize + sizeDelta;
    if (targetGenomeSize > MAX_GENOME_BYTES) {
        return;
    }
    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);
    for (int i = 0; i < nodeIndex + Const::CellBasicBytes; ++i) {
        targetGenome[i] = genome[i];
    }
    setNextCellFunctionType(targetGenome, nodeIndex, newCellFunction);
    setRandomCellFunctionData(data, targetGenome, nodeIndex + Const::CellBasicBytes, newCellFunction, makeSelfCopy, Const::GenomeInfoSize);

    for (int i = nodeIndex + Const::CellBasicBytes + origCellFunctionSize; i < genomeSize; ++i) {
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

__inline__ __device__ void MutationProcessor::insertMutation(SimulationData& data, Cell* cell)
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
    auto makeSelfCopy = cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication ? data.numberGen1.randomBool() : false;

    auto newCellFunctionSize = getCellFunctionDataSize(newCellFunction, makeSelfCopy, Const::GenomeInfoSize);
    auto sizeDelta = newCellFunctionSize + Const::CellBasicBytes;

    auto targetGenomeSize = genomeSize + sizeDelta;
    if (targetGenomeSize > MAX_GENOME_BYTES) {
        return;
    }
    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);
    for (int i = 0; i < nodeIndex; ++i) {
        targetGenome[i] = genome[i];
    }
    data.numberGen1.randomBytes(targetGenome + nodeIndex, Const::CellBasicBytes);
    setNextCellFunctionType(targetGenome, nodeIndex, newCellFunction);
    setNextCellColor(targetGenome, nodeIndex, newColor);
    setRandomCellFunctionData(data, targetGenome, nodeIndex + Const::CellBasicBytes, newCellFunction, makeSelfCopy, Const::GenomeInfoSize);

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

__inline__ __device__ void MutationProcessor::deleteMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeInfoSize) {
        return;
    }

    int subGenomesSizeIndices[MAX_SUBGENOME_RECURSION_DEPTH];
    int numSubGenomesSizeIndices;
    auto nodeIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto origCellFunctionSize = getNextCellFunctionDataSize(genome, genomeSize, nodeIndex);
    auto deleteSize = Const::CellBasicBytes + origCellFunctionSize;

    if (!cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication) {
        if (hasSelfCopy(genome + nodeIndex, deleteSize)) {
            return;
        }
    }

    auto targetGenomeSize = genomeSize - deleteSize;
    for (int i = nodeIndex; i < targetGenomeSize; ++i) {
        genome[i] = genome[i + deleteSize];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = readWord(genome, subGenomesSizeIndices[i]);
        writeWord(genome, subGenomesSizeIndices[i], subGenomeSize - deleteSize);
    }
    if (constructor.currentGenomePos > nodeIndex || constructor.currentGenomePos == constructor.genomeSize) {
        constructor.currentGenomePos -= deleteSize;
    }
    constructor.genomeSize = targetGenomeSize;
}

__inline__ __device__ void MutationProcessor::translateMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeInfoSize) {
        return;
    }

    //calc source range
    int subGenomesSizeIndices1[MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices1;
    auto startSourceIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false, subGenomesSizeIndices1, &numSubGenomesSizeIndices1);

    int subGenomeSize;
    uint8_t* subGenome;
    if (numSubGenomesSizeIndices1 > 0) {
        auto sizeIndex = subGenomesSizeIndices1[numSubGenomesSizeIndices1 - 1];
        subGenome = genome + sizeIndex + 2;  //after the 2 size bytes the subGenome starts
        subGenomeSize = readWord(genome, sizeIndex);
    } else {
        subGenome = genome;
        subGenomeSize = genomeSize;
    }
    auto numCells = getNumGenomeCells(subGenome, subGenomeSize);
    auto endRelativeCellIndex = data.numberGen1.random(numCells - 1) + 1;
    auto endRelativeNodeIndex = getNodeIndex(subGenome, subGenomeSize, endRelativeCellIndex);
    auto endSourceIndex = toInt(endRelativeNodeIndex + (subGenome - genome));
    if (endSourceIndex <= startSourceIndex) {
        return;
    }
    auto sourceRangeSize = endSourceIndex - startSourceIndex;

    //calc target insertion point
    int subGenomesSizeIndices2[MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices2;
    auto startTargetIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, true, subGenomesSizeIndices2, &numSubGenomesSizeIndices2);

    if (startTargetIndex >= startSourceIndex && startTargetIndex <= endSourceIndex) {
        return;
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

        if (constructor.currentGenomePos >= startSourceIndex && constructor.currentGenomePos <= startTargetIndex) {
            constructor.currentGenomePos = 0;
        }

        //adjust sub genome size fields
        for (int i = 0; i < numSubGenomesSizeIndices1; ++i) {
            auto subGenomeSize = readWord(targetGenome, subGenomesSizeIndices1[i]);
            writeWord(targetGenome, subGenomesSizeIndices1[i], subGenomeSize - sourceRangeSize);
        }
        for (int i = 0; i < numSubGenomesSizeIndices2; ++i) {
            auto address = subGenomesSizeIndices2[i];
            if (address >= startSourceIndex) {
                address -= sourceRangeSize;
            }
            auto subGenomeSize = readWord(targetGenome, address);
            writeWord(targetGenome, address, subGenomeSize + sourceRangeSize);
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
        if (constructor.currentGenomePos >= startTargetIndex && constructor.currentGenomePos <= endSourceIndex) {
            constructor.currentGenomePos = 0;
        }

        //adjust sub genome size fields
        for (int i = 0; i < numSubGenomesSizeIndices1; ++i) {
            auto address = subGenomesSizeIndices1[i];
            if (address >= startTargetIndex) {
                address += sourceRangeSize;
            }
            auto subGenomeSize = readWord(targetGenome, address);
            writeWord(targetGenome, address, subGenomeSize - sourceRangeSize);
        }
        for (int i = 0; i < numSubGenomesSizeIndices2; ++i) {
            auto subGenomeSize = readWord(targetGenome, subGenomesSizeIndices2[i]);
            writeWord(targetGenome, subGenomesSizeIndices2[i], subGenomeSize + sourceRangeSize);
        }
    }

    constructor.genome = targetGenome;
}

__inline__ __device__ void MutationProcessor::duplicateMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeInfoSize) {
        return;
    }

    int startSourceIndex;
    int endSourceIndex;
    {
        int subGenomesSizeIndices[MAX_SUBGENOME_RECURSION_DEPTH + 1];
        int numSubGenomesSizeIndices;
        startSourceIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);

        int subGenomeSize;
        uint8_t* subGenome;
        if (numSubGenomesSizeIndices > 0) {
            auto sizeIndex = subGenomesSizeIndices[numSubGenomesSizeIndices - 1];
            subGenome = genome + sizeIndex + 2;  //after the 2 size bytes the subGenome starts
            subGenomeSize = readWord(genome, sizeIndex);
        } else {
            subGenome = genome;
            subGenomeSize = genomeSize;
        }
        auto numCells = getNumGenomeCells(subGenome, subGenomeSize);
        auto endRelativeCellIndex = data.numberGen1.random(numCells - 1) + 1;
        auto endRelativeNodeIndex = getNodeIndex(subGenome, subGenomeSize, endRelativeCellIndex);
        endSourceIndex = toInt(endRelativeNodeIndex + (subGenome - genome));
        if (endSourceIndex <= startSourceIndex) {
            return;
        }
    }
    auto sizeDelta = endSourceIndex - startSourceIndex;
    if (!cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication) {
        if (hasSelfCopy(genome + startSourceIndex, sizeDelta)) {
            return;
        }
    }

    int subGenomesSizeIndices[MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices;
    auto startTargetIndex = getRandomGenomeNodeIndex(data, genome, genomeSize, true, subGenomesSizeIndices, &numSubGenomesSizeIndices);


    auto targetGenomeSize = genomeSize + sizeDelta;
    if (targetGenomeSize > MAX_GENOME_BYTES) {
        return;
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
        auto subGenomeSize = readWord(targetGenome, subGenomesSizeIndices[i]);
        writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize + sizeDelta);
    }
    if (constructor.currentGenomePos > startTargetIndex || constructor.currentGenomePos == constructor.genomeSize) {
        constructor.currentGenomePos += sizeDelta;
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
}

__inline__ __device__ void MutationProcessor::colorMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto genomeSize = toInt(constructor.genomeSize);
    if (genomeSize <= Const::GenomeInfoSize) {
        return;
    }

    int subGenomesSizeIndices[MAX_SUBGENOME_RECURSION_DEPTH];
    int numSubGenomesSizeIndices;
    getRandomGenomeNodeIndex(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);  //return value will be discarded

    int nodeIndex = Const::GenomeInfoSize;
    auto subgenome = genome;
    int subgenomeSize = genomeSize;
    if (numSubGenomesSizeIndices > 0) {
        subgenome = genome + subGenomesSizeIndices[numSubGenomesSizeIndices - 1] + 2;  //+2 because 2 bytes encode the sub-genome length
        subgenomeSize = readWord(genome, subGenomesSizeIndices[numSubGenomesSizeIndices - 1]);
    }

    auto origColor = getNextCellColor(subgenome, nodeIndex);
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
    int newColor = 0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        if (cudaSimulationParameters.cellFunctionConstructorMutationColorTransitions[origColor][i]) {
            if (allowedColorIndex == randomAllowedColorIndex) {
                newColor = i;
                break;
            }
            ++allowedColorIndex;
        }
    }

    for (int dummy = 0; nodeIndex < subgenomeSize && dummy < subgenomeSize; ++dummy) {
        setNextCellColor(subgenome, nodeIndex, newColor);
        nodeIndex += Const::CellBasicBytes + getNextCellFunctionDataSize(subgenome, subgenomeSize, nodeIndex);
    }
}

__inline__ __device__ bool MutationProcessor::isRandomEvent(SimulationData& data, float probability)
{
    if (probability > 0.001f) {
        return data.numberGen1.random() < probability;
    } else {
        return data.numberGen1.random() < probability * 1000 && data.numberGen2.random() < 0.001f;
    }
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
            auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? Const::ConstructorFixedBytes : Const::InjectorFixedBytes;
            auto makeSelfCopy = GenomeDecoder::convertByteToBool(genome[nodeIndex + Const::CellBasicBytes + cellFunctionFixedBytes]);
            if (makeSelfCopy) {
                break;
            } else {
                if (numSubGenomesSizeIndices) {
                    subGenomesSizeIndices[*numSubGenomesSizeIndices] = result + Const::CellBasicBytes + cellFunctionFixedBytes + 1;
                    ++(*numSubGenomesSizeIndices);
                }
                auto subGenomeStartIndex = nodeIndex + Const::CellBasicBytes + cellFunctionFixedBytes + 3;
                auto subGenomeSize = getNextSubGenomeSize(genome, genomeSize, nodeIndex);
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
            cellFunction == CellFunction_Constructor ? Const::ConstructorFixedBytes : Const::InjectorFixedBytes;
        genomeData[byteIndex + cellFunctionFixedBytes] = makeSelfCopy ? 1 : 0;
        if (!makeSelfCopy) {
            writeWord(genomeData, byteIndex + cellFunctionFixedBytes + 1, subGenomeSize);
        }
    }
}

__inline__ __device__ int MutationProcessor::getNumGenomeCells(uint8_t* genome, int genomeSize)
{
    int result = 0;
    int currentNodeIndex = Const::GenomeInfoSize;
    for (; result < genomeSize && currentNodeIndex < genomeSize; ++result) {
        currentNodeIndex += Const::CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeIndex);
    }

    return result;
}

__inline__ __device__ int MutationProcessor::getNodeIndex(uint8_t* genome, int genomeSize, int cellIndex)
{
    int currentNodeIndex = Const::GenomeInfoSize;
    for (int currentCellIndex = 0; currentCellIndex < cellIndex; ++currentCellIndex) {
        if (currentNodeIndex >= genomeSize) {
            break;
        }
        currentNodeIndex += Const::CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeIndex);
    }

    return currentNodeIndex;
}


__inline__ __device__ int MutationProcessor::findStartNodeIndex(uint8_t* genome, int genomeSize, int refIndex)
{
    int currentNodeIndex = Const::GenomeInfoSize;
    for (; currentNodeIndex <= refIndex;) {
        auto prevCurrentNodeIndex = currentNodeIndex;
        currentNodeIndex += Const::CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, currentNodeIndex);
        if (currentNodeIndex > refIndex) {
            return prevCurrentNodeIndex;
        }
    }
    return Const::GenomeInfoSize;
}

__inline__ __device__ int MutationProcessor::getNextCellFunctionDataSize(uint8_t* genome, int genomeSize, int nodeIndex)
{
    auto cellFunction = getNextCellFunctionType(genome, nodeIndex);
    switch (cellFunction) {
    case CellFunction_Neuron:
        return Const::NeuronBytes;
    case CellFunction_Transmitter:
        return Const::TransmitterBytes;
    case CellFunction_Constructor: {
        auto isMakeCopy = GenomeDecoder::convertByteToBool(genome[nodeIndex + Const::CellBasicBytes + Const::ConstructorFixedBytes]);
        if (isMakeCopy) {
            return Const::ConstructorFixedBytes + 1;
        } else {
            return Const::ConstructorFixedBytes + 3 + getNextSubGenomeSize(genome, genomeSize, nodeIndex);
        }
    }
    case CellFunction_Sensor:
        return Const::SensorBytes;
    case CellFunction_Nerve:
        return Const::NerveBytes;
    case CellFunction_Attacker:
        return Const::AttackerBytes;
    case CellFunction_Injector: {
        auto isMakeCopy = GenomeDecoder::convertByteToBool(genome[nodeIndex + Const::CellBasicBytes + Const::InjectorFixedBytes]);
        if (isMakeCopy) {
            return Const::InjectorFixedBytes + 1;
        } else {
            return Const::InjectorFixedBytes + 3 + getNextSubGenomeSize(genome, genomeSize, nodeIndex);
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

__inline__ __device__ int MutationProcessor::getNextCellFunctionType(uint8_t* genome, int nodeIndex)
{
    return genome[nodeIndex] % CellFunction_Count;
}

__inline__ __device__ bool MutationProcessor::isNextCellSelfCopy(uint8_t* genome, int nodeIndex)
{
    switch (getNextCellFunctionType(genome, nodeIndex)) {
    case CellFunction_Constructor:
        return GenomeDecoder::convertByteToBool(genome[nodeIndex + Const::CellBasicBytes + Const::ConstructorFixedBytes]);
    case CellFunction_Injector:
        return GenomeDecoder::convertByteToBool(genome[nodeIndex + Const::CellBasicBytes + Const::InjectorFixedBytes]);
    }
    return false;
}

__inline__ __device__ int MutationProcessor::getNextCellColor(uint8_t* genome, int nodeIndex)
{
    return genome[nodeIndex + Const::CellColorPos] % MAX_COLORS;
}

__inline__ __device__ void MutationProcessor::setNextCellFunctionType(uint8_t* genome, int nodeIndex, CellFunction cellFunction)
{
    genome[nodeIndex] = static_cast<uint8_t>(cellFunction);
}

__inline__ __device__ void MutationProcessor::setNextCellColor(uint8_t* genome, int nodeIndex, int color)
{
    genome[nodeIndex + Const::CellColorPos] = color;
}

__inline__ __device__ void MutationProcessor::setNextAngle(uint8_t* genome, int nodeIndex, uint8_t angle)
{
    genome[nodeIndex + Const::CellAnglePos] = angle;
}

__inline__ __device__ void MutationProcessor::setNextRequiredConnections(uint8_t* genome, int nodeIndex, uint8_t angle)
{
    genome[nodeIndex + Const::CellRequiredConnectionsPos] = angle;
}

__inline__ __device__ int MutationProcessor::getNextSubGenomeSize(uint8_t* genome, int genomeSize, int nodeIndex)
{
    auto cellFunction = getNextCellFunctionType(genome, nodeIndex);
    auto cellFunctionFixedBytes = cellFunction == CellFunction_Constructor ? Const::ConstructorFixedBytes : Const::InjectorFixedBytes;
    auto subGenomeSizeIndex = nodeIndex + Const::CellBasicBytes + cellFunctionFixedBytes + 1;
    return max(
        min(GenomeDecoder::convertBytesToWord(genome[subGenomeSizeIndex], genome[subGenomeSizeIndex + 1]),
            genomeSize - (subGenomeSizeIndex + 2)),
        0);
}

__inline__ __device__ int MutationProcessor::getCellFunctionDataSize(CellFunction cellFunction, bool makeSelfCopy, int genomeSize)
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

__inline__ __device__ bool MutationProcessor::hasSelfCopy(uint8_t* genome, int genomeSize)
{
    int nodeIndex = 0;
    for (; nodeIndex < genomeSize; ) {
        if (isNextCellSelfCopy(genome, nodeIndex)) {
            return true;
        }
        nodeIndex += Const::CellBasicBytes + getNextCellFunctionDataSize(genome, genomeSize, nodeIndex);
    }

    return false;
}

__inline__ __device__ int MutationProcessor::readWord(uint8_t* genome, int byteIndex)
{
    return GenomeDecoder::convertBytesToWord(genome[byteIndex], genome[byteIndex + 1]);
}

__inline__ __device__ void MutationProcessor::writeWord(uint8_t* genome, int byteIndex, int word)
{
    GenomeDecoder::convertWordToBytes(word, genome[byteIndex], genome[byteIndex + 1]);
}
