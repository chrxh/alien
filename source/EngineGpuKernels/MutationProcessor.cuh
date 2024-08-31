#pragma once

#include "EngineInterface/CellFunctionConstants.h"
#include "EngineInterface/GenomeConstants.h"

#include "CellConnectionProcessor.cuh"
#include "GenomeDecoder.cuh"
#include "CudaShapeGenerator.cuh"

class MutationProcessor
{
public:

    __inline__ __device__ static void applyRandomMutations(SimulationData& data);
    __inline__ __device__ static void applyRandomMutationsForCell(SimulationData& data, Cell* cell);

    __inline__ __device__ static void neuronDataMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void propertiesMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void geometryMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void customGeometryMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void cellFunctionMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void insertMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void deleteMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void translateMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void duplicateMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void cellColorMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void subgenomeColorMutation(SimulationData& data, Cell* cell);
    __inline__ __device__ static void genomeColorMutation(SimulationData& data, Cell* cell);

private:
    template <typename Func>
    __inline__ __device__ static void executeEvent(SimulationData& data, float probability, Func eventFunc);
    template <typename Func>
    __inline__ __device__ static void executeMultipleEvents(SimulationData& data, float probability, Func eventFunc);
    __inline__ __device__ static void adaptMutationId(SimulationData& data, ConstructorFunction& constructor);
    __inline__ __device__ static bool isRandomEvent(SimulationData& data, float probability);
    __inline__ __device__ static int getNewColorFromTransition(SimulationData& data, int origColor);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void MutationProcessor::applyRandomMutations(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->livingState== LivingState_Activating && cell->cellFunction == CellFunction_Constructor) {
            MutationProcessor::applyRandomMutationsForCell(data, cell);
        }
    }
}

__inline__ __device__ void MutationProcessor::applyRandomMutationsForCell(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto numNodes = toFloat(GenomeDecoder::getNumNodesRecursively(constructor.genome, constructor.genomeSize, false, true));
    auto cellCopyMutationNeuronData = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationNeuronData,
        &SimulationParametersSpotActivatedValues::cellCopyMutationNeuronData,
        data,
        cell->pos,
        cell->color) * numNodes;
    auto cellCopyMutationCellProperties = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationCellProperties,
        &SimulationParametersSpotActivatedValues::cellCopyMutationCellProperties,
        data,
        cell->pos,
        cell->color) * numNodes;
    auto cellCopyMutationGeometry = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationGeometry,
        &SimulationParametersSpotActivatedValues::cellCopyMutationGeometry,
        data,
        cell->pos,
        cell->color) * numNodes;
    auto cellCopyMutationCustomGeometry = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationCustomGeometry,
        &SimulationParametersSpotActivatedValues::cellCopyMutationCustomGeometry,
        data,
        cell->pos,
        cell->color) * numNodes;
    auto cellCopyMutationCellFunction = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationCellFunction,
        &SimulationParametersSpotActivatedValues::cellCopyMutationCellFunction,
        data,
        cell->pos,
        cell->color) * numNodes;
    auto cellCopyMutationInsertion = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationInsertion,
        &SimulationParametersSpotActivatedValues::cellCopyMutationInsertion,
        data,
        cell->pos,
        cell->color) * numNodes;
    auto cellCopyMutationDeletion = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationDeletion,
        &SimulationParametersSpotActivatedValues::cellCopyMutationDeletion,
        data,
        cell->pos,
        cell->color) * numNodes;
    auto cellCopyMutationCellColor = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationCellColor,
        &SimulationParametersSpotActivatedValues::cellCopyMutationCellColor,
        data,
        cell->pos,
        cell->color) * numNodes;
    auto cellCopyMutationTranslation = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationTranslation,
        &SimulationParametersSpotActivatedValues::cellCopyMutationTranslation,
        data,
        cell->pos,
        cell->color);
    auto cellCopyMutationDuplication = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationDuplication,
        &SimulationParametersSpotActivatedValues::cellCopyMutationDuplication,
        data,
        cell->pos,
        cell->color);
    auto cellCopyMutationSubgenomeColor = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationSubgenomeColor,
        &SimulationParametersSpotActivatedValues::cellCopyMutationSubgenomeColor,
        data,
        cell->pos,
        cell->color);
    auto cellCopyMutationGenomeColor = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellCopyMutationGenomeColor,
        &SimulationParametersSpotActivatedValues::cellCopyMutationGenomeColor,
        data,
        cell->pos,
        cell->color);

    executeMultipleEvents(data, cellCopyMutationCellProperties, [&]() { propertiesMutation(data, cell); });
    executeMultipleEvents(data, cellCopyMutationNeuronData, [&]() { neuronDataMutation(data, cell); });
    executeEvent(data, cellCopyMutationGeometry, [&]() { geometryMutation(data, cell); });
    executeEvent(data, cellCopyMutationCustomGeometry, [&]() { customGeometryMutation(data, cell); });
    executeMultipleEvents(data, cellCopyMutationCellFunction, [&]() { cellFunctionMutation(data, cell); });
    executeEvent(data, cellCopyMutationInsertion, [&]() {
        auto numNonSeparatedNodes = toFloat(GenomeDecoder::getNumNodesRecursively(constructor.genome, constructor.genomeSize, false, false));
        if (numNodes < 2 * numNonSeparatedNodes) {
            insertMutation(data, cell);
        }
    });
    executeEvent(data, cellCopyMutationDeletion, [&]() { deleteMutation(data, cell); });
    executeEvent(data, cellCopyMutationCellColor, [&]() { cellColorMutation(data, cell); });
    executeEvent(data, cellCopyMutationTranslation, [&]() { translateMutation(data, cell); });
    executeEvent(data, cellCopyMutationDuplication, [&]() {
        auto& constructor = cell->cellFunctionData.constructor;
        auto numNodes = toFloat(GenomeDecoder::getNumNodesRecursively(constructor.genome, constructor.genomeSize, false, true));
        auto numNonSeparatedNodes = toFloat(GenomeDecoder::getNumNodesRecursively(constructor.genome, constructor.genomeSize, false, false));
        if (numNodes < 2 * numNonSeparatedNodes) {
            duplicateMutation(data, cell);
        }
    });
    executeEvent(data, cellCopyMutationSubgenomeColor, [&]() { subgenomeColorMutation(data, cell); });
    executeEvent(data, cellCopyMutationGenomeColor, [&]() { genomeColorMutation(data, cell); });
}

__inline__ __device__ void MutationProcessor::neuronDataMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;
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
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;
    auto numNodes = GenomeDecoder::getNumNodesRecursively(genome, genomeSize, false, true);
    auto node = data.numberGen1.random(numNodes - 1);
    auto sequenceNumber = 0;

    uint8_t prevExecutionNumber = data.numberGen1.randomByte();
    uint8_t nextExecutionNumber = data.numberGen1.randomByte();
    uint8_t prevInputExecutionNumber = data.numberGen1.randomByte();
    uint8_t nextInputExecutionNumber = data.numberGen1.randomByte();
    int nodeAddress = 0;
    GenomeDecoder::executeForEachNodeRecursively(genome, genomeSize, true, false, [&](int depth, int nodeAddressIntern, int repetition) {
        auto origSequenceNumber = sequenceNumber;
        ++sequenceNumber;
        if (origSequenceNumber == node - 1) {
            prevExecutionNumber = GenomeDecoder::getNextExecutionNumber(genome, nodeAddressIntern);
            prevInputExecutionNumber = GenomeDecoder::getNextInputExecutionNumber(genome, nodeAddressIntern);
        }
        if (origSequenceNumber == node + 1) {
            nextExecutionNumber = GenomeDecoder::getNextExecutionNumber(genome, nodeAddressIntern);
            nextInputExecutionNumber = GenomeDecoder::getNextInputExecutionNumber(genome, nodeAddressIntern);
        }
        if (origSequenceNumber == node) {
            nodeAddress = nodeAddressIntern;
        }
    });
    if (nodeAddress == 0) {
        return;
    }

    //already fitting input-output connection? => use other input
    auto executionNumber = GenomeDecoder::getNextInputExecutionNumber(genome, nodeAddress) % cudaSimulationParameters.cellNumExecutionOrderNumbers;
    if (executionNumber == (prevInputExecutionNumber % cudaSimulationParameters.cellNumExecutionOrderNumbers)) {
        prevExecutionNumber = nextExecutionNumber;
    }
    if (executionNumber == (nextInputExecutionNumber % cudaSimulationParameters.cellNumExecutionOrderNumbers)) {
        nextExecutionNumber = prevExecutionNumber;
    }

    //basic property mutation
    if (data.numberGen1.randomBool()) {
        if (data.numberGen1.randomBool()) {
            auto randomByte = data.numberGen1.randomByte();
            if (data.numberGen1.random() < 0.8f) {
                randomByte = data.numberGen1.randomBool() ? prevExecutionNumber : nextExecutionNumber;
            }
            GenomeDecoder::setNextInputExecutionNumber(genome, nodeAddress, randomByte);
        } else {
            auto randomDelta = data.numberGen1.random(Const::CellBasicBytes - 1);
            auto randomByte = data.numberGen1.randomByte();
            if (randomDelta == 0) {  //no cell function type change
                return;
            }
            if (randomDelta == Const::CellColorPos) {  //no color change
                return;
            }
            if (randomDelta == Const::CellAnglePos || randomDelta == Const::CellRequiredConnectionsPos) {  //no structure change
                return;
            }
            genome[nodeAddress + randomDelta] = randomByte;
        }
    }

    //cell function specific mutation
    else {
        auto nextCellFunctionDataSize = GenomeDecoder::getNextCellFunctionDataSize(genome, genomeSize, nodeAddress, false);
        if (nextCellFunctionDataSize > 0) {
            auto randomDelta = data.numberGen1.random(nextCellFunctionDataSize - 1);
            auto cellFunction = GenomeDecoder::getNextCellFunctionType(genome, nodeAddress);
            if (cellFunction == CellFunction_Constructor
                && (randomDelta == Const::ConstructorConstructionAngle1Pos
                    || randomDelta == Const::ConstructorConstructionAngle2Pos)) {  //no construction angles change
                return;
            }
            genome[nodeAddress + Const::CellBasicBytes + randomDelta] = data.numberGen1.randomByte();
            //adaptMutationId(data, constructor);
        }
    }
}

__inline__ __device__ void MutationProcessor::geometryMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;

    auto subgenome = genome;
    auto subgenomeSize = genomeSize;
    if (genomeSize > Const::GenomeHeaderSize) {
        int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH];
        int numSubGenomesSizeIndices;
        GenomeDecoder::getRandomGenomeNodeAddress(
            data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);  //return value will be discarded

        if (numSubGenomesSizeIndices > 0) {
            subgenome = genome + subGenomesSizeIndices[numSubGenomesSizeIndices - 1] + 2;  //+2 because 2 bytes encode the sub-genome length
            subgenomeSize = GenomeDecoder::readWord(genome, subGenomesSizeIndices[numSubGenomesSizeIndices - 1]);
        }
    }

    auto delta = data.numberGen1.random(Const::GenomeHeaderSize - 1);

    if (delta == Const::GenomeHeaderNumRepetitionsPos) {
        auto choice = data.numberGen1.random(250);
        if (choice < 230) {
            subgenome[delta] = static_cast<uint8_t>(1 + data.numberGen1.random(2));
        } else if (choice < 240) {
            subgenome[delta] = static_cast<uint8_t>(1 + data.numberGen1.random(10));
        } else if (choice == 240) {
            subgenome[delta] = static_cast<uint8_t>(1 + data.numberGen1.random(20));
        } else {
            //no infinite repetitions
            //subgenome[delta] = 255;
        }
        return;
    }
    if (delta == Const::GenomeHeaderNumBranchesPos) {
        subgenome[delta] = data.numberGen1.randomBool() ? 1 : data.numberGen1.randomByte();
    }

    auto mutatedByte = data.numberGen1.randomByte();
    if (delta == Const::GenomeHeaderSeparationPos && GenomeDecoder::convertByteToBool(mutatedByte)) {
        return;
    }
    if (delta == Const::GenomeHeaderShapePos) {
        auto shape = mutatedByte % ConstructionShape_Count;
        auto origShape = subgenome[delta] % ConstructionShape_Count;
        if (origShape != ConstructionShape_Custom && shape == ConstructionShape_Custom) {
            CudaShapeGenerator shapeGenerator;
            subgenome[Const::GenomeHeaderAlignmentPos] = shapeGenerator.getConstructorAngleAlignment(shape);

            GenomeDecoder::executeForEachNode(subgenome, subgenomeSize, [&](int nodeAddress) {
                auto generationResult = shapeGenerator.generateNextConstructionData(origShape);
                subgenome[nodeAddress + Const::CellAnglePos] = GenomeDecoder::convertAngleToByte(generationResult.angle);
                subgenome[nodeAddress + Const::CellRequiredConnectionsPos] =
                    GenomeDecoder::convertOptionalByteToByte(generationResult.numRequiredAdditionalConnections);
            }); 
        }
    }
    if (subgenome[delta] != mutatedByte) {
        adaptMutationId(data, constructor);
    }
    subgenome[delta] = mutatedByte;
}

__inline__ __device__ void MutationProcessor::customGeometryMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }


    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;

    auto numNodes = GenomeDecoder::getNumNodesRecursively(genome, genomeSize, false, true);
    auto node = data.numberGen1.random(numNodes - 1);
    auto sequenceNumber = 0;
    GenomeDecoder::executeForEachNodeRecursively(genome, genomeSize, true, false, [&](int depth, int nodeAddress, int repetition) {
        if (sequenceNumber++ != node) {
            return;
        }
        auto cellFunction = GenomeDecoder::getNextCellFunctionType(genome, nodeAddress);
        auto choice =
            cellFunction == CellFunction_Constructor ? data.numberGen1.random(3) : data.numberGen1.random(1);
        switch (choice) {
        case 0:
            GenomeDecoder::setNextAngle(genome, nodeAddress, data.numberGen1.randomByte());
            break;
        case 1:
            GenomeDecoder::setNextRequiredConnections(genome, nodeAddress, data.numberGen1.randomByte());
            break;
        case 2:
            GenomeDecoder::setNextConstructionAngle1(genome, nodeAddress, data.numberGen1.randomByte());
            break;
        case 3:
            GenomeDecoder::setNextConstructionAngle2(genome, nodeAddress, data.numberGen1.randomByte());
            break;
        }
        //adaptMutationId(data, constructor);
    });
}

__inline__ __device__ void MutationProcessor::cellFunctionMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;

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
        if (GenomeDecoder::containsSectionSelfReplication(genome + nodeAddress, Const::CellBasicBytes + origCellFunctionSize)) {
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
    if (newCellFunction == CellFunction_Constructor && !makeSelfCopy) {
        GenomeDecoder::setNextConstructorSeparation(targetGenome, nodeAddress, false);  //currently no sub-genome with separation property wished
    }

    for (int i = nodeAddress + Const::CellBasicBytes + origCellFunctionSize; i < genomeSize; ++i) {
        targetGenome[i + sizeDelta] = genome[i];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = GenomeDecoder::readWord(genome, subGenomesSizeIndices[i]);
        GenomeDecoder::writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize + sizeDelta);
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
    //adaptMutationId(data, constructor);
}

__inline__ __device__ void MutationProcessor::insertMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;

    int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices;

    int nodeAddress = 0;
    uint8_t prevExecutionNumber = data.numberGen1.randomByte();
    uint8_t nextExecutionNumber = data.numberGen1.randomByte();

    //calculate addess where the new node should be inserted
    if (data.numberGen1.randomBool() && genomeSize > Const::GenomeHeaderSize) {

        //choose a random node position to a constructor with a subgenome
        int numConstructorsWithSubgenome = 0;
        GenomeDecoder::executeForEachNodeRecursively(genome, genomeSize, true, false, [&](int depth, int nodeAddressIntern, int repetition) {
            auto cellFunctionType = GenomeDecoder::getNextCellFunctionType(genome, nodeAddressIntern);
            if (cellFunctionType == CellFunction_Constructor && !GenomeDecoder::isNextCellSelfReplication(genome, nodeAddressIntern)) {
                ++numConstructorsWithSubgenome;
            }
        });
        if (numConstructorsWithSubgenome > 0) {
            auto randomIndex = data.numberGen1.random(numConstructorsWithSubgenome - 1);
            auto counter = 0;
            GenomeDecoder::executeForEachNodeRecursively(genome, genomeSize, true, false, [&](int depth, int nodeAddressIntern, int repetition) {
                auto cellFunctionType = GenomeDecoder::getNextCellFunctionType(genome, nodeAddressIntern);
                if (cellFunctionType == CellFunction_Constructor && !GenomeDecoder::isNextCellSelfReplication(genome, nodeAddressIntern)) {
                    if (randomIndex == counter) {
                        nodeAddress = nodeAddressIntern + Const::CellBasicBytes + Const::ConstructorFixedBytes + 3 + 1;
                        prevExecutionNumber = genome[nodeAddressIntern + Const::CellExecutionNumberPos];
                    }
                    ++counter;
                }
            });
        }
    }
    nodeAddress = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, true, subGenomesSizeIndices, &numSubGenomesSizeIndices, nodeAddress);
    if (numSubGenomesSizeIndices >= GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH - 2) {
        return;
    }

    //insert node
    auto newColor = cell->color;
    if (nodeAddress < genomeSize) {
        newColor = GenomeDecoder::getNextCellColor(genome, nodeAddress);
        nextExecutionNumber = GenomeDecoder::getNextExecutionNumber(genome, nodeAddress);
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
    if (data.numberGen1.random() < 0.9f) {  //fitting input execution number should be often
        GenomeDecoder::setNextInputExecutionNumber(targetGenome, nodeAddress, data.numberGen1.randomBool() ? prevExecutionNumber : nextExecutionNumber);
    }
    if (data.numberGen1.random() < 0.9f) {
        GenomeDecoder::setNextOutputBlocked(targetGenome, nodeAddress, false);  //non-blocking output should be often
    }
    GenomeDecoder::setRandomCellFunctionData(data, targetGenome, nodeAddress + Const::CellBasicBytes, newCellFunction, makeSelfCopy, Const::GenomeHeaderSize);
    if (newCellFunction == CellFunction_Constructor && !makeSelfCopy) {
        GenomeDecoder::setNextConstructorSeparation(targetGenome, nodeAddress, false);      //currently no sub-genome with separation property wished
        auto numBranches = data.numberGen1.randomBool() ? 1 : data.numberGen1.randomByte();
        GenomeDecoder::setNextConstructorNumBranches(targetGenome, nodeAddress, numBranches);
    }

    for (int i = nodeAddress; i < genomeSize; ++i) {
        targetGenome[i + sizeDelta] = genome[i];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = GenomeDecoder::readWord(genome, subGenomesSizeIndices[i]);
        GenomeDecoder::writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize + sizeDelta);
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
    adaptMutationId(data, constructor);
}

__inline__ __device__ void MutationProcessor::deleteMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;

    int subGenomesSizeIndices[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH];
    int numSubGenomesSizeIndices;
    auto nodeAddress = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, false, subGenomesSizeIndices, &numSubGenomesSizeIndices);

    auto origCellFunctionSize = GenomeDecoder::getNextCellFunctionDataSize(genome, genomeSize, nodeAddress);
    auto deleteSize = Const::CellBasicBytes + origCellFunctionSize;

    if (!cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication) {
        if (GenomeDecoder::containsSectionSelfReplication(genome + nodeAddress, deleteSize)) {
            return;
        }
    }
    //auto deletedCellFunctionType = GenomeDecoder::getNextCellFunctionType(genome, nodeAddress);
    //if (deletedCellFunctionType == CellFunction_Constructor || deletedCellFunctionType == CellFunction_Injector) {
    //    adaptMutationId(data, constructor);
    //}

    auto targetGenomeSize = genomeSize - deleteSize;
    for (int i = nodeAddress; i < targetGenomeSize; ++i) {
        genome[i] = genome[i + deleteSize];
    }

    for (int i = 0; i < numSubGenomesSizeIndices; ++i) {
        auto subGenomeSize = GenomeDecoder::readWord(genome, subGenomesSizeIndices[i]);
        GenomeDecoder::writeWord(genome, subGenomesSizeIndices[i], subGenomeSize - deleteSize);
    }
    constructor.genomeCurrentNodeIndex = 0;
    constructor.genomeSize = targetGenomeSize;
    adaptMutationId(data, constructor);
}

__inline__ __device__ void MutationProcessor::translateMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    //calc source range
    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;
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
    if (!cudaSimulationParameters.cellFunctionConstructorMutationSelfReplication) {
        if (GenomeDecoder::containsSectionSelfReplication(genome + startSourceIndex, sourceRangeSize)) {
            return;
        }
    }

    //calc target insertion point
    int subGenomesSizeIndices2[GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH + 1];
    int numSubGenomesSizeIndices2;
    auto startTargetIndex = GenomeDecoder::getRandomGenomeNodeAddress(data, genome, genomeSize, true, subGenomesSizeIndices2, &numSubGenomesSizeIndices2);

    if (startTargetIndex >= startSourceIndex && startTargetIndex <= endSourceIndex) {
        return;
    }
    auto sourceRangeDepth = GenomeDecoder::getGenomeDepth(subGenome, subGenomeSize);
    if (cudaSimulationParameters.cellFunctionConstructorMutationPreventDepthIncrease) {
        auto genomeDepth = GenomeDecoder::getGenomeDepth(genome, genomeSize);
        if (genomeDepth < sourceRangeDepth + numSubGenomesSizeIndices2) {
            return;
        }
    }
    if (sourceRangeDepth + numSubGenomesSizeIndices2 >= GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH - 2) {
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
    constructor.genomeCurrentNodeIndex = 0;
    adaptMutationId(data, constructor);
}

__inline__ __device__ void MutationProcessor::duplicateMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;

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
        if (GenomeDecoder::containsSectionSelfReplication(genome + startSourceIndex, sizeDelta)) {
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

    auto sourceRangeDepth = GenomeDecoder::getGenomeDepth(subGenome, subGenomeSize);
    if (cudaSimulationParameters.cellFunctionConstructorMutationPreventDepthIncrease) {
        auto genomeDepth = GenomeDecoder::getGenomeDepth(genome, genomeSize);
        if (genomeDepth < sourceRangeDepth + numSubGenomesSizeIndices) {
            return;
        }
    }
    if (sourceRangeDepth + numSubGenomesSizeIndices >= GenomeDecoder::MAX_SUBGENOME_RECURSION_DEPTH - 2) {
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
        auto subGenomeSize = GenomeDecoder::readWord(targetGenome, subGenomesSizeIndices[i]);
        GenomeDecoder::writeWord(targetGenome, subGenomesSizeIndices[i], subGenomeSize + sizeDelta);
    }
    constructor.genomeSize = targetGenomeSize;
    constructor.genome = targetGenome;
    adaptMutationId(data, constructor);
}

__inline__ __device__ void MutationProcessor::cellColorMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;
    auto numNodes = GenomeDecoder::getNumNodesRecursively(genome, genomeSize, false, true);
    auto randomNode = data.numberGen1.random(numNodes - 1);
    auto sequenceNumber = 0;
    GenomeDecoder::executeForEachNodeRecursively(genome, genomeSize, true, false, [&](int depth, int nodeAddress, int repetition) {
        if (sequenceNumber++ != randomNode) {
            return;
        }
        auto origColor = GenomeDecoder::getNextCellColor(genome, nodeAddress);
        auto newColor = getNewColorFromTransition(data, origColor);
        if (newColor == -1) {
            return;
        }
        GenomeDecoder::setNextCellColor(genome, nodeAddress, newColor);
    });
}

__inline__ __device__ void MutationProcessor::subgenomeColorMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;

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
    if (newColor == -1) {
        return;
    }
    if (origColor != newColor) {
        adaptMutationId(data, constructor);
    }

    for (int dummy = 0; nodeAddress < subgenomeSize && dummy < subgenomeSize; ++dummy) {
        GenomeDecoder::setNextCellColor(subgenome, nodeAddress, newColor);
        nodeAddress += Const::CellBasicBytes + GenomeDecoder::getNextCellFunctionDataSize(subgenome, subgenomeSize, nodeAddress);
    }
}

__inline__ __device__ void MutationProcessor::genomeColorMutation(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (GenomeDecoder::hasEmptyGenome(constructor)) {
        return;
    }

    auto& genome = constructor.genome;
    auto const& genomeSize = constructor.genomeSize;

    auto origColor = GenomeDecoder::getNextCellColor(genome, Const::GenomeHeaderSize);
    auto newColor = getNewColorFromTransition(data, origColor);
    if (newColor == -1) {
        return;
    }
    if (origColor != newColor) {
        adaptMutationId(data, constructor);
    }

    GenomeDecoder::executeForEachNodeRecursively(
        genome, genomeSize, true, false, [&](int depth, int nodeAddress, int repetition) { GenomeDecoder::setNextCellColor(genome, nodeAddress, newColor); });
}

template <typename Func>
__inline__ __device__ void MutationProcessor::executeMultipleEvents(SimulationData& data, float probability, Func eventFunc)
{
    for (int i = 0, j = toInt(probability); i < j; ++i) {
        eventFunc();
    }
    if (isRandomEvent(data, probability)) {
        eventFunc();
    }
}

template <typename Func>
__inline__ __device__ void MutationProcessor::executeEvent(SimulationData& data, float probability, Func eventFunc)
{
    if (isRandomEvent(data, probability)) {
        eventFunc();
    }
}

__inline__ __device__ void MutationProcessor::adaptMutationId(SimulationData& data, ConstructorFunction& constructor)
{
    if (GenomeDecoder::containsSelfReplication(constructor)) {
        constructor.offspringMutationId = data.numberGen1.createNewSmallId();
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

__inline__ __device__ int MutationProcessor::getNewColorFromTransition(SimulationData& data, int origColor)
{
    int numAllowedColors = 0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        if (cudaSimulationParameters.cellFunctionConstructorMutationColorTransitions[origColor][i]) {
            ++numAllowedColors;
        }
    }
    if (numAllowedColors == 0) {
        return -1;
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

