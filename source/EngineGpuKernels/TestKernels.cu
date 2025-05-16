#include "TestKernels.cuh"

#include "MutationProcessor.cuh"

__global__ void cudaTestMutate(SimulationData data, uint64_t cellId, MutationType mutationType)
{
    auto& cells = data.objects.cells;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->id == cellId) {
            switch (mutationType) {
            case MutationType::Properties:
                MutationProcessor::propertiesMutation(data, cell);
                break;
            case MutationType::NeuronData:
                MutationProcessor::neuronDataMutation(data, cell);
                break;
            case MutationType::Geometry:
                MutationProcessor::geometryMutation(data, cell);
                break;
            case MutationType::CustomGeometry:
                MutationProcessor::customGeometryMutation(data, cell);
                break;
            case MutationType::CellType:
                MutationProcessor::cellTypeMutation(data, cell);
                break;
            case MutationType::Insertion:
                MutationProcessor::insertMutation(data, cell);
                break;
            case MutationType::Deletion:
                MutationProcessor::deleteMutation(data, cell);
                break;
            case MutationType::Translation:
                MutationProcessor::translateMutation(data, cell);
                break;
            case MutationType::Duplication:
                MutationProcessor::duplicateMutation(data, cell);
                break;
            case MutationType::CellColor:
                MutationProcessor::cellColorMutation(data, cell);
                break;
            case MutationType::SubgenomeColor:
                MutationProcessor::subgenomeColorMutation(data, cell);
                break;
            case MutationType::GenomeColor:
                MutationProcessor::genomeColorMutation(data, cell);
                break;
            }
        }
    }
}

__global__ void cudaTestCreateConnection(SimulationData data, uint64_t cellId1, uint64_t cellId2)
{
    CUDA_CHECK(blockDim.x == 1 && gridDim.x == 1);

    auto& cells = data.objects.cells;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());
    Cell* cell1 = nullptr;
    Cell* cell2 = nullptr;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->id == cellId1) {
            cell1 = cell;
        }
        if (cell->id == cellId2) {
            cell2 = cell;
        }
    }

    if (cell1 != nullptr && cell2 != nullptr) {
        data.cellMap.reset();
        data.particleMap.reset();
        data.processMemory.reset();

        // Heuristics
        auto maxStructureOperations = 1000 + data.objects.cells.getNumEntries() / 2;
        auto maxCellTypeOperations = data.objects.cells.getNumEntries();

        data.structuralOperations.setMemory(data.processMemory.getTypedSubArray<StructuralOperation>(maxStructureOperations), maxStructureOperations);
        for (int i = CellType_Base; i < CellType_Count; ++i) {
            data.cellTypeOperations[i].setMemory(data.processMemory.getTypedSubArray<CellTypeOperation>(maxCellTypeOperations), maxCellTypeOperations);
        }
        *data.externalEnergy = cudaSimulationParameters.externalEnergy.value;
        data.objects.saveNumEntries();

        CellConnectionProcessor::scheduleAddConnectionPair(data, cell1, cell2);
        data.structuralOperations.saveNumEntries();
        CellConnectionProcessor::processAddOperations(data);
    }
}

__global__ void cudaTestAreArraysValid(SimulationData data, bool* result)
{
    auto& cells = data.objects.cells;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        if (auto& cell = cells.at(index)) {

            bool isValid = true;
            if (reinterpret_cast<uint64_t>(cell) < reinterpret_cast<uint64_t>(data.objects.heap.getArray())
                || reinterpret_cast<uint64_t>(cell) >= reinterpret_cast<uint64_t>(data.objects.heap.getArray() + data.objects.heap.getCapacity())) {
                *result = false;
                isValid = false;
            }

            if (isValid) {
                for (int i = 0; i < cell->numConnections; ++i) {
                    auto connectingCell = cell->connections[i].cell;
                    if (reinterpret_cast<uint64_t>(connectingCell) < reinterpret_cast<uint64_t>(data.objects.heap.getArray())
                        || reinterpret_cast<uint64_t>(connectingCell)
                            >= reinterpret_cast<uint64_t>(data.objects.heap.getArray() + data.objects.heap.getCapacity())) {
                        *result = false;
                    }
                }
            }
        }
    }
}

__global__ void cudaTestMutationCheck(SimulationData data, uint64_t cellId)
{
    auto& cells = data.objects.cells;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->id == cellId) {
            MutationProcessor::checkMutationsForCell(data, cell);
        }
    }
}

