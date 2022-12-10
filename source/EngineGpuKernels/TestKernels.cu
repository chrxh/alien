#include "TestKernels.cuh"

#include "MutationProcessor.cuh"

__global__ void cudaMutateNeuronData(SimulationData data, uint64_t cellId)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->id == cellId) {
            MutationProcessor::mutateNeuronData(data, cell);
        }
    }
}

__global__ void cudaMutateData(SimulationData data, uint64_t cellId)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->id == cellId) {
            MutationProcessor::mutateData(data, cell);
        }
    }
}
