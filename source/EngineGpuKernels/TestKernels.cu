#include "TestKernels.cuh"

#include "MutationProcessor.cuh"

__global__ void cudaTestMutate(SimulationData data, uint64_t cellId, MutationType mutationType)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->id == cellId) {
            switch (mutationType) {
            case MutationType::Data:
                MutationProcessor::mutateData(data, cell);
                break;
            case MutationType::NeuronData:
                MutationProcessor::mutateNeuronData(data, cell);
                break;
            case MutationType::CellFunction:
                MutationProcessor::mutateCellFunction(data, cell);
                break;
            case MutationType::Insertion:
                MutationProcessor::mutateInsertion(data, cell);
                break;
            }
        }
    }
}

