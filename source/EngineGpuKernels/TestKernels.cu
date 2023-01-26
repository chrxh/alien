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
                MutationProcessor::changeDataMutation(data, cell);
                break;
            case MutationType::NeuronData:
                MutationProcessor::changeNeuronDataMutation(data, cell);
                break;
            case MutationType::CellFunction:
                MutationProcessor::changeCellFunctionMutation(data, cell);
                break;
            case MutationType::Insertion:
                MutationProcessor::insertMutation(data, cell);
                break;
            case MutationType::Deletion:
                MutationProcessor::deleteMutation(data, cell);
                break;
            case MutationType::Duplication:
                MutationProcessor::duplicateMutation(data, cell);
                break;
            }
        }
    }
}

