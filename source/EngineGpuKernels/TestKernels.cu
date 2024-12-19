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
            case MutationType::CellFunction:
                MutationProcessor::cellFunctionMutation(data, cell);
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

__global__ void cudaTestMutationCheck(SimulationData data, uint64_t cellId)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->id == cellId) {
            MutationProcessor::checkMutationsForCell(data, cell);
        }
    }
}

