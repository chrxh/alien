#include "SelectionKernels.cuh"

__global__ void cudaRemoveSelection(SimulationData data, bool onlyClusterSelection)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cells.at(index);
        if (!onlyClusterSelection || cell->selected == 2) {
            cell->selected = 0;
        }
    }

    auto const particlePartition = calcAllThreadsPartition(data.objects.particles.getNumEntries());

    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto const& particle = data.objects.particles.at(index);
        if (!onlyClusterSelection || particle->selected == 2) {
            particle->selected = 0;
        }
    }
}

__global__ void cudaSwapSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cells.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cells.at(index);
        if (data.cellMap.getDistance(pos, cell->pos) < radius) {
            if (cell->selected == 0) {
                cell->selected = 1;
            } else if (cell->selected == 1) {
                cell->selected = 0;
            }
        }
    }

    auto const particlePartition = calcAllThreadsPartition(data.objects.particles.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto const& particle = data.objects.particles.at(index);
        if (data.particleMap.getDistance(pos, particle->pos) < radius) {
            particle->selected = 1 - particle->selected;
        }
    }
}

__global__ void cudaExistsSelection(PointSelectionData pointData, SimulationData data, int* result)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cells.at(index);
        if (1 == cell->selected && data.cellMap.getDistance(pointData.pos, cell->pos) < pointData.radius) {
            atomicExch(result, 1);
        }
    }

    auto const particlePartition = calcAllThreadsPartition(data.objects.particles.getNumEntries());

    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto const& particle = data.objects.particles.at(index);
        if (1 == particle->selected && data.cellMap.getDistance(pointData.pos, particle->pos) < pointData.radius) {
            atomicExch(result, 1);
        }
    }
}

__global__ void cudaSetSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cells.at(index);
        if (data.cellMap.getDistance(pos, cell->pos) < radius) {
            cell->selected = 1;
        } else {
            cell->selected = 0;
        }
    }

    auto const particlePartition = calcAllThreadsPartition(data.objects.particles.getNumEntries());

    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto const& particle = data.objects.particles.at(index);
        if (data.particleMap.getDistance(pos, particle->pos) < radius) {
            particle->selected = 1;
        } else {
            particle->selected = 0;
        }
    }
}

__global__ void cudaSetSelection(AreaSelectionData selectionData, SimulationData data)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cells.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cells.at(index);

        if (Math::isInBetweenModulo(toFloat(selectionData.startPos.x), toFloat(selectionData.endPos.x), cell->pos.x, toFloat(data.worldSize.x))
            && Math::isInBetweenModulo(toFloat(selectionData.startPos.y), toFloat(selectionData.endPos.y), cell->pos.y, toFloat(data.worldSize.y))) {
            cell->selected = 1;
        } else {
            cell->selected = 0;
        }
    }

    auto const particlePartition = calcAllThreadsPartition(data.objects.particles.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto const& particle = data.objects.particles.at(index);
        if (Math::isInBetweenModulo(toFloat(selectionData.startPos.x), toFloat(selectionData.endPos.x), particle->pos.x, toFloat(data.worldSize.x))
            && Math::isInBetweenModulo(toFloat(selectionData.startPos.y), toFloat(selectionData.endPos.y), particle->pos.y, toFloat(data.worldSize.y))) {
            particle->selected = 1;
        } else {
            particle->selected = 0;
        }
    }
}

__global__ void cudaRolloutSelectionStep(SimulationData data, int* result)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cells.at(index);

        if (0 != cell->selected) {
            auto currentCell = cell;

            //heuristics to cover connected cells
            for (int i = 0; i < 30; ++i) {
                bool found = false;
                for (int j = 0; j < currentCell->numConnections; ++j) {
                    auto candidateCell = currentCell->connections[j].cell;
                    if (0 == candidateCell->selected) {
                        currentCell = candidateCell;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    break;
                }

                currentCell->selected = 2;
                atomicExch(result, 1);
            }
        }
    }
}
