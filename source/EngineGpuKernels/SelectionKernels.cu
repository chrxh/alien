#include "SelectionKernels.cuh"

__global__ void cudaRemoveSelection(SimulationData data, bool onlyClusterSelection)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (!onlyClusterSelection || object->selected == 2) {
            object->selected = 0;
        }
    }

    auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (!onlyClusterSelection || particle->selected == 2) {
            particle->selected = 0;
        }
    }
}

__global__ void cudaSwapSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (data.objectMap.getDistance(pos, object->pos) < radius) {
            if (object->selected == 0) {
                object->selected = 1;
            } else if (object->selected == 1) {
                object->selected = 0;
            }
        }
    }

    auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (data.energyMap.getDistance(pos, particle->pos) < radius) {
            particle->selected = 1 - particle->selected;
        }
    }
}

__global__ void cudaExistsSelection(PointSelectionData pointData, SimulationData data, int* result)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (1 == object->selected && data.objectMap.getDistance(pointData.pos, object->pos) < pointData.radius) {
            atomicExch(result, 1);
        }
    }

    auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (1 == particle->selected && data.objectMap.getDistance(pointData.pos, particle->pos) < pointData.radius) {
            atomicExch(result, 1);
        }
    }
}

__global__ void cudaSetSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (data.objectMap.getDistance(pos, object->pos) < radius) {
            object->selected = 1;
        } else {
            object->selected = 0;
        }
    }

    auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (data.energyMap.getDistance(pos, particle->pos) < radius) {
            particle->selected = 1;
        } else {
            particle->selected = 0;
        }
    }
}

__global__ void cudaSetSelection(AreaSelectionData selectionData, SimulationData data)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);

        if (Math::isInBetweenModulo(toFloat(selectionData.startPos.x), toFloat(selectionData.endPos.x), object->pos.x, toFloat(data.worldSize.x))
            && Math::isInBetweenModulo(toFloat(selectionData.startPos.y), toFloat(selectionData.endPos.y), object->pos.y, toFloat(data.worldSize.y))) {
            object->selected = 1;
        } else {
            object->selected = 0;
        }
    }

    auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
        auto const& particle = data.entities.energies.at(index);
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
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);

        if (0 != object->selected) {
            auto currentCell = object;

            //heuristics to cover connected cells
            for (int i = 0; i < 30; ++i) {
                bool found = false;
                for (int j = 0; j < currentCell->numConnections; ++j) {
                    auto candidateCell = currentCell->connections[j].object;
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
