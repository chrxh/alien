#include "MassOperationsService.h"

#include <unordered_map>
#include <unordered_set>
#include <queue>

#include "GenomeDesc.h"
#include "NumberGenerator.h"

namespace
{
    std::vector<std::vector<size_t>> calcClusters(ContentDesc const& description)
    {
        std::vector<std::vector<size_t>> clusters;

        // Step 1: Group cells by creatureId
        std::unordered_map<uint64_t, std::vector<size_t>> cellsByCreatureId;
        for (size_t i = 0; i < description._objects.size(); ++i) {
            auto const& object = description._objects.at(i);
            if (object.getObjectType() == ObjectType_Cell) {
                cellsByCreatureId[object.getCellRef()._creatureId].push_back(i);
            }
        }
        for (auto& [_, indices] : cellsByCreatureId) {
            clusters.push_back(std::move(indices));
        }

        // Step 2: For non-cell objects, find path-connected components via BFS
        std::unordered_map<uint64_t, size_t> idToIndex;
        std::unordered_set<size_t> nonCellIndices;
        for (size_t i = 0; i < description._objects.size(); ++i) {
            auto const& object = description._objects.at(i);
            idToIndex[object._id] = i;
            if (object.getObjectType() != ObjectType_Cell) {
                nonCellIndices.insert(i);
            }
        }

        std::unordered_set<size_t> visited;
        for (auto index : nonCellIndices) {
            if (visited.count(index)) {
                continue;
            }

            std::vector<size_t> component;
            std::queue<size_t> bfsQueue;
            bfsQueue.push(index);
            visited.insert(index);

            while (!bfsQueue.empty()) {
                auto current = bfsQueue.front();
                bfsQueue.pop();
                component.push_back(current);

                for (auto const& connection : description._objects.at(current)._connections) {
                    auto it = idToIndex.find(connection._objectId);
                    if (it != idToIndex.end() && nonCellIndices.count(it->second) && !visited.count(it->second)) {
                        visited.insert(it->second);
                        bfsQueue.push(it->second);
                    }
                }
            }
            clusters.push_back(std::move(component));
        }

        return clusters;
    }
}

void MassOperationsService::randomizeCellColors(ContentDesc& description, std::vector<int> const& colorCodes) const
{
    auto clusters = calcClusters(description);
    for (auto const& cluster : clusters) {
        auto color = colorCodes.at(NumberGenerator::get().getRandomInt(toInt(colorCodes.size())));
        for (auto index : cluster) {
            description._objects.at(index)._color = color;
        }
    }
}

void MassOperationsService::randomizeGenomeColors(ContentDesc& description, std::vector<int> const& colorCodes) const
{
    for (auto& genome : description._genomes) {
        auto newColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
        for (auto& gene : genome._genes) {
            for (auto& node : gene._nodes) {
                node._color = newColor;
            }
        }
    }
}

void MassOperationsService::randomizeEnergies(ContentDesc& description, float minEnergy, float maxEnergy) const
{
    auto clusters = calcClusters(description);
    for (auto const& cluster : clusters) {
        auto energy = NumberGenerator::get().getRandomFloat(toFloat(minEnergy), toFloat(maxEnergy));
        for (auto index : cluster) {
            auto& object = description._objects.at(index);
            auto type = object.getObjectType();
            if (type == ObjectType_Cell) {
                object.getCellRef()._usableEnergy = energy;
            } else if (type == ObjectType_FreeCell) {
                object.getFreeCellRef()._energy = energy;
            } else if (type == ObjectType_Solid) {
                object.getSolidRef()._energy = energy;
            } else if (type == ObjectType_Fluid) {
                object.getFluidRef()._energy = energy;
            }
        }
    }
}

void MassOperationsService::randomizeAges(ContentDesc& description, int minAge, int maxAge) const
{
    auto clusters = calcClusters(description);
    for (auto const& cluster : clusters) {
        auto age = static_cast<int>(NumberGenerator::get().getRandomFloat(toFloat(minAge), toFloat(maxAge)));
        for (auto index : cluster) {
            auto& object = description._objects.at(index);
            auto type = object.getObjectType();
            if (type == ObjectType_Cell) {
                object.getCellRef()._age = age;
            } else if (type == ObjectType_FreeCell) {
                object.getFreeCellRef()._age = age;
            }
        }
    }
}

void MassOperationsService::randomizeCountdowns(ContentDesc& description, int minValue, int maxValue) const
{
    auto clusters = calcClusters(description);
    for (auto const& cluster : clusters) {
        auto countdown = static_cast<int>(NumberGenerator::get().getRandomDouble(toDouble(minValue), toDouble(maxValue)));
        for (auto index : cluster) {
            auto& object = description._objects.at(index);
            if (object.getObjectType() != ObjectType_Cell) {
                continue;
            }
            if (object.getCellRef().getCellType() == CellType_Detonator) {
                std::get<DetonatorDesc>(object.getCellRef()._cellType)._countdown = countdown;
            }
        }
    }
}

void MassOperationsService::randomizeLineageIds(ContentDesc& description) const
{
    for (auto& creature : description._creatures) {
        creature._lineageId = toInt(NumberGenerator::get().createLineageId());
    }
}

void MassOperationsService::randomizeGlow(ContentDesc& description, float minGlow, float maxGlow) const
{
    auto clusters = calcClusters(description);
    for (auto const& cluster : clusters) {
        auto glow = NumberGenerator::get().getRandomFloat(minGlow, maxGlow);
        for (auto index : cluster) {
            auto& object = description._objects.at(index);
            if (object.getObjectType() == ObjectType_Fluid) {
                object.getFluidRef()._glow = glow;
            }
        }
    }
}

void MassOperationsService::setMutationRates(ContentDesc& description, MutationRatesDesc const& mutationRates) const
{
    for (auto& genome : description._genomes) {
        genome._mutationRates = mutationRates;
    }
}
