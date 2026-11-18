#pragma once

#include <algorithm>
#include <type_traits>
#include <variant>
#include <vector>

#include <Data/Descs.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Engine/Interface/TestData/DescTestDataFactory.h>

#include "IntegrationTestFramework.h"

class MutationTestsBase : public IntegrationTestFramework
{
public:
    MutationTestsBase()
        : IntegrationTestFramework()
    {
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~MutationTestsBase() = default;

protected:
    template <typename T>
    struct IsVariant : std::false_type
    {};
    template <typename... Ts>
    struct IsVariant<std::variant<Ts...>> : std::true_type
    {};

    // Resets all mutable cell type properties to their defaults while preserving the active cell type and its active mode.
    static void resetCellTypeProperties(CellTypeGenomeDesc& cellType)
    {
        std::visit(
            [](auto& cellTypeData) {
                using CellTypeData = std::decay_t<decltype(cellTypeData)>;
                if constexpr (requires { cellTypeData._mode; }) {
                    auto mode = cellTypeData._mode;
                    if constexpr (IsVariant<std::decay_t<decltype(mode)>>::value) {
                        std::visit([](auto& modeData) { modeData = std::decay_t<decltype(modeData)>{}; }, mode);
                    }
                    cellTypeData = CellTypeData{};
                    cellTypeData._mode = mode;
                    return;
                }
                cellTypeData = CellTypeData{};
            },
            cellType);
    }

    GenomeDesc createTestGenome() const
    {
        auto const& factory = DescTestDataFactory::get();
        auto nodeParameters = factory.getAllNodeParameters();

        std::vector<GeneDesc> genes;
        size_t constexpr nodesPerGene = 4;
        for (size_t index = 0; index < nodeParameters.size(); index += nodesPerGene) {
            auto nodeCount = std::min(nodesPerGene, nodeParameters.size() - index);
            std::vector<NodeDesc> nodes;
            nodes.reserve(nodeCount);
            for (size_t offset = 0; offset < nodeCount; ++offset) {
                nodes.push_back(factory.createNonDefaultNodeDesc(nodeParameters.at(index + offset)));
            }
            genes.emplace_back(GeneDesc().nodes(nodes));
        }

        return GenomeDesc().genes(genes);
    }

    GenomeDesc getMutatedGenome(uint64_t objectId = 1) const
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCell = actualData.getObjectRef(objectId).getCellRef();
        auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
        return actualData.getGenomeRef(actualCreature._genomeId);
    }

    CreatureDesc getMutatedCreature(uint64_t objectId = 1) const
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCell = actualData.getObjectRef(objectId).getCellRef();
        return actualData.getCreatureRef(actualCell._creatureId);
    }
};
