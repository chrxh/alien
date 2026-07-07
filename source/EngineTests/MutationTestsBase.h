#pragma once

#include <algorithm>
#include <vector>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescTestDataFactory.h>

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

        // Chain the genes via the first node's constructor (gene i -> gene i+1) so that every gene stays reachable
        // from the root gene (gene 0) and is not removed as unused by MutationProcessor::removeUnreachableGenesFromRoot.
        for (size_t geneIndex = 0; geneIndex + 1 < genes.size(); ++geneIndex) {
            auto& constructor = genes.at(geneIndex)._nodes.at(0)._constructor;
            if (!constructor.has_value()) {
                constructor = ConstructorGenomeDesc();
            }
            constructor->_geneIndex = static_cast<int>(geneIndex + 1);
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
