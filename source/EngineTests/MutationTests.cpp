#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ranges>

#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class MutationTests : public IntegrationTestFramework
{
public:
    MutationTests()
        : IntegrationTestFramework()
    {
        _parameters.mutationNeuralNetwork.value = 1.0f;
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~MutationTests() = default;

protected:
    GenomeDesc createTestGenome() const
    {
        // TODO create a genome with multiple genes where each has multiple nodes of different type, use DescriptionTestDataFactory::createNonDefaultNodeDesc
    }

    bool compareExceptNeuralNetwork(GenomeDesc expected, GenomeDesc actual)
    {
        // compare genomes except neural network, good strategy: set all neural network properties of `expected` and `actual` to 0, then compare both
        return true;
    }
};

TEST_F(MutationTests, neuralNetworkMutation)
{
    auto genome = createTestGenome();

    auto data = Desc().addCreature({ObjectDesc().id(1).type(CellDesc())});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::NeuralNetwork);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // Mutated genome must be equal to original genome except the neural network properties
    EXPECT_TRUE(compareExceptNeuralNetwork(genome, actualGenome));
}
