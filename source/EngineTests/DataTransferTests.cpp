#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class DataTransferTests : public IntegrationTestFramework
{
public:
    DataTransferTests()
        : IntegrationTestFramework()
    {}

    ~DataTransferTests() = default;
};

TEST_F(DataTransferTests, singleCell)
{
    CollectionDescription data;
    NeuralNetworkDescription nn;
    nn.weight(2, 1, 1.0f);
    data.addCell(CellDescription()
                     .neuralNetwork(nn)
                     .id(1)
                     .pos({2.0f, 4.0f})
                     .vel({0.5f, 1.0f})
                     .energy(100.0f)
                     .age(1)
                     .color(2)
                     .barrier(true)
                     .livingState(false)
                     .creatureId(3534)
                     .signalAndRelaxTime({1, 0, -1, 0, 0, 0, 0, 0}));

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, singleParticle)
{
    CollectionDescription data;

    data.addParticle(ParticleDescription().id(1).pos({2.0f, 4.0f}).vel({0.5f, 1.0f}).energy(100.0f).color(2));

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, cellCluster)
{
    NeuralNetworkDescription nn1;
    nn1.weight(2, 1, 1.0f);
    NeuralNetworkDescription nn2;
    nn2.weight(5, 3, 1.0f);

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({2.0f, 4.0f}).vel({0.5f, 1.0f}).age(1).color(2).barrier(false).livingState(false).neuralNetwork(nn1),
        CellDescription().id(2).pos({3.0f, 4.0f}).vel({0.2f, 1.0f}).age(1).color(4).barrier(true).livingState(false).neuralNetwork(nn2),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, largeData)
{
    auto& numberGen = NumberGenerator::get();
    auto addCellAndParticles = [&](CollectionDescription& data) {
        data.addCell(CellDescription()
                         .id(numberGen.getId())
                .pos({numberGen.getRandomFloat(0.0f, 100.0f), numberGen.getRandomFloat(0.0f, 100.0f)})
                         .vel({numberGen.getRandomFloat(-1.0f, 1.0f), numberGen.getRandomFloat(-1.0f, 1.0f)})
                        .energy(numberGen.getRandomFloat(0.0f, 100.0f))
                         .age(1)
                         .color(2)
                         .barrier(true)
                         .livingState(false));
        data.addParticle(ParticleDescription()
                             .id(numberGen.getId())
                             .pos({numberGen.getRandomFloat(0.0f, 100.0f), numberGen.getRandomFloat(0.0f, 100.0f)})
                             .vel({numberGen.getRandomFloat(-1.0f, 1.0f), numberGen.getRandomFloat(-1.0f, 1.0f)})
                             .energy(numberGen.getRandomFloat(0.0f, 100.0f)));
    };

    CollectionDescription data;
    for (int i = 0; i < 100000; ++i) {
        addCellAndParticles(data);
    }
    {
        _simulationFacade->setSimulationData(data);
        auto actualData = _simulationFacade->getSimulationData();
        EXPECT_TRUE(compare(data, actualData));
    }

    CollectionDescription newData;
    for (int i = 0; i < 1000000; ++i) {
        addCellAndParticles(newData);
    }
    {
        _simulationFacade->addAndSelectSimulationData(newData);
        auto actualData = _simulationFacade->getSimulationData();
        EXPECT_EQ(data._cells.size() + newData._cells.size(), actualData._cells.size());
        EXPECT_EQ(data._particles.size() + newData._particles.size(), actualData._particles.size());
    }
}
