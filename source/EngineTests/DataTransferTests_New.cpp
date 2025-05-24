#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"


class DataTransferTests_New : public IntegrationTestFramework
{
public:
    DataTransferTests_New()
        : IntegrationTestFramework()
    {}
};

TEST_F(DataTransferTests_New, singleCellWithoutGenome)
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
                     .signal({1, 0, -1, 0, 0, 0, 0, 0}));

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests_New, singleCellWithGenome)
{
    NeuralNetworkDescription nn1;
    nn1.weight(2, 1, 1.0f);
    NeuralNetworkGenomeDescription nn2;
    nn2.weight(1, 3, -1.0f);
    auto data = CollectionDescription().addCreature(
        GenomeDescription_New().id(1).genes({GeneDescription().nodes({NodeDescription().neuralNetwork(nn2).cellTypeData(AttackerGenomeDescription())})}),
        {CellDescription()
             .neuralNetwork(nn1)
             .id(1)
             .pos({2.0f, 4.0f})
             .vel({0.5f, 1.0f})
             .energy(100.0f)
             .age(1)
             .color(2)
             .barrier(true)
             .livingState(false)
             .creatureId(3534)
             .signal({1, 0, -1, 0, 0, 0, 0, 0})});

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}
