#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class DataTransferTests : public IntegrationTestFramework
{
public:
    DataTransferTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~DataTransferTests() = default;
};

TEST_F(DataTransferTests, singleCell)
{
    DataDescription data;
    NeuronDescription neuron;
    neuron.weigths[2][1] = 1.0f;
    data.addCell(CellDescription()
                     .setPos({2.0f, 4.0f})
                     .setVel({0.5f, 1.0f})
                     .setEnergy(100.0f)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(3)
                     .setAge(1)
                     .setColor(2)
                     .setBarrier(true)
                     .setUnderConstruction(false)
                     .setInputBlocked(true)
                     .setOutputBlocked(false)
    );

    _simController->setSimulationData(data);
    auto actualData = _simController->getSimulationData();
    ASSERT_EQ(1, actualData.cells.size());
    EXPECT_TRUE(compare(data.cells.front(), actualData.cells.front()));
}

TEST_F(DataTransferTests, singleParticle)
{
    DataDescription data;
    NeuronDescription neuron;
    neuron.weigths[2][1] = 1.0f;
    data.addParticle(ParticleDescription().setPos({2.0f, 4.0f}).setVel({0.5f, 1.0f}).setEnergy(100.0f).setColor(2));

    _simController->setSimulationData(data);
    auto actualData = _simController->getSimulationData();
    ASSERT_EQ(1, actualData.particles.size());
    EXPECT_TRUE(compare(data.particles.front(), actualData.particles.front()));
}

TEST_F(DataTransferTests, cellCluster)
{
    RealVector2D const pos1{2.0f, 4.0f};
    RealVector2D const pos2{3.0f, 4.0f};

    DataDescription data;
    NeuronDescription neuron;
    neuron.weigths[2][1] = 1.0f;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos(pos1)
             .setVel({0.5f, 1.0f})
             .setMaxConnections(1)
             .setConnectingCells({ConnectionDescription().setCellId(2).setDistance(1.2f)})
             .setExecutionOrderNumber(3)
             .setAge(1)
             .setColor(2)
             .setBarrier(true)
             .setUnderConstruction(false)
             .setInputBlocked(true)
             .setOutputBlocked(false),
         CellDescription()
             .setId(2)
             .setPos(pos2)
             .setVel({0.2f, 1.0f})
             .setMaxConnections(1)
             .setConnectingCells({ConnectionDescription().setCellId(1).setDistance(1.2f)})
             .setExecutionOrderNumber(3)
             .setAge(1)
             .setColor(2)
             .setBarrier(true)
             .setUnderConstruction(false)
             .setInputBlocked(true)
             .setOutputBlocked(false)});

    _simController->setSimulationData(data);
    auto actualData = _simController->getSimulationData();

    auto cellsByPos = getCellsByPosition(data);
    auto actualCellsByPos = getCellsByPosition(actualData);

    ASSERT_EQ(1, actualCellsByPos.at(pos1).size());
    ASSERT_EQ(1, actualCellsByPos.at(pos2).size());

    EXPECT_TRUE(compare(cellsByPos.at(pos1).front(), actualCellsByPos.at(pos1).front()));
    EXPECT_TRUE(compare(cellsByPos.at(pos2).front(), actualCellsByPos.at(pos2).front()));
}

TEST_F(DataTransferTests, massTransfer)
{
    auto addCellAndParticles = [](DataDescription& data) {
        data.addCell(CellDescription()
                         .setPos({2.0f, 4.0f})
                         .setVel({0.5f, 1.0f})
                         .setMaxConnections(1)
                         .setExecutionOrderNumber(3)
                         .setAge(1)
                         .setColor(2)
                         .setBarrier(true)
                         .setUnderConstruction(false)
                         .setInputBlocked(true)
                         .setOutputBlocked(false));
        data.addParticle(ParticleDescription());
    };

    DataDescription data;
    for (int i = 0; i < 100000; ++i) {
        addCellAndParticles(data);
    }
    {
        _simController->setSimulationData(data);
        auto actualData = _simController->getSimulationData();
        EXPECT_EQ(data.cells.size(), actualData.cells.size());
        EXPECT_EQ(data.particles.size(), actualData.particles.size());
    }

    DataDescription newData;
    for (int i = 0; i < 1000000; ++i) {
        addCellAndParticles(newData);
    }
    {
        _simController->addAndSelectSimulationData(newData);
        auto actualData = _simController->getSimulationData();
        EXPECT_EQ(data.cells.size() + newData.cells.size(), actualData.cells.size());
        EXPECT_EQ(data.particles.size() + newData.particles.size(), actualData.particles.size());
    }
}
