#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class RibosomeTests : public IntegrationTestFramework
{
public:
    RibosomeTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~RibosomeTests() = default;
};

TEST_F(RibosomeTests, constructSingleCell_noSeparation)
{
    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(
                         RibosomeDescription().setGenome({CellDescription().setColor(2).setExecutionOrderNumber(4)}).setSeparateConstruction(false)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<RibosomeDescription>(*actualHostCell.cellFunction).currentGenomePos);
    expectApproxEqual(_parameters.cellNormalEnergy * 2, actualHostCell.energy);
    expectApproxEqual(1.0f, actualHostCell.activity.channels[0]);

    EXPECT_EQ(1, actualConstructedCell.connections.size());
    EXPECT_EQ(1, actualConstructedCell.maxConnections);
    EXPECT_EQ(2, actualConstructedCell.color);
    EXPECT_EQ(4, actualConstructedCell.executionOrderNumber);
    EXPECT_FALSE(actualConstructedCell.underConstruction);
    expectApproxEqual(_parameters.cellNormalEnergy, actualConstructedCell.energy);
}

TEST_F(RibosomeTests, constructSingleCell_separation)
{
    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(RibosomeDescription()
                                          .setGenome({CellDescription()})
                                          .setSeparateConstruction(true)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, actualHostCell.maxConnections);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(0, actualConstructedCell.maxConnections);
    EXPECT_FALSE(actualConstructedCell.underConstruction);
}

TEST_F(RibosomeTests, constructSingleCell_makeSticky)
{
    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(RibosomeDescription().setGenome({CellDescription().setMaxConnections(3)}).setSeparateConstruction(true).setMakeSticky(true)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(1, actualHostCell.maxConnections);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(3, actualConstructedCell.maxConnections);
    EXPECT_FALSE(actualConstructedCell.underConstruction);
}

TEST_F(RibosomeTests, constructSingleCell_singleConstruction)
{
    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(RibosomeDescription().setGenome({CellDescription()}).setSeparateConstruction(true).setSingleConstruction(true)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    auto const& ribosome = std::get<RibosomeDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(ribosome.genome.size(), ribosome.currentGenomePos);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_FALSE(actualConstructedCell.underConstruction);
}
