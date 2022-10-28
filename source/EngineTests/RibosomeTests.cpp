#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
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

    bool hostCellChecked = false;
    bool constructedCellChecked = false;
    for (auto const& cell : actualData.cells) {
        if (cell.id == 1) {
            hostCellChecked = true;
            EXPECT_EQ(1, cell.connections.size());
            EXPECT_EQ(0, std::get<RibosomeDescription>(*cell.cellFunction).currentGenomePos);
            expectApproxEqual(_parameters.cellNormalEnergy * 2, cell.energy);
            expectApproxEqual(1.0f, cell.activity.channels[0]);
        } else {
            constructedCellChecked = true;
            EXPECT_EQ(1, cell.connections.size());
            EXPECT_EQ(1, cell.maxConnections);
            EXPECT_EQ(2, cell.color);
            EXPECT_EQ(4, cell.executionOrderNumber);
            EXPECT_FALSE(cell.underConstruction);
            expectApproxEqual(_parameters.cellNormalEnergy, cell.energy);
        }
    }
    EXPECT_TRUE(hostCellChecked);
    EXPECT_TRUE(constructedCellChecked);
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

    bool hostCellChecked = false;
    bool constructedCellChecked = false;
    for (auto const& cell : actualData.cells) {
        if (cell.id == 1) {
            hostCellChecked = true;
            EXPECT_EQ(0, cell.connections.size());
            EXPECT_EQ(0, cell.maxConnections);
        } else {
            constructedCellChecked = true;
            EXPECT_EQ(0, cell.connections.size());
            EXPECT_EQ(0, cell.maxConnections);
            EXPECT_FALSE(cell.underConstruction);
        }
    }
    EXPECT_TRUE(hostCellChecked);
    EXPECT_TRUE(constructedCellChecked);
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

    bool hostCellChecked = false;
    bool constructedCellChecked = false;
    for (auto const& cell : actualData.cells) {
        if (cell.id == 1) {
            hostCellChecked = true;
            EXPECT_EQ(0, cell.connections.size());
            EXPECT_EQ(1, cell.maxConnections);
        } else {
            constructedCellChecked = true;
            EXPECT_EQ(0, cell.connections.size());
            EXPECT_EQ(3, cell.maxConnections);
            EXPECT_FALSE(cell.underConstruction);
        }
    }
    EXPECT_TRUE(hostCellChecked);
    EXPECT_TRUE(constructedCellChecked);
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

    bool hostCellChecked = false;
    bool constructedCellChecked = false;
    for (auto const& cell : actualData.cells) {
        if (cell.id == 1) {
            hostCellChecked = true;
            EXPECT_EQ(0, cell.connections.size());
            auto const& ribosome = std::get<RibosomeDescription>(*cell.cellFunction);
            EXPECT_EQ(ribosome.genome.size(), ribosome.currentGenomePos);
        } else {
            constructedCellChecked = true;
            EXPECT_EQ(0, cell.connections.size());
            EXPECT_FALSE(cell.underConstruction);
        }
    }
    EXPECT_TRUE(hostCellChecked);
    EXPECT_TRUE(constructedCellChecked);
}
