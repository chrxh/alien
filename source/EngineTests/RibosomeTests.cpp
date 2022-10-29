#include <gtest/gtest.h>

#include "Base/Math.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class RibosomeTests : public IntegrationTestFramework
{
public:
    RibosomeTests()
        : IntegrationTestFramework()
    {}

    ~RibosomeTests() = default;
};

TEST_F(RibosomeTests, noEnergy)
{
    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy * 2 - 1.0f)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(RibosomeDescription().setGenome({CellDescription()}).setSeparateConstruction(false)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<RibosomeDescription>(*actualHostCell.cellFunction).currentGenomePos);
    expectApproxEqual(_parameters.cellNormalEnergy * 2 - 1.0f, actualHostCell.energy);
    expectApproxEqual(0.0f, actualHostCell.activity.channels[0]);
}

TEST_F(RibosomeTests, alreadyFinished)
{
    DataDescription data;

    auto ribosome = RibosomeDescription().setGenome({CellDescription()}).setSingleConstruction(true);
    ribosome.setCurrentGenomePos(ribosome.genome.size());

    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ribosome));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualRibosome = std::get<RibosomeDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(actualRibosome.genome.size(), actualRibosome.currentGenomePos);
    expectApproxEqual(0.0f, actualHostCell.activity.channels[0]);
}

TEST_F(RibosomeTests, manualConstruction_noInputActivity)
{
    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(
                         RibosomeDescription().setMode(Enums::ConstructionMode_Manual).setGenome({CellDescription()})));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<RibosomeDescription>(*actualHostCell.cellFunction).currentGenomePos);
    expectApproxEqual(_parameters.cellNormalEnergy * 3, actualHostCell.energy);
    expectApproxEqual(0.0f, actualHostCell.activity.channels[0]);
}


TEST_F(RibosomeTests, constructSingleCell_noSeparation)
{
    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setPos({10.0f, 10.0f})
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
    expectApproxEqual(1.6f, Math::length(actualHostCell.pos - actualConstructedCell.pos));
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

TEST_F(RibosomeTests, constructSingleCell_manualConstruction)
{
    DataDescription data;
    data.addCells({
       CellDescription()
            .setId(1)
             .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(RibosomeDescription().setMode(Enums::ConstructionMode_Manual).setGenome({CellDescription()})),
        CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setEnergy(100)
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell.connections.size());

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_FALSE(actualConstructedCell.underConstruction);

    expectApproxEqual(10.0f - 1.6f, actualConstructedCell.pos.x);
    expectApproxEqual(10.0f, actualConstructedCell.pos.y);
}
