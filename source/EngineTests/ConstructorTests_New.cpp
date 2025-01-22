#include <gtest/gtest.h>

#include "Base/Math.h"
#include "Base/NumberGenerator.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeConstants.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/SimulationFacade.h"

#include "IntegrationTestFramework.h"

class ConstructorTests_New : public IntegrationTestFramework
{
public:
    ConstructorTests_New()
        : IntegrationTestFramework()
    {}

    ~ConstructorTests_New() = default;

protected:
    float getOffspringDistance() const
    {
        return 1.0f + _parameters.cellTypeConstructorAdditionalOffspringDistance;
    }

    float getConstructorEnergy() const { return _parameters.cellNormalEnergy[0] * 3; }
};

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_upperSide)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellTypeData(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().setId(2).setPos({10.0f + getOffspringDistance(), 10.0f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(3).setPos({10.0f + getOffspringDistance(), 9.5f}).setLivingState(LivingState_UnderConstruction),
    });
    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(1, 2);
    data.addConnection(2, 3, cell3_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    ASSERT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_bottomSide)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellTypeData(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().setId(2).setPos({10.0f + getOffspringDistance(), 10.0f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(3).setPos({10.0f + getOffspringDistance(), 10.5f}).setLivingState(LivingState_UnderConstruction),
    });
    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, -60.0f);
    data.addConnection(1, 2);
    data.addConnection(2, 3, cell3_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    ASSERT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_bothSidesPresent)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellTypeData(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().setId(2).setPos({10.0f + getOffspringDistance(), 10.0f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(3).setPos({10.0f + getOffspringDistance(), 9.5f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos({10.0f + getOffspringDistance(), 10.5f}).setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f + 180.0f);
    data.addConnection(2, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualUpperConstructedCell = getCell(actualData, 3);
    auto actualLowerConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(3, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualUpperConstructedCell.connections.size());
    ASSERT_EQ(1, actualLowerConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualUpperConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualUpperConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualPrevConstructedCell, actualLowerConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualUpperConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualUpperConstructedCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualLowerConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
}


TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_threeCellsWithSmallAngles)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellTypeData(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().setId(2).setPos({10.0f + getOffspringDistance(), 10.0f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(3).setPos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    auto cell3_refPos = data.cells.at(1).pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = data.cells.at(2).pos + Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(4, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(3, actualPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_threeCellsWithSmallAngles2)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription().setNumRequiredAdditionalConnections(2)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({458.20f, 239.23f})
            .setEnergy(getConstructorEnergy())
            .setCellTypeData(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().setId(2).setPos({456.40f, 238.88f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(3).setPos({455.96f, 239.75f})
            .setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos({456.07f, 240.77f}).setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    auto cell3_refPos = data.cells.at(1).pos + Math::rotateClockwise(data.cells.at(0).pos - data.cells.at(1).pos, 120.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = data.cells.at(2).pos + Math::rotateClockwise(data.cells.at(1).pos - data.cells.at(2).pos, 120.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(4, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(3, actualPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_threeCellsWithSmallAngles_restrictAdditionalConnections)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
            .setCells({CellGenomeDescription(), CellGenomeDescription().setNumRequiredAdditionalConnections(1)}));

    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellTypeData(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().setId(2).setPos({10.0f + getOffspringDistance(), 10.0f}).setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(3)
            .setPos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f)
            .setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    auto cell3_refPos = getCell(data, 2).pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = getCell(data, 3).pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto origPrevPrevConstructedCell = getCell(data, 3);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_EQ(origPrevPrevConstructedCell.connections, actualPrevPrevConstructedCell.connections);

    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_90degAlignment)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setSeparateConstruction(false).setAngleAlignment(ConstructorAngleAlignment_90))
            .setCells({CellGenomeDescription(), CellGenomeDescription().setNumRequiredAdditionalConnections(1)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellTypeData(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().setId(2).setPos({10.0f + getOffspringDistance(), 10.0f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(3).setPos({10.0f + getOffspringDistance(), 9.0f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos({10.0f + getOffspringDistance() - 1.0f, 9.0f - 0.2f}).setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = getCell(data, 3).pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToNoExistingCells_90degAlignment)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setSeparateConstruction(false).setAngleAlignment(ConstructorAngleAlignment_90))
            .setCells({CellGenomeDescription(), CellGenomeDescription().setNumRequiredAdditionalConnections(0)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellTypeData(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().setId(2).setPos({10.0f + getOffspringDistance(), 10.0f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(3).setPos({10.0f + getOffspringDistance(), 9.0f}).setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    ASSERT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(2, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(1, actualPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToCellWithAngleSpace_90degAlignment)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setSeparateConstruction(false).setAngleAlignment(ConstructorAngleAlignment_90))
            .setCells({CellGenomeDescription(), CellGenomeDescription().setNumRequiredAdditionalConnections(1)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellTypeData(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().setId(2).setPos({10.0f + getOffspringDistance(), 10.0f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(3).setPos({10.0f + getOffspringDistance(), 10.0f - 0.5f}).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos({10.0f + getOffspringDistance(), 10.0f - 1.0f}).setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = getCell(data, 3).pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}


