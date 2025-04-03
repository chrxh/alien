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
        return 1.0f + _parameters.constructorAdditionalOffspringDistance;
    }

    float getConstructorEnergy() const { return _parameters.normalCellEnergy.value[0] * 2.5f; }
};

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_upperSide)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.5f}).livingState(LivingState_UnderConstruction),
    });
    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(1, 2);
    data.addConnection(2, 3, cell3_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    ASSERT_EQ(4, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());

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
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 10.5f}).livingState(LivingState_UnderConstruction),
    });
    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, -60.0f);
    data.addConnection(1, 2);
    data.addConnection(2, 3, cell3_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());

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
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.5f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(4).pos({10.0f + getOffspringDistance(), 10.5f}).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f + 180.0f);
    data.addConnection(2, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualUpperConstructedCell = getCell(actualData, 3);
    auto actualLowerConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(3, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualUpperConstructedCell._connections.size());
    ASSERT_EQ(1, actualLowerConstructedCell._connections.size());

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
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(2)}));

    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f).livingState(LivingState_UnderConstruction),
        CellDescription().id(4).pos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    auto cell3_refPos = data._cells.at(1)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = data._cells.at(2)._pos + Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(4, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(3, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

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
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(2)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({458.20f, 239.23f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({456.40f, 238.88f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos({455.96f, 239.75f})
            .livingState(LivingState_UnderConstruction),
        CellDescription().id(4).pos({456.07f, 240.77f}).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    auto cell3_refPos = data._cells.at(1)._pos + Math::rotateClockwise(data._cells.at(0)._pos - data._cells.at(1)._pos, 120.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = data._cells.at(2)._pos + Math::rotateClockwise(data._cells.at(1)._pos - data._cells.at(2)._pos, 120.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(4, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(3, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

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
            .header(GenomeHeaderDescription().separateConstruction(false))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));

    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription()
            .id(3)
            .pos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f)
            .livingState(LivingState_UnderConstruction),
        CellDescription().id(4).pos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    auto cell3_refPos = getCell(data, 2)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = getCell(data, 3)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto origPrevPrevConstructedCell = getCell(data, 3);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_EQ(origPrevPrevConstructedCell._connections, actualPrevPrevConstructedCell._connections);

    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_90degAlignment)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false).angleAlignment(ConstructorAngleAlignment_90))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(4).pos({10.0f + getOffspringDistance() - 1.0f, 9.0f - 0.2f}).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = getCell(data, 3)._pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

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
            .header(GenomeHeaderDescription().separateConstruction(false).angleAlignment(ConstructorAngleAlignment_90))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(0)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(2, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(1, actualPrevPrevConstructedCell._connections.size());

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
            .header(GenomeHeaderDescription().separateConstruction(false).angleAlignment(ConstructorAngleAlignment_90))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 10.0f - 0.5f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(4).pos({10.0f + getOffspringDistance(), 10.0f - 1.0f}).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = getCell(data, 3)._pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

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

TEST_F(ConstructorTests_New, constructFurtherCell_onSpike)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false))
            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(0)}));

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(3)),
        CellDescription().id(3).pos({11.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualOtherCell = getCell(actualData, 1);
    auto actualHostCell = getCell(actualData, 2);
    auto actualPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    ASSERT_EQ(1, actualOtherCell._connections.size());
    ASSERT_EQ(2, actualHostCell._connections.size());
    ASSERT_EQ(2, actualConstructedCell._connections.size());
    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualOtherCell, actualHostCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualHostCell, actualOtherCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests_New, finishCreature_angleToFront_upperSide)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(true).frontAngle(45.0f))
            .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy())}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(2).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
    auto prevConstructedCell = getCell(actualData, 2);
    auto prevPrevConstructedCell = getCell(actualData, 3);

    EXPECT_EQ(LivingState_Ready, actualConstructedCell._livingState);
    EXPECT_TRUE(approxCompare(45.0f, actualConstructedCell._angleToFront));

    EXPECT_EQ(LivingState_Ready, prevConstructedCell._livingState);
    EXPECT_TRUE(approxCompare(135.0f, prevConstructedCell._angleToFront));

    EXPECT_EQ(LivingState_Ready, prevPrevConstructedCell._livingState);
    EXPECT_TRUE(approxCompare(-45.0f, prevPrevConstructedCell._angleToFront));
}

TEST_F(ConstructorTests_New, finishCreature_angleToFront_lowerSide)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(true).frontAngle(45.0f))
            .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy())}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(2).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).livingState(LivingState_UnderConstruction),
        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 11.0f}).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
    auto prevConstructedCell = getCell(actualData, 2);
    auto prevPrevConstructedCell = getCell(actualData, 3);

    EXPECT_EQ(LivingState_Ready, actualConstructedCell._livingState);
    EXPECT_TRUE(approxCompare(45.0f, actualConstructedCell._angleToFront));

    EXPECT_EQ(LivingState_Ready, prevConstructedCell._livingState);
    EXPECT_TRUE(approxCompare(-45.0f, prevConstructedCell._angleToFront));

    EXPECT_EQ(LivingState_Ready, prevPrevConstructedCell._livingState);
    EXPECT_TRUE(approxCompare(135.0f, prevPrevConstructedCell._angleToFront));
}

TEST_F(ConstructorTests_New, finishBodyPart_angleToFront_leftSide)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false).frontAngle(45.0f))
            .cells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}),
        CellDescription()
            .id(2)
            .pos({9.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(0).autoTriggerInterval(1).genome(genome))
            .angleToFront(-45.0f),
        CellDescription().id(3).pos({9.0f, 11.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_EQ(LivingState_Ready, actualConstructedCell._livingState);
    EXPECT_TRUE(approxCompare(-90.0f, actualConstructedCell._angleToFront));
}

TEST_F(ConstructorTests_New, finishBodyPart_angleToFront_rightSide)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(false).frontAngle(-45.0f)).cells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({8.0f, 10.0f}),
        CellDescription()
            .id(2)
            .pos({9.0f, 10.0f})
            .energy(getConstructorEnergy())
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(0).autoTriggerInterval(1).genome(genome))
            .angleToFront(45.0f),
        CellDescription().id(3).pos({9.0f, 11.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_EQ(LivingState_Ready, actualConstructedCell._livingState);
    EXPECT_TRUE(approxCompare(90.0f, actualConstructedCell._angleToFront));
}
