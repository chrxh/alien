#include <gtest/gtest.h>

#include "Base/Math.h"
#include "Base/NumberGenerator.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeConstants.h"
#include "EngineInterface/GenomeDescriptionService.h"
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
        return 1.0f + _parameters.cellFunctionConstructorAdditionalOffspringDistance;
    }

    float getConstructorEnergy() const { return _parameters.cellNormalEnergy[0] * 3; }
};


TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_upperSide)
{
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setActivationMode(1).setGenome(genome).setLastConstructedCellId(2)),
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

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(3, actualConstructedCell.connections.size());
    EXPECT_EQ(2, actualPrevConstructedCell.connections.size());
    EXPECT_EQ(2, actualPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell).angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_bottomSide)
{
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setActivationMode(1).setGenome(genome).setLastConstructedCellId(2)),
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

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(3, actualConstructedCell.connections.size());
    EXPECT_EQ(2, actualPrevConstructedCell.connections.size());
    EXPECT_EQ(2, actualPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualHostCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualPrevConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell).angleFromPrevious));
}

TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_bothSides)
{
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setActivationMode(1).setGenome(genome).setLastConstructedCellId(2)),
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

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(3, actualConstructedCell.connections.size());
    EXPECT_EQ(3, actualPrevConstructedCell.connections.size());
    EXPECT_EQ(2, actualUpperConstructedCell.connections.size());
    EXPECT_EQ(1, actualLowerConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualUpperConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevConstructedCell, actualConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualUpperConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualPrevConstructedCell, actualLowerConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualUpperConstructedCell, actualPrevConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualUpperConstructedCell, actualConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualLowerConstructedCell, actualPrevConstructedCell).angleFromPrevious));
}


TEST_F(ConstructorTests_New, constructFurtherCell_connectToExistingCell_threeCellsWithSmallAngles)
{
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(getConstructorEnergy())
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setActivationMode(1).setGenome(genome).setLastConstructedCellId(2)),
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

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(4, actualConstructedCell.connections.size());
    EXPECT_EQ(2, actualPrevConstructedCell.connections.size());
    EXPECT_EQ(3, actualPrevPrevConstructedCell.connections.size());
    EXPECT_EQ(2, actualPrevPrevPrevConstructedCell.connections.size());

    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell).angleFromPrevious));

    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell).angleFromPrevious));
    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell).angleFromPrevious));
}
