#include <QEventLoop>

#include <boost/range/adaptors.hpp>

#include <gtest/gtest.h>

#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Base/ServiceLocator.h"
#include "EngineGpu/EngineGpuBuilderFacade.h"
#include "EngineGpu/EngineGpuData.h"
#include "EngineGpu/SimulationAccessGpu.h"
#include "EngineGpu/SimulationControllerGpu.h"
#include "EngineInterface/DescriptionFactory.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SpaceProperties.h"
#include "IntegrationGpuTestFramework.h"
#include "IntegrationTestFramework.h"
#include "IntegrationTestHelper.h"
#include "Tests/Predicates.h"

class MovementGpuTests : public IntegrationGpuTestFramework
{
public:
    MovementGpuTests()
        : IntegrationGpuTestFramework()
    {}
    virtual ~MovementGpuTests() = default;

    virtual void SetUp()
    {
        _parameters.radiationProb = 0;
        _parameters.cellMaxForce = 2;
        _parameters.cellFusionVelocity = 0;
        _context->setSimulationParameters(_parameters);
    }
};

TEST_F(MovementGpuTests, testTwoRectCollision)
{
    auto const factory = ServiceLocator::getInstance().getService<DescriptionFactory>();

    DataDescription dataBefore;

    auto createRect = [&](auto const& pos, auto const& vel) {
        auto result = factory->createRect(
            DescriptionFactory::CreateRectParameters()
                .size({10, 10}).centerPosition(pos).velocity(vel),
            _context->getNumberGenerator());
        for (auto& cell : *result.cells) {
            cell.maxConnections = cell.connections->size();
        }
        return result;
    };
    dataBefore.addCluster(createRect(QVector2D(10, 10), QVector2D(0.1, 0)));
    dataBefore.addCluster(createRect(QVector2D(30, 10), QVector2D(0, 0)));

    IntegrationTestHelper::updateData(_access, _context, dataBefore);
    IntegrationTestHelper::runSimulation(500, _controller);

    DataDescription dataAfter =
        IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    auto beforeAndAfterCells = IntegrationTestHelper::getBeforeAndAfterCells(dataBefore, dataAfter);
    ASSERT_EQ(2, dataAfter.clusters->size());
    ASSERT_EQ(200, beforeAndAfterCells.size());
    for (auto const& [cellBefore, cellAfter] : beforeAndAfterCells) {
        if (cellBefore.vel->length() > FLOATINGPOINT_MEDIUM_PRECISION) {
            EXPECT_TRUE(cellAfter.vel->length() < 0.03);
        } else {
            EXPECT_TRUE((*cellAfter.vel - QVector2D(0.1, 0)).length() < 0.03);
        }
    }
}

TEST_F(MovementGpuTests, testTwoLineFusion)
{
    auto const factory = ServiceLocator::getInstance().getService<DescriptionFactory>();

    DataDescription dataBefore;

    auto createRect = [&](auto const& pos, auto const& vel) {
        auto result = factory->createRect(
            DescriptionFactory::CreateRectParameters().size({1, 10}).centerPosition(pos).velocity(vel),
            _context->getNumberGenerator());
        return result;
    };
    dataBefore.addCluster(createRect(QVector2D(10, 10), QVector2D(0.1, 0)));
    dataBefore.addCluster(createRect(QVector2D(20, 10), QVector2D(0, 0)));

    IntegrationTestHelper::updateData(_access, _context, dataBefore);
    IntegrationTestHelper::runSimulation(1500, _controller);

    DataDescription dataAfter =
        IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    auto beforeAndAfterCells = IntegrationTestHelper::getBeforeAndAfterCells(dataBefore, dataAfter);
    ASSERT_EQ(1, dataAfter.clusters->size());
    ASSERT_EQ(20, beforeAndAfterCells.size());

    auto firstCellIdOfFirstCluster = dataBefore.clusters->at(0).cells->at(0).id;
    auto secondCellIdOfFirstCluster = dataBefore.clusters->at(0).cells->at(1).id;
    auto firstCellIdOfSecondCluster = dataBefore.clusters->at(1).cells->at(0).id;
    for (auto const& [cellBefore, cellAfter] : beforeAndAfterCells) {
        EXPECT_TRUE((*cellAfter.vel - QVector2D(0.05, 0)).length() < 0.01);
        if (cellBefore.id == firstCellIdOfFirstCluster) {
            EXPECT_EQ(2, cellAfter.connections->size());
            {
                auto connection = *cellAfter.connections->begin();
                EXPECT_EQ(firstCellIdOfSecondCluster, connection.cellId);
                EXPECT_LE(0.9, connection.distance);
            }
            {
                auto connection = *(++cellAfter.connections->begin());
                EXPECT_EQ(secondCellIdOfFirstCluster, connection.cellId);
                EXPECT_LE(0.9, connection.distance);
                EXPECT_EQ(90, connection.angleFromPrevious);
            }
        }
    }
}

TEST_F(MovementGpuTests, testTwoRectFusion)
{
    auto const factory = ServiceLocator::getInstance().getService<DescriptionFactory>();

    DataDescription dataBefore;

    auto createRect = [&](auto const& pos, auto const& vel) {
        auto result = factory->createRect(
            DescriptionFactory::CreateRectParameters().size({10, 10}).centerPosition(pos).velocity(vel),
            _context->getNumberGenerator());
        return result;
    };
    dataBefore.addCluster(createRect(QVector2D(10, 10), QVector2D(0.1, 0)));
    dataBefore.addCluster(createRect(QVector2D(30, 10), QVector2D(0, 0)));

    IntegrationTestHelper::updateData(_access, _context, dataBefore);
    IntegrationTestHelper::runSimulation(3500, _controller);

    DataDescription dataAfter =
        IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    auto beforeAndAfterCells = IntegrationTestHelper::getBeforeAndAfterCells(dataBefore, dataAfter);
    ASSERT_EQ(1, dataAfter.clusters->size());
    ASSERT_EQ(200, beforeAndAfterCells.size());
    for (auto const& [cellBefore, cellAfter] : beforeAndAfterCells) {
        EXPECT_TRUE((*cellAfter.vel - QVector2D(0.05, 0)).length() < 0.005);
    }
}

TEST_F(MovementGpuTests, testRectMovement)
{
    auto const factory = ServiceLocator::getInstance().getService<DescriptionFactory>();

    DataDescription dataBefore;

    auto createRect = [&](auto const& pos, auto const& vel) {
        auto result = factory->createRect(
            DescriptionFactory::CreateRectParameters().size({2, 2}).centerPosition(pos).velocity(vel),
            _context->getNumberGenerator());
        return result;
    };
    dataBefore.addCluster(createRect(QVector2D(10, 10), QVector2D(0.1, 0)));

    IntegrationTestHelper::updateData(_access, _context, dataBefore);
    IntegrationTestHelper::runSimulation(10, _controller);

    DataDescription dataAfter =
        IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    auto beforeAndAfterCells = IntegrationTestHelper::getBeforeAndAfterCells(dataBefore, dataAfter);
    ASSERT_EQ(1, dataAfter.clusters->size());
    ASSERT_EQ(4, dataAfter.clusters->front().cells->size());

    auto firstCellId = dataBefore.clusters->at(0).cells->at(2).id;
    auto secondCellId = dataBefore.clusters->at(0).cells->at(3).id;
    for (auto const& [cellBefore, cellAfter] : beforeAndAfterCells) {
        EXPECT_TRUE((*cellAfter.vel - QVector2D(0.1, 0)).length() < 0.01);
        if (cellBefore.id == firstCellId) {
            EXPECT_TRUE((*cellAfter.pos - QVector2D(11.5, 9.5)).length() < 0.1);
        }
        if (cellBefore.id == secondCellId) {
            EXPECT_TRUE((*cellAfter.pos - QVector2D(11.5, 10.5)).length() < 0.1);
        }
    }
}
