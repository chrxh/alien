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

class CollisionGpuTests : public IntegrationGpuTestFramework
{
public:
    CollisionGpuTests()
        : IntegrationGpuTestFramework()
    {}
    virtual ~CollisionGpuTests() = default;
};

TEST_F(CollisionGpuTests, testTwoRectCollision)
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
    IntegrationTestHelper::runSimulation(1000, _controller);

    DataDescription dataAfter =
        IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    auto beforeAndAfterCells = IntegrationTestHelper::getBeforeAndAfterCells(dataBefore, dataAfter);
    ASSERT_EQ(200, beforeAndAfterCells.size());
    for (auto const& [cellBefore, cellAfter] : beforeAndAfterCells) {
        if (cellBefore.vel->length() > FLOATINGPOINT_MEDIUM_PRECISION) {
            EXPECT_TRUE(cellAfter.vel->length() < 0.03);
        } else {
            EXPECT_TRUE((*cellAfter.vel - QVector2D(0.1, 0)).length() < 0.03);
        }
    }
}

