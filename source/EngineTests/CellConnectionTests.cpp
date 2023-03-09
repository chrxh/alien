#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class CellConnectionTests : public IntegrationTestFramework
{
public:
    CellConnectionTests()
        : IntegrationTestFramework()
    {}

    ~CellConnectionTests() = default;
};

TEST_F(CellConnectionTests, decay)
{
    auto origData =
        DescriptionHelper::createRect(DescriptionHelper::CreateRectParameters().width(10).height(10).energy(_parameters.baseValues.cellMinEnergy[0] / 2));

    _simController->setSimulationData(origData);
    _simController->calcSingleTimestep();

    auto data = _simController->getSimulationData();
    EXPECT_EQ(0, data.cells.size());
    EXPECT_EQ(100, data.particles.size());
}
