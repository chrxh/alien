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
    _parameters.baseValues.radiationAbsorption[0] = 0;
    _simController->setSimulationParameters(_parameters);
    auto origData =
        DescriptionHelper::createRect(DescriptionHelper::CreateRectParameters().width(10).height(10).energy(_parameters.baseValues.cellMinEnergy[0] / 2));

    _simController->setSimulationData(origData);
    for (int i = 0; i < 1000; ++i) {
        _simController->calcTimesteps(1);
    }

    auto data = _simController->getSimulationData();
    EXPECT_EQ(0, data.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(origData)));
}

