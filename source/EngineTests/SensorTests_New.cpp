#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "IntegrationTestFramework.h"

class SensorTests_New : public IntegrationTestFramework
{
public:
    SensorTests_New()
        : IntegrationTestFramework()
    {}

    ~SensorTests_New() = default;
};

TEST_F(SensorTests_New, autoTriggered)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(15)),
    });
    _simulationFacade->setSimulationData(data);

    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_TRUE(actualSensor.signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_FALSE(actualSensor.signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(14);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_TRUE(actualSensor.signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_FALSE(actualSensor.signal.has_value());
    }
}

TEST_F(SensorTests_New, manuallyTriggered_noSignal)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(0)),
    });
    _simulationFacade->setSimulationData(data);

    for(int i = 0; i < 100; ++i) {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_FALSE(actualSensor.signal.has_value());
    }
}

TEST_F(SensorTests_New, manuallyTriggered_signal)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(0)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}).setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
    EXPECT_TRUE(actualSensor.signal.has_value());
}

