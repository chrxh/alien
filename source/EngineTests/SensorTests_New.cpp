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

TEST_F(SensorTests_New, aboveMinDensity)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.2f)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
    EXPECT_TRUE(actualSensor.signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
    EXPECT_TRUE(actualSensor.signal->channels[1] > 0.2f);
    EXPECT_TRUE(actualSensor.signal->channels[2] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[2] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[3] > -15.0f / 365);
    EXPECT_TRUE(actualSensor.signal->channels[3] < 15.0f / 365);
}

TEST_F(SensorTests_New, belowMinDensity)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.1f)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({2.0f, 2.0f}).width(10).height(10).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
    EXPECT_TRUE(actualSensor.signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
}

TEST_F(SensorTests_New, targetAbove)
{
    _parameters.cellTypeMuscleMovementTowardTargetedObject = false;
    _simulationFacade->setSimulationParameters(_parameters);

        DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.2f)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
    EXPECT_TRUE(actualSensor.signal->channels[1] > 0.2f);
    EXPECT_TRUE(actualSensor.signal->channels[2] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[2] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[3] > 70.0f / 365);
    EXPECT_TRUE(actualSensor.signal->channels[3] < 105.0f / 365);
}

TEST_F(SensorTests_New, targetBelow)
{
    _parameters.cellTypeMuscleMovementTowardTargetedObject = false;
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.2f)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
    EXPECT_TRUE(actualSensor.signal->channels[1] > 0.2f);
    EXPECT_TRUE(actualSensor.signal->channels[2] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[2] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[3] < -70.0f / 365);
    EXPECT_TRUE(actualSensor.signal->channels[3] > -105.0f / 365);
}
