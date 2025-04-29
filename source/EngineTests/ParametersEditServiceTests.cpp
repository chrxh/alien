#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "EngineInterface/ParametersEditService.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"

#include "IntegrationTestFramework.h"

class ParametersEditServiceTests : public IntegrationTestFramework
{
};


TEST_F(ParametersEditServiceTests, cloneZone)
{
    SimulationParameters parameters;

    parameters.numZones = 2;
    parameters.numSources = 2;

    parameters.zoneLocationIndex[0] = 1;
    parameters.zoneLocationIndex[1] = 2;
    parameters.sourceLocationIndex[0] = 3;
    parameters.sourceLocationIndex[1] = 4;

    parameters.zoneCoreRadius.zoneValues[0] = 1.0f;
    parameters.zoneCoreRadius.zoneValues[1] = 2.0f;

    parameters.sourceCircularRadius.sourceValues[0] = 3.0f;
    parameters.sourceCircularRadius.sourceValues[1] = 4.0f;

    ParametersEditService::get().cloneLocation(parameters, 1);

    EXPECT_EQ(3, parameters.numZones);
    EXPECT_EQ(2, parameters.numSources);

    EXPECT_EQ(1, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(2, parameters.zoneLocationIndex[1]);
    EXPECT_EQ(3, parameters.zoneLocationIndex[2]);

    EXPECT_EQ(4, parameters.sourceLocationIndex[0]);
    EXPECT_EQ(5, parameters.sourceLocationIndex[1]);

    EXPECT_TRUE(approxCompare(1.0f, parameters.zoneCoreRadius.zoneValues[0]));
    EXPECT_TRUE(approxCompare(1.0f, parameters.zoneCoreRadius.zoneValues[1]));
    EXPECT_TRUE(approxCompare(2.0f, parameters.zoneCoreRadius.zoneValues[2]));

    EXPECT_TRUE(approxCompare(3.0f, parameters.sourceCircularRadius.sourceValues[0]));
    EXPECT_TRUE(approxCompare(4.0f, parameters.sourceCircularRadius.sourceValues[1]));
}

TEST_F(ParametersEditServiceTests, cloneSource)
{
    SimulationParameters parameters;

    parameters.numZones = 2;
    parameters.numSources = 2;
    parameters.zoneLocationIndex[0] = 1;
    parameters.zoneLocationIndex[1] = 2;
    parameters.sourceLocationIndex[0] = 3;
    parameters.sourceLocationIndex[1] = 4;
    parameters.zoneCoreRadius.zoneValues[0] = 1.0f;
    parameters.zoneCoreRadius.zoneValues[1] = 2.0f;
    parameters.sourceCircularRadius.sourceValues[0] = 3.0f;
    parameters.sourceCircularRadius.sourceValues[1] = 4.0f;

    ParametersEditService::get().cloneLocation(parameters, 3);

    EXPECT_EQ(2, parameters.numZones);
    EXPECT_EQ(3, parameters.numSources);
    EXPECT_EQ(1, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(2, parameters.zoneLocationIndex[1]);
    EXPECT_EQ(3, parameters.sourceLocationIndex[0]);
    EXPECT_EQ(4, parameters.sourceLocationIndex[1]);
    EXPECT_EQ(5, parameters.sourceLocationIndex[2]);

    EXPECT_TRUE(approxCompare(1.0f, parameters.zoneCoreRadius.zoneValues[0]));
    EXPECT_TRUE(approxCompare(2.0f, parameters.zoneCoreRadius.zoneValues[1]));

    EXPECT_TRUE(approxCompare(3.0f, parameters.sourceCircularRadius.sourceValues[0]));
    EXPECT_TRUE(approxCompare(3.0f, parameters.sourceCircularRadius.sourceValues[1]));
    EXPECT_TRUE(approxCompare(4.0f, parameters.sourceCircularRadius.sourceValues[2]));
}

TEST_F(ParametersEditServiceTests, insertZoneAfterBase)
{
    SimulationParameters parameters;
    parameters.numSources = 1;
    parameters.sourceLocationIndex[0] = 1;

    ParametersEditService::get().insertZone(parameters, 0);

    EXPECT_EQ(1, parameters.numZones);
    EXPECT_EQ(1, parameters.numSources);
    EXPECT_EQ(1, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(2, parameters.sourceLocationIndex[0]);
}

TEST_F(ParametersEditServiceTests, insertZoneAfterZone)
{
    SimulationParameters parameters;
    parameters.numZones = 2;
    parameters.numSources = 1;
    parameters.zoneLocationIndex[0] = 1;
    parameters.zoneLocationIndex[1] = 3;
    parameters.sourceLocationIndex[0] = 2;

    ParametersEditService::get().insertZone(parameters, 1);

    EXPECT_EQ(3, parameters.numZones);
    EXPECT_EQ(1, parameters.numSources);
    EXPECT_EQ(1, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(2, parameters.zoneLocationIndex[1]);
    EXPECT_EQ(4, parameters.zoneLocationIndex[2]);
    EXPECT_EQ(3, parameters.sourceLocationIndex[0]);
}

TEST_F(ParametersEditServiceTests, insertZoneAfterSource)
{
    SimulationParameters parameters;
    parameters.numZones = 1;
    parameters.numSources = 2;
    parameters.zoneLocationIndex[0] = 2;
    parameters.sourceLocationIndex[0] = 1;
    parameters.sourceLocationIndex[1] = 3;

    ParametersEditService::get().insertZone(parameters, 1);

    EXPECT_EQ(2, parameters.numZones);
    EXPECT_EQ(2, parameters.numSources);
    EXPECT_EQ(2, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(3, parameters.zoneLocationIndex[1]);
    EXPECT_EQ(1, parameters.sourceLocationIndex[0]);
    EXPECT_EQ(4, parameters.sourceLocationIndex[1]);
}
