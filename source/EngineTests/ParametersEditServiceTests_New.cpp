#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "EngineInterface/ParametersEditService.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"

#include "IntegrationTestFramework.h"

class ParametersEditServiceTests_New : public IntegrationTestFramework
{
};


TEST_F(ParametersEditServiceTests_New, cloneZone)
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

TEST_F(ParametersEditServiceTests_New, cloneSource)
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

TEST_F(ParametersEditServiceTests_New, insertDefaultZoneAfterBase)
{
    SimulationParameters parameters;
    parameters.numSources = 1;
    parameters.sourceLocationIndex[0] = 1;

    ParametersEditService::get().insertDefaultZone(parameters, 0);

    EXPECT_EQ(1, parameters.numZones);
    EXPECT_EQ(1, parameters.numSources);
    EXPECT_EQ(1, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(2, parameters.sourceLocationIndex[0]);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSourceAfterBase)
{
    SimulationParameters parameters;
    parameters.numZones = 1;
    parameters.zoneLocationIndex[0] = 1;

    ParametersEditService::get().insertDefaultSource(parameters, 0);

    EXPECT_EQ(1, parameters.numZones);
    EXPECT_EQ(1, parameters.numSources);
    EXPECT_EQ(2, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(1, parameters.sourceLocationIndex[0]);
}


TEST_F(ParametersEditServiceTests_New, insertDefaultZoneAfterZone)
{
    SimulationParameters parameters;
    parameters.numZones = 2;
    parameters.numSources = 1;
    parameters.zoneLocationIndex[0] = 1;
    parameters.zoneLocationIndex[1] = 3;
    parameters.sourceLocationIndex[0] = 2;
    parameters.friction.zoneValues[0].value = 0.05f;
    parameters.friction.zoneValues[0].enabled = true;

    auto origParameters = parameters;

    ParametersEditService::get().insertDefaultZone(parameters, 1);

    EXPECT_EQ(3, parameters.numZones);
    EXPECT_EQ(1, parameters.numSources);
    EXPECT_EQ(1, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(2, parameters.zoneLocationIndex[1]);
    EXPECT_EQ(4, parameters.zoneLocationIndex[2]);
    EXPECT_EQ(3, parameters.sourceLocationIndex[0]);

    SimulationParameters defaultParameters;
    EXPECT_TRUE(approxCompare(origParameters.friction.zoneValues[0].value, parameters.friction.zoneValues[0].value));
    EXPECT_EQ(origParameters.friction.zoneValues[0].enabled, parameters.friction.zoneValues[0].enabled);
    EXPECT_TRUE(approxCompare(defaultParameters.friction.zoneValues[0].value, parameters.friction.zoneValues[1].value));
    EXPECT_EQ(defaultParameters.friction.zoneValues[0].enabled, parameters.friction.zoneValues[1].enabled);
    EXPECT_TRUE(approxCompare(origParameters.friction.zoneValues[1].value, parameters.friction.zoneValues[2].value));
    EXPECT_EQ(origParameters.friction.zoneValues[1].enabled, parameters.friction.zoneValues[2].enabled);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSourceAfterZone)
{
    SimulationParameters parameters;
    parameters.numZones = 1;
    parameters.numSources = 2;
    parameters.sourceLocationIndex[0] = 1;
    parameters.sourceLocationIndex[1] = 3;
    parameters.zoneLocationIndex[0] = 2;
    parameters.sourceCircularRadius.sourceValues[0] = 0.05f;

    auto origParameters = parameters;

    ParametersEditService::get().insertDefaultSource(parameters, 1);

    EXPECT_EQ(1, parameters.numZones);
    EXPECT_EQ(3, parameters.numSources);
    EXPECT_EQ(1, parameters.sourceLocationIndex[0]);
    EXPECT_EQ(2, parameters.sourceLocationIndex[1]);
    EXPECT_EQ(4, parameters.sourceLocationIndex[2]);
    EXPECT_EQ(3, parameters.zoneLocationIndex[0]);

    SimulationParameters defaultParameters;
    EXPECT_TRUE(approxCompare(origParameters.sourceCircularRadius.sourceValues[0], parameters.sourceCircularRadius.sourceValues[0]));
    EXPECT_TRUE(approxCompare(defaultParameters.sourceCircularRadius.sourceValues[0], parameters.sourceCircularRadius.sourceValues[1]));
    EXPECT_TRUE(approxCompare(origParameters.sourceCircularRadius.sourceValues[1], parameters.sourceCircularRadius.sourceValues[2]));
}

TEST_F(ParametersEditServiceTests_New, insertDefaultZoneAfterSource)
{
    SimulationParameters parameters;
    parameters.numZones = 1;
    parameters.numSources = 2;
    parameters.zoneLocationIndex[0] = 2;
    parameters.sourceLocationIndex[0] = 1;
    parameters.sourceLocationIndex[1] = 3;

    ParametersEditService::get().insertDefaultZone(parameters, 1);

    EXPECT_EQ(2, parameters.numZones);
    EXPECT_EQ(2, parameters.numSources);
    EXPECT_EQ(2, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(3, parameters.zoneLocationIndex[1]);
    EXPECT_EQ(1, parameters.sourceLocationIndex[0]);
    EXPECT_EQ(4, parameters.sourceLocationIndex[1]);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSourceAfterSource)
{
    SimulationParameters parameters;
    parameters.numZones = 2;
    parameters.numSources = 1;
    parameters.sourceLocationIndex[0] = 2;
    parameters.zoneLocationIndex[0] = 1;
    parameters.zoneLocationIndex[1] = 3;

    ParametersEditService::get().insertDefaultSource(parameters, 1);

    EXPECT_EQ(2, parameters.numZones);
    EXPECT_EQ(2, parameters.numSources);
    EXPECT_EQ(2, parameters.sourceLocationIndex[0]);
    EXPECT_EQ(3, parameters.sourceLocationIndex[1]);
    EXPECT_EQ(1, parameters.zoneLocationIndex[0]);
    EXPECT_EQ(4, parameters.zoneLocationIndex[1]);
}
