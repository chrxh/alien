#include <boost/range/adaptors.hpp>

#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "Base/StringHelper.h"
#include "EngineInterface/ParametersEditService.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/LocationHelper.h"

#include "IntegrationTestFramework.h"

class ParametersEditServiceTests_New : public IntegrationTestFramework
{
protected:
    SimulationParameters createTestData(std::vector<LocationType> const& locationTypes)
    {
        SimulationParameters result;
        for (auto const& [orderIndex, locationType] : locationTypes | boost::adaptors::indexed(0)) {
            if (locationType == LocationType::Zone) {
                result.zoneLocationIndex[result.numZones] = orderIndex + 1;
                result.zoneCoreRadius.zoneValues[result.numZones] = toFloat(orderIndex) + 0.5f;
                StringHelper::copy(result.zoneName.zoneValues[result.numZones], sizeof(Char64), "Zone " + std::to_string(result.numZones + 1));
                ++result.numZones;
            } else if (locationType == LocationType::Source) {
                result.sourceLocationIndex[result.numSources] = orderIndex + 1;
                result.sourceCircularRadius.sourceValues[result.numSources] = toFloat(orderIndex) + 0.5f;
                StringHelper::copy(result.sourceName.sourceValues[result.numSources], sizeof(Char64), "Radiation " + std::to_string(result.numSources + 1));
                ++result.numSources;
            }
        }
        return result;
    }

    void checkParameters(SimulationParameters const& parameters, std::vector<LocationType> const& locationTypes)
    {
        std::set<int> locationIndices;
        int lastLocationIndex = 0;
        for (int i = 0; i < parameters.numZones; ++i) {
            EXPECT_FALSE(locationIndices.contains(parameters.zoneLocationIndex[i]));
            locationIndices.insert(parameters.zoneLocationIndex[i]);
            EXPECT_TRUE(lastLocationIndex < parameters.zoneLocationIndex[i]);
            lastLocationIndex = parameters.zoneLocationIndex[i];
        }
        lastLocationIndex = 0;
        for (int i = 0; i < parameters.numSources; ++i) {
            EXPECT_FALSE(locationIndices.contains(parameters.sourceLocationIndex[i]));
            locationIndices.insert(parameters.sourceLocationIndex[i]);
            EXPECT_TRUE(lastLocationIndex < parameters.sourceLocationIndex[i]);
            lastLocationIndex = parameters.sourceLocationIndex[i];
        }
        EXPECT_EQ(parameters.numZones + parameters.numSources, locationIndices.size());
        if (!locationIndices.empty()) {
            EXPECT_EQ(parameters.numZones + parameters.numSources, *std::max_element(locationIndices.begin(), locationIndices.end()));
        }

        for (int i = 0; i < parameters.numZones; ++i) {
            auto locationIndex = parameters.zoneLocationIndex[i];
            EXPECT_EQ(LocationType::Zone, locationTypes.at(locationIndex - 1));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto locationIndex = parameters.sourceLocationIndex[i];
            EXPECT_EQ(LocationType::Source, locationTypes.at(locationIndex - 1));
        }
    }

    void checkParametersAfterInsertion(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int insertedLocationIndex)
    {
        checkParameters(parameters, locationTypes);
        for (int i = 0; i < parameters.numZones; ++i) {
            auto locationIndex = parameters.zoneLocationIndex[i];
            if (locationIndex == insertedLocationIndex) {
                continue;
            }
            auto origLocationIndex = locationIndex < insertedLocationIndex ? locationIndex : locationIndex - 1;
            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.zoneCoreRadius.zoneValues[origArrayIndex], parameters.zoneCoreRadius.zoneValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.zoneName.zoneValues[origArrayIndex], sizeof(Char64), parameters.zoneName.zoneValues[i]));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto locationIndex = parameters.sourceLocationIndex[i];
            if (locationIndex == insertedLocationIndex) {
                continue;
            }
            auto origLocationIndex = locationIndex < insertedLocationIndex ? locationIndex : locationIndex - 1;
            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.sourceCircularRadius.sourceValues[origArrayIndex], parameters.sourceCircularRadius.sourceValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.sourceName.sourceValues[origArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[i]));
        }
    }

    void checkParametersAfterDefaultInsertion(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int insertedLocationIndex)
    {
        checkParametersAfterInsertion(parameters, origParameters, locationTypes, insertedLocationIndex);

        SimulationParameters defaultParameters;
        auto locationType = LocationHelper::getLocationType(insertedLocationIndex, parameters);
        auto insertedArrayIndex = LocationHelper::findLocationArrayIndex(parameters, insertedLocationIndex);

        if (locationType == LocationType::Zone) {
            EXPECT_EQ(defaultParameters.zoneCoreRadius.zoneValues[0], parameters.zoneCoreRadius.zoneValues[insertedArrayIndex]);
            
            Char64 zoneName;
            StringHelper::copy(zoneName, sizeof(Char64), LocationHelper::generateZoneName(origParameters));
            EXPECT_TRUE(StringHelper::compare(zoneName, sizeof(Char64), parameters.zoneName.zoneValues[insertedArrayIndex]));
        } else if (locationType == LocationType::Source) {
            EXPECT_EQ(defaultParameters.sourceCircularRadius.sourceValues[0], parameters.sourceCircularRadius.sourceValues[insertedArrayIndex]);

            Char64 sourceName;
            StringHelper::copy(sourceName, sizeof(Char64), LocationHelper::generateSourceName(origParameters));
            EXPECT_TRUE(StringHelper::compare(sourceName, sizeof(Char64), parameters.sourceName.sourceValues[insertedArrayIndex]));
        }
    }

    void checkParametersAfterCloning(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int insertedLocationIndex)
    {
        checkParametersAfterInsertion(parameters, origParameters, locationTypes, insertedLocationIndex);

        SimulationParameters defaultParameters;
        auto locationType = LocationHelper::getLocationType(insertedLocationIndex, parameters);
        auto insertedArrayIndex = LocationHelper::findLocationArrayIndex(parameters, insertedLocationIndex);
        auto prevArrayIndex = LocationHelper::findLocationArrayIndex(parameters, insertedLocationIndex);

        if (locationType == LocationType::Zone) {
            EXPECT_EQ(parameters.zoneCoreRadius.zoneValues[prevArrayIndex], parameters.zoneCoreRadius.zoneValues[insertedArrayIndex]);

            Char64 zoneName;
            StringHelper::copy(zoneName, sizeof(Char64), LocationHelper::generateZoneName(origParameters));
            EXPECT_TRUE(StringHelper::compare(zoneName, sizeof(Char64), parameters.zoneName.zoneValues[insertedArrayIndex]));
        } else if (locationType == LocationType::Source) {
            EXPECT_EQ(parameters.sourceCircularRadius.sourceValues[prevArrayIndex], parameters.sourceCircularRadius.sourceValues[insertedArrayIndex]);

            Char64 sourceName;
            StringHelper::copy(sourceName, sizeof(Char64), LocationHelper::generateSourceName(origParameters));
            EXPECT_TRUE(StringHelper::compare(sourceName, sizeof(Char64), parameters.sourceName.sourceValues[insertedArrayIndex]));
        }
    }

    void checkParametersAfterDeletion(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int deletedLocationIndex)
    {
        checkParameters(parameters, locationTypes);
        for (int i = 0; i < parameters.numZones; ++i) {
            auto locationIndex = parameters.zoneLocationIndex[i];
            auto origLocationIndex = locationIndex < deletedLocationIndex ? locationIndex : locationIndex + 1;
            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.zoneCoreRadius.zoneValues[origArrayIndex], parameters.zoneCoreRadius.zoneValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.zoneName.zoneValues[origArrayIndex], sizeof(Char64), parameters.zoneName.zoneValues[i]));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto locationIndex = parameters.sourceLocationIndex[i];
            auto origLocationIndex = locationIndex < deletedLocationIndex ? locationIndex : locationIndex + 1;
            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.sourceCircularRadius.sourceValues[origArrayIndex], parameters.sourceCircularRadius.sourceValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.sourceName.sourceValues[origArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[i]));
        }
    }
};

TEST_F(ParametersEditServiceTests_New, cloneZone)
{
    auto origParameters = createTestData({LocationType::Zone, LocationType::Zone, LocationType::Source, LocationType::Source});
    auto parameters = origParameters;
    ParametersEditService::get().cloneLocation(parameters, 1);
    checkParametersAfterCloning(
        parameters, origParameters, {LocationType::Zone, LocationType::Zone, LocationType::Zone, LocationType::Source, LocationType::Source}, 2);
}

TEST_F(ParametersEditServiceTests_New, cloneSource)
{
    auto origParameters = createTestData({LocationType::Zone, LocationType::Zone, LocationType::Source, LocationType::Source});
    auto parameters = origParameters;
    ParametersEditService::get().cloneLocation(parameters, 3);
    checkParametersAfterCloning(
        parameters, origParameters, {LocationType::Zone, LocationType::Zone, LocationType::Source, LocationType::Source, LocationType::Source}, 4);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultZone_empty)
{
    auto origParameters = createTestData({});
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultZone(parameters, 0);
    checkParametersAfterDefaultInsertion(parameters, origParameters, {LocationType::Zone}, 1);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultZone_onlySources)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultZone(parameters, 1);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultZone_base)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultZone(parameters, 0);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultZone_firstZone1)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultZone(parameters, 1);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultZone_firstZone2)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultZone(parameters, 2);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Zone,
            LocationType::Zone,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultZone_middle1)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultZone(parameters, 3);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
        },
        4);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultZone_middle2)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultZone(parameters, 4);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
        },
        5);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultZone_end)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultZone(parameters, 5);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Zone,
        },
        6);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_empty)
{
    auto origParameters = createTestData({});
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 0);
    checkParametersAfterDefaultInsertion(parameters, origParameters, {LocationType::Source}, 1);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_onlyZones)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 1);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_base)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 0);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_firstSource1)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 1);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_firstSource2)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 2);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Source,
            LocationType::Source,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_middle1)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 3);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
        },
        4);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_middle2)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 4);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
        },
        5);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_end)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 5);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Source,
        },
        6);
}

TEST_F(ParametersEditServiceTests_New, deleteZone_afterwardEmpty)
{
    auto origParameters = createTestData({
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 1);
    checkParametersAfterDeletion(parameters, origParameters, {}, 1);
}

TEST_F(ParametersEditServiceTests_New, deleteZone_afterwardOnlySources)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 2);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Source,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, deleteZone_firstZone)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 1);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, deleteZone_middle)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 3);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Source,
            LocationType::Source,
            LocationType::Zone,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, deleteZone_end)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 5);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
        },
        5);
}

TEST_F(ParametersEditServiceTests_New, deleteSource_afterwardEmpty)
{
    auto origParameters = createTestData({
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 1);
    checkParametersAfterDeletion(parameters, origParameters, {}, 1);
}

TEST_F(ParametersEditServiceTests_New, deleteSource_afterwardOnlyZones)
{
    auto origParameters = createTestData({
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 2);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Zone,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, deleteSource_firstSource)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 1);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, deleteSource_middle)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 3);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Zone,
            LocationType::Zone,
            LocationType::Source,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, deleteSource_end)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
        LocationType::Zone,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 5);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Zone,
            LocationType::Source,
            LocationType::Zone,
        },
        5);
}
