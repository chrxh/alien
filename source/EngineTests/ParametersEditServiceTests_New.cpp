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
            if (locationType == LocationType::Layer) {
                result.layerLocationIndex[result.numLayers] = orderIndex + 1;
                result.layerCoreRadius.layerValues[result.numLayers] = toFloat(orderIndex) + 0.5f;
                StringHelper::copy(result.layerName.layerValues[result.numLayers], sizeof(Char64), "Layer " + std::to_string(result.numLayers + 1));
                ++result.numLayers;
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
        for (int i = 0; i < parameters.numLayers; ++i) {
            EXPECT_FALSE(locationIndices.contains(parameters.layerLocationIndex[i]));
            locationIndices.insert(parameters.layerLocationIndex[i]);
            EXPECT_TRUE(lastLocationIndex < parameters.layerLocationIndex[i]);
            lastLocationIndex = parameters.layerLocationIndex[i];
        }
        lastLocationIndex = 0;
        for (int i = 0; i < parameters.numSources; ++i) {
            EXPECT_FALSE(locationIndices.contains(parameters.sourceLocationIndex[i]));
            locationIndices.insert(parameters.sourceLocationIndex[i]);
            EXPECT_TRUE(lastLocationIndex < parameters.sourceLocationIndex[i]);
            lastLocationIndex = parameters.sourceLocationIndex[i];
        }
        EXPECT_EQ(parameters.numLayers + parameters.numSources, locationIndices.size());
        if (!locationIndices.empty()) {
            EXPECT_EQ(parameters.numLayers + parameters.numSources, *std::max_element(locationIndices.begin(), locationIndices.end()));
        }

        for (int i = 0; i < parameters.numLayers; ++i) {
            auto locationIndex = parameters.layerLocationIndex[i];
            EXPECT_EQ(LocationType::Layer, locationTypes.at(locationIndex - 1));
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
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto locationIndex = parameters.layerLocationIndex[i];
            if (locationIndex == insertedLocationIndex) {
                continue;
            }
            auto origLocationIndex = locationIndex < insertedLocationIndex ? locationIndex : locationIndex - 1;
            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.layerCoreRadius.layerValues[origArrayIndex], parameters.layerCoreRadius.layerValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.layerName.layerValues[origArrayIndex], sizeof(Char64), parameters.layerName.layerValues[i]));
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

        if (locationType == LocationType::Layer) {
            EXPECT_EQ(defaultParameters.layerCoreRadius.layerValues[0], parameters.layerCoreRadius.layerValues[insertedArrayIndex]);
            
            Char64 layerName;
            StringHelper::copy(layerName, sizeof(Char64), LocationHelper::generateLayerName(origParameters));
            EXPECT_TRUE(StringHelper::compare(layerName, sizeof(Char64), parameters.layerName.layerValues[insertedArrayIndex]));
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

        if (locationType == LocationType::Layer) {
            EXPECT_EQ(parameters.layerCoreRadius.layerValues[prevArrayIndex], parameters.layerCoreRadius.layerValues[insertedArrayIndex]);

            Char64 layerName;
            StringHelper::copy(layerName, sizeof(Char64), LocationHelper::generateLayerName(origParameters));
            EXPECT_TRUE(StringHelper::compare(layerName, sizeof(Char64), parameters.layerName.layerValues[insertedArrayIndex]));
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
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto locationIndex = parameters.layerLocationIndex[i];
            auto origLocationIndex = locationIndex < deletedLocationIndex ? locationIndex : locationIndex + 1;
            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.layerCoreRadius.layerValues[origArrayIndex], parameters.layerCoreRadius.layerValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.layerName.layerValues[origArrayIndex], sizeof(Char64), parameters.layerName.layerValues[i]));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto locationIndex = parameters.sourceLocationIndex[i];
            auto origLocationIndex = locationIndex < deletedLocationIndex ? locationIndex : locationIndex + 1;
            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.sourceCircularRadius.sourceValues[origArrayIndex], parameters.sourceCircularRadius.sourceValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.sourceName.sourceValues[origArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[i]));
        }
    }

    void checkParametersAfterMovingUpwards(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int movedLocationIndex)
    {
        checkParameters(parameters, locationTypes);
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto locationIndex = parameters.layerLocationIndex[i];
            auto origLocationIndex = [&] {
                if (locationIndex < movedLocationIndex - 1 || locationIndex > movedLocationIndex) {
                    return locationIndex;
                } else if (locationIndex == movedLocationIndex - 1) {
                    return movedLocationIndex;
                } else if (locationIndex == movedLocationIndex) {
                    return movedLocationIndex - 1;
                } else {
                    CHECK(false);
                }
            }();

            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.layerCoreRadius.layerValues[origArrayIndex], parameters.layerCoreRadius.layerValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.layerName.layerValues[origArrayIndex], sizeof(Char64), parameters.layerName.layerValues[i]));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto locationIndex = parameters.sourceLocationIndex[i];
            auto origLocationIndex = [&] {
                if (locationIndex < movedLocationIndex - 1 || locationIndex > movedLocationIndex) {
                    return locationIndex;
                } else if (locationIndex == movedLocationIndex - 1) {
                    return movedLocationIndex;
                } else if (locationIndex == movedLocationIndex) {
                    return movedLocationIndex - 1;
                } else {
                    CHECK(false);
                }
            }();
            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.sourceCircularRadius.sourceValues[origArrayIndex], parameters.sourceCircularRadius.sourceValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.sourceName.sourceValues[origArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[i]));
        }
    }

    void checkParametersAfterMovingDownwards(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int movedLocationIndex)
    {
        checkParameters(parameters, locationTypes);
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto locationIndex = parameters.layerLocationIndex[i];
            auto origLocationIndex = [&] {
                if (locationIndex < movedLocationIndex || locationIndex > movedLocationIndex + 1) {
                    return locationIndex;
                } else if (locationIndex == movedLocationIndex + 1) {
                    return movedLocationIndex;
                } else if (locationIndex == movedLocationIndex) {
                    return movedLocationIndex + 1;
                } else {
                    CHECK(false);
                }
            }();

            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.layerCoreRadius.layerValues[origArrayIndex], parameters.layerCoreRadius.layerValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.layerName.layerValues[origArrayIndex], sizeof(Char64), parameters.layerName.layerValues[i]));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto locationIndex = parameters.sourceLocationIndex[i];
            auto origLocationIndex = [&] {
                if (locationIndex < movedLocationIndex || locationIndex > movedLocationIndex + 1) {
                    return locationIndex;
                } else if (locationIndex == movedLocationIndex + 1) {
                    return movedLocationIndex;
                } else if (locationIndex == movedLocationIndex) {
                    return movedLocationIndex + 1;
                } else {
                    CHECK(false);
                }
            }();
            auto origArrayIndex = LocationHelper::findLocationArrayIndex(origParameters, origLocationIndex);

            EXPECT_EQ(origParameters.sourceCircularRadius.sourceValues[origArrayIndex], parameters.sourceCircularRadius.sourceValues[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.sourceName.sourceValues[origArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[i]));
        }
    }
};

TEST_F(ParametersEditServiceTests_New, cloneLayer)
{
    auto origParameters = createTestData({LocationType::Layer, LocationType::Layer, LocationType::Source, LocationType::Source});
    auto parameters = origParameters;
    ParametersEditService::get().cloneLocation(parameters, 1);
    checkParametersAfterCloning(
        parameters, origParameters, {LocationType::Layer, LocationType::Layer, LocationType::Layer, LocationType::Source, LocationType::Source}, 2);
}

TEST_F(ParametersEditServiceTests_New, cloneSource)
{
    auto origParameters = createTestData({LocationType::Layer, LocationType::Layer, LocationType::Source, LocationType::Source});
    auto parameters = origParameters;
    ParametersEditService::get().cloneLocation(parameters, 3);
    checkParametersAfterCloning(
        parameters, origParameters, {LocationType::Layer, LocationType::Layer, LocationType::Source, LocationType::Source, LocationType::Source}, 4);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultLayer_empty)
{
    auto origParameters = createTestData({});
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 0);
    checkParametersAfterDefaultInsertion(parameters, origParameters, {LocationType::Layer}, 1);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultLayer_onlySources)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 1);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultLayer_base)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 0);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultLayer_firstLayer1)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 1);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultLayer_firstLayer2)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 2);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Layer,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultLayer_middle1)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 3);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
        },
        4);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultLayer_middle2)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 4);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
        },
        5);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultLayer_end)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 5);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Layer,
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

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_onlyLayers)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 1);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_base)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
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
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_firstSource1)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
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
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_firstSource2)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 2);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Source,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_middle1)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 3);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
        },
        4);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_middle2)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 4);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
        },
        5);
}

TEST_F(ParametersEditServiceTests_New, insertDefaultSource_end)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 5);
    checkParametersAfterDefaultInsertion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Source,
        },
        6);
}

TEST_F(ParametersEditServiceTests_New, deleteLayer_afterwardEmpty)
{
    auto origParameters = createTestData({
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 1);
    checkParametersAfterDeletion(parameters, origParameters, {}, 1);
}

TEST_F(ParametersEditServiceTests_New, deleteLayer_afterwardOnlySources)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
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

TEST_F(ParametersEditServiceTests_New, deleteLayer_firstLayer)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 1);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, deleteLayer_middle)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 3);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Source,
            LocationType::Layer,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, deleteLayer_end)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 5);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
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

TEST_F(ParametersEditServiceTests_New, deleteSource_afterwardOnlyLayers)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 2);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Layer,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, deleteSource_firstSource)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 1);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, deleteSource_middle)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 3);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Layer,
            LocationType::Source,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, deleteSource_end)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 5);
    checkParametersAfterDeletion(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Layer,
        },
        5);
}

TEST_F(ParametersEditServiceTests_New, moveLayerUpwards_afterOtherSource)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationUpwards(parameters, 2);
    checkParametersAfterMovingUpwards(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, moveLayerUpwards_afterOtherLayer)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationUpwards(parameters, 3);
    checkParametersAfterMovingUpwards(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Layer,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, moveLayerUpwards_afterOtherLayerAndSources)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationUpwards(parameters, 3);
    checkParametersAfterMovingUpwards(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Source,
            LocationType::Layer,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, moveSourceUpwards_afterOtherLayer)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationUpwards(parameters, 2);
    checkParametersAfterMovingUpwards(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
        },
        2);
}

TEST_F(ParametersEditServiceTests_New, moveSourceUpwards_afterOtherSource)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationUpwards(parameters, 3);
    checkParametersAfterMovingUpwards(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Source,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, moveSourceUpwards_afterOtherLayerAndSources)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationUpwards(parameters, 3);
    checkParametersAfterMovingUpwards(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Layer,
            LocationType::Source,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, moveLayerDownwards_beforeOtherSource)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationDownwards(parameters, 1);
    checkParametersAfterMovingDownwards(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, moveLayerDownwards_beforeOtherLayer)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationDownwards(parameters, 1);
    checkParametersAfterMovingDownwards(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Layer,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, moveLayerDownwards_beforeOtherLayerAndSources)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationDownwards(parameters, 3);
    checkParametersAfterMovingDownwards(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
            LocationType::Source,
            LocationType::Layer,
            LocationType::Layer,
        },
        3);
}

TEST_F(ParametersEditServiceTests_New, moveSourceDownwards_beforeOtherLayer)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationDownwards(parameters, 1);
    checkParametersAfterMovingDownwards(
        parameters,
        origParameters,
        {
            LocationType::Layer,
            LocationType::Source,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, moveSourceDownwards_beforeOtherSource)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationDownwards(parameters, 1);
    checkParametersAfterMovingDownwards(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Source,
        },
        1);
}

TEST_F(ParametersEditServiceTests_New, moveSourceDownwards_beforeOtherLayerAndSources)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,

    });
    auto parameters = origParameters;
    ParametersEditService::get().moveLocationDownwards(parameters, 3);
    checkParametersAfterMovingDownwards(
        parameters,
        origParameters,
        {
            LocationType::Source,
            LocationType::Layer,
            LocationType::Layer,
            LocationType::Source,
            LocationType::Source,
        },
        3);
}