#include <boost/range/adaptors.hpp>

#include <gtest/gtest.h>

#include <Base/Definitions.h>
#include <Base/StringHelper.h>

#include <EngineInterface/DescEditService.h>
#include <EngineInterface/Descs.h>
#include <EngineInterface/LocationEditService.h>
#include <EngineInterface/ParametersEditService.h>

class ParametersEditServiceTests : public ::testing::Test
{
protected:
    SimulationParameters createTestData(std::vector<LocationType> const& locationTypes)
    {
        SimulationParameters result;
        for (auto const& [orderIndex, locationType] : locationTypes | boost::adaptors::indexed(0)) {
            if (locationType == LocationType::Layer) {
                result.layerOrderNumbers[result.numLayers] = orderIndex + 1;
                result.layerCoreRadius.layerValues[result.numLayers] = toFloat(orderIndex) + 0.5f;
                StringHelper::copy(result.layerName.layerValues[result.numLayers], sizeof(Char64), "Layer " + std::to_string(result.numLayers + 1));
                ++result.numLayers;
            } else if (locationType == LocationType::Source) {
                result.sourceOrderNumbers[result.numSources] = orderIndex + 1;
                result.sourceCircularRadius.sourceValues[result.numSources] = toFloat(orderIndex) + 0.5f;
                StringHelper::copy(result.sourceName.sourceValues[result.numSources], sizeof(Char64), "Radiation " + std::to_string(result.numSources + 1));
                ++result.numSources;
            }
        }
        LocationEditService::get().assignLocationIds(result);
        return result;
    }

    void checkParameters(SimulationParameters const& parameters, std::vector<LocationType> const& locationTypes)
    {
        std::set<int> locationIndices;
        int lastOrderNumber = 0;
        for (int i = 0; i < parameters.numLayers; ++i) {
            EXPECT_FALSE(locationIndices.contains(parameters.layerOrderNumbers[i]));
            locationIndices.insert(parameters.layerOrderNumbers[i]);
            EXPECT_TRUE(lastOrderNumber < parameters.layerOrderNumbers[i]);
            lastOrderNumber = parameters.layerOrderNumbers[i];
        }
        lastOrderNumber = 0;
        for (int i = 0; i < parameters.numSources; ++i) {
            EXPECT_FALSE(locationIndices.contains(parameters.sourceOrderNumbers[i]));
            locationIndices.insert(parameters.sourceOrderNumbers[i]);
            EXPECT_TRUE(lastOrderNumber < parameters.sourceOrderNumbers[i]);
            lastOrderNumber = parameters.sourceOrderNumbers[i];
        }
        EXPECT_EQ(parameters.numLayers + parameters.numSources, locationIndices.size());
        if (!locationIndices.empty()) {
            EXPECT_EQ(parameters.numLayers + parameters.numSources, *std::max_element(locationIndices.begin(), locationIndices.end()));
        }

        std::set<int> locationIds;
        for (int i = 0; i < parameters.numLayers; ++i) {
            locationIds.insert(parameters.layerIds[i]);
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            locationIds.insert(parameters.sourceIds[i]);
        }
        EXPECT_EQ(parameters.numLayers + parameters.numSources, locationIds.size());
        EXPECT_FALSE(locationIds.contains(0));

        for (int i = 0; i < parameters.numLayers; ++i) {
            auto orderNumber = parameters.layerOrderNumbers[i];
            EXPECT_EQ(LocationType::Layer, locationTypes.at(orderNumber - 1));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto orderNumber = parameters.sourceOrderNumbers[i];
            EXPECT_EQ(LocationType::Source, locationTypes.at(orderNumber - 1));
        }
    }

    void checkParametersAfterInsertion(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int insertedOrderNumber)
    {
        checkParameters(parameters, locationTypes);
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto orderNumber = parameters.layerOrderNumbers[i];
            if (orderNumber == insertedOrderNumber) {
                continue;
            }
            auto origOrderNumber = orderNumber < insertedOrderNumber ? orderNumber : orderNumber - 1;
            auto origArrayIndex = LocationEditService::get().findLocationArrayIndex(origParameters, origOrderNumber);

            EXPECT_EQ(origParameters.layerCoreRadius.layerValues[origArrayIndex], parameters.layerCoreRadius.layerValues[i]);
            EXPECT_EQ(origParameters.layerIds[origArrayIndex], parameters.layerIds[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.layerName.layerValues[origArrayIndex], sizeof(Char64), parameters.layerName.layerValues[i]));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto orderNumber = parameters.sourceOrderNumbers[i];
            if (orderNumber == insertedOrderNumber) {
                continue;
            }
            auto origOrderNumber = orderNumber < insertedOrderNumber ? orderNumber : orderNumber - 1;
            auto origArrayIndex = LocationEditService::get().findLocationArrayIndex(origParameters, origOrderNumber);

            EXPECT_EQ(origParameters.sourceCircularRadius.sourceValues[origArrayIndex], parameters.sourceCircularRadius.sourceValues[i]);
            EXPECT_EQ(origParameters.sourceIds[origArrayIndex], parameters.sourceIds[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.sourceName.sourceValues[origArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[i]));
        }
    }

    void checkParametersAfterDefaultInsertion(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int insertedOrderNumber)
    {
        checkParametersAfterInsertion(parameters, origParameters, locationTypes, insertedOrderNumber);

        SimulationParameters defaultParameters;
        auto locationType = LocationEditService::get().getLocationType(insertedOrderNumber, parameters);
        auto insertedArrayIndex = LocationEditService::get().findLocationArrayIndex(parameters, insertedOrderNumber);

        if (locationType == LocationType::Layer) {
            EXPECT_EQ(defaultParameters.layerCoreRadius.layerValues[0], parameters.layerCoreRadius.layerValues[insertedArrayIndex]);

            Char64 layerName;
            StringHelper::copy(layerName, sizeof(Char64), LocationEditService::get().generateLayerName(origParameters));
            EXPECT_TRUE(StringHelper::compare(layerName, sizeof(Char64), parameters.layerName.layerValues[insertedArrayIndex]));
        } else if (locationType == LocationType::Source) {
            EXPECT_EQ(defaultParameters.sourceCircularRadius.sourceValues[0], parameters.sourceCircularRadius.sourceValues[insertedArrayIndex]);

            Char64 sourceName;
            StringHelper::copy(sourceName, sizeof(Char64), LocationEditService::get().generateSourceName(origParameters));
            EXPECT_TRUE(StringHelper::compare(sourceName, sizeof(Char64), parameters.sourceName.sourceValues[insertedArrayIndex]));
        }
    }

    void checkParametersAfterCloning(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int insertedOrderNumber)
    {
        checkParametersAfterInsertion(parameters, origParameters, locationTypes, insertedOrderNumber);

        auto locationType = LocationEditService::get().getLocationType(insertedOrderNumber, parameters);
        auto insertedArrayIndex = LocationEditService::get().findLocationArrayIndex(parameters, insertedOrderNumber);
        auto prevArrayIndex = LocationEditService::get().findLocationArrayIndex(parameters, insertedOrderNumber - 1);

        if (locationType == LocationType::Layer) {
            EXPECT_EQ(parameters.layerCoreRadius.layerValues[prevArrayIndex], parameters.layerCoreRadius.layerValues[insertedArrayIndex]);
            EXPECT_TRUE(
                StringHelper::compare(parameters.layerName.layerValues[prevArrayIndex], sizeof(Char64), parameters.layerName.layerValues[insertedArrayIndex]));
        } else if (locationType == LocationType::Source) {
            EXPECT_EQ(parameters.sourceCircularRadius.sourceValues[prevArrayIndex], parameters.sourceCircularRadius.sourceValues[insertedArrayIndex]);
            EXPECT_TRUE(StringHelper::compare(
                parameters.sourceName.sourceValues[prevArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[insertedArrayIndex]));
        }
    }

    void checkParametersAfterDeletion(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int deletedOrderNumber)
    {
        checkParameters(parameters, locationTypes);
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto orderNumber = parameters.layerOrderNumbers[i];
            auto origOrderNumber = orderNumber < deletedOrderNumber ? orderNumber : orderNumber + 1;
            auto origArrayIndex = LocationEditService::get().findLocationArrayIndex(origParameters, origOrderNumber);

            EXPECT_EQ(origParameters.layerCoreRadius.layerValues[origArrayIndex], parameters.layerCoreRadius.layerValues[i]);
            EXPECT_EQ(origParameters.layerIds[origArrayIndex], parameters.layerIds[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.layerName.layerValues[origArrayIndex], sizeof(Char64), parameters.layerName.layerValues[i]));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto orderNumber = parameters.sourceOrderNumbers[i];
            auto origOrderNumber = orderNumber < deletedOrderNumber ? orderNumber : orderNumber + 1;
            auto origArrayIndex = LocationEditService::get().findLocationArrayIndex(origParameters, origOrderNumber);

            EXPECT_EQ(origParameters.sourceCircularRadius.sourceValues[origArrayIndex], parameters.sourceCircularRadius.sourceValues[i]);
            EXPECT_EQ(origParameters.sourceIds[origArrayIndex], parameters.sourceIds[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.sourceName.sourceValues[origArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[i]));
        }
    }

    void checkParametersAfterMovingUpwards(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int movedOrderNumber)
    {
        checkParameters(parameters, locationTypes);
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto orderNumber = parameters.layerOrderNumbers[i];
            auto origOrderNumber = [&] {
                if (orderNumber < movedOrderNumber - 1 || orderNumber > movedOrderNumber) {
                    return orderNumber;
                } else if (orderNumber == movedOrderNumber - 1) {
                    return movedOrderNumber;
                } else if (orderNumber == movedOrderNumber) {
                    return movedOrderNumber - 1;
                } else {
                    CHECK(false);
                }
            }();

            auto origArrayIndex = LocationEditService::get().findLocationArrayIndex(origParameters, origOrderNumber);

            EXPECT_EQ(origParameters.layerCoreRadius.layerValues[origArrayIndex], parameters.layerCoreRadius.layerValues[i]);
            EXPECT_EQ(origParameters.layerIds[origArrayIndex], parameters.layerIds[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.layerName.layerValues[origArrayIndex], sizeof(Char64), parameters.layerName.layerValues[i]));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto orderNumber = parameters.sourceOrderNumbers[i];
            auto origOrderNumber = [&] {
                if (orderNumber < movedOrderNumber - 1 || orderNumber > movedOrderNumber) {
                    return orderNumber;
                } else if (orderNumber == movedOrderNumber - 1) {
                    return movedOrderNumber;
                } else if (orderNumber == movedOrderNumber) {
                    return movedOrderNumber - 1;
                } else {
                    CHECK(false);
                }
            }();
            auto origArrayIndex = LocationEditService::get().findLocationArrayIndex(origParameters, origOrderNumber);

            EXPECT_EQ(origParameters.sourceCircularRadius.sourceValues[origArrayIndex], parameters.sourceCircularRadius.sourceValues[i]);
            EXPECT_EQ(origParameters.sourceIds[origArrayIndex], parameters.sourceIds[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.sourceName.sourceValues[origArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[i]));
        }
    }

    void checkParametersAfterMovingDownwards(
        SimulationParameters const& parameters,
        SimulationParameters const& origParameters,
        std::vector<LocationType> const& locationTypes,
        int movedOrderNumber)
    {
        checkParameters(parameters, locationTypes);
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto orderNumber = parameters.layerOrderNumbers[i];
            auto origOrderNumber = [&] {
                if (orderNumber < movedOrderNumber || orderNumber > movedOrderNumber + 1) {
                    return orderNumber;
                } else if (orderNumber == movedOrderNumber + 1) {
                    return movedOrderNumber;
                } else if (orderNumber == movedOrderNumber) {
                    return movedOrderNumber + 1;
                } else {
                    CHECK(false);
                }
            }();

            auto origArrayIndex = LocationEditService::get().findLocationArrayIndex(origParameters, origOrderNumber);

            EXPECT_EQ(origParameters.layerCoreRadius.layerValues[origArrayIndex], parameters.layerCoreRadius.layerValues[i]);
            EXPECT_EQ(origParameters.layerIds[origArrayIndex], parameters.layerIds[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.layerName.layerValues[origArrayIndex], sizeof(Char64), parameters.layerName.layerValues[i]));
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            auto orderNumber = parameters.sourceOrderNumbers[i];
            auto origOrderNumber = [&] {
                if (orderNumber < movedOrderNumber || orderNumber > movedOrderNumber + 1) {
                    return orderNumber;
                } else if (orderNumber == movedOrderNumber + 1) {
                    return movedOrderNumber;
                } else if (orderNumber == movedOrderNumber) {
                    return movedOrderNumber + 1;
                } else {
                    CHECK(false);
                }
            }();
            auto origArrayIndex = LocationEditService::get().findLocationArrayIndex(origParameters, origOrderNumber);

            EXPECT_EQ(origParameters.sourceCircularRadius.sourceValues[origArrayIndex], parameters.sourceCircularRadius.sourceValues[i]);
            EXPECT_EQ(origParameters.sourceIds[origArrayIndex], parameters.sourceIds[i]);
            EXPECT_TRUE(StringHelper::compare(origParameters.sourceName.sourceValues[origArrayIndex], sizeof(Char64), parameters.sourceName.sourceValues[i]));
        }
    }
};

TEST_F(ParametersEditServiceTests, cloneLayer)
{
    auto origParameters = createTestData({LocationType::Layer, LocationType::Layer, LocationType::Source, LocationType::Source});
    auto parameters = origParameters;
    ParametersEditService::get().cloneLocation(parameters, 1, ParametersEditService::generateLocationId(parameters));
    checkParametersAfterCloning(
        parameters, origParameters, {LocationType::Layer, LocationType::Layer, LocationType::Layer, LocationType::Source, LocationType::Source}, 2);
}

TEST_F(ParametersEditServiceTests, cloneSource)
{
    auto origParameters = createTestData({LocationType::Layer, LocationType::Layer, LocationType::Source, LocationType::Source});
    auto parameters = origParameters;
    ParametersEditService::get().cloneLocation(parameters, 3, ParametersEditService::generateLocationId(parameters));
    checkParametersAfterCloning(
        parameters, origParameters, {LocationType::Layer, LocationType::Layer, LocationType::Source, LocationType::Source, LocationType::Source}, 4);
}

TEST_F(ParametersEditServiceTests, insertDefaultLayer_empty)
{
    auto origParameters = createTestData({});
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 0, ParametersEditService::generateLocationId(parameters));
    checkParametersAfterDefaultInsertion(parameters, origParameters, {LocationType::Layer}, 1);
}

TEST_F(ParametersEditServiceTests, insertDefaultLayer_onlySources)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 1, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultLayer_base)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 0, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultLayer_firstLayer1)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 1, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultLayer_firstLayer2)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 2, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultLayer_middle1)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 3, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultLayer_middle2)
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
    ParametersEditService::get().insertDefaultLayer(parameters, 4, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultLayer_end)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultLayer(parameters, 5, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, initNewLayer)
{
    auto parameters = createTestData({LocationType::Source, LocationType::Layer});
    auto backgroundColor = FloatColorRGB{0.1f, 0.2f, 0.3f};

    ParametersEditService::get().initNewLayer(parameters, 2, {600, 300}, {100.0f, 50.0f}, backgroundColor);

    EXPECT_TRUE(parameters.backgroundColor.layerValues[0].enabled);
    EXPECT_EQ(backgroundColor, parameters.backgroundColor.layerValues[0].value);
    EXPECT_EQ(RealVector2D(100.0f, 50.0f), parameters.layerPosition.layerValues[0]);
    EXPECT_EQ(50.0f, parameters.layerCoreRadius.layerValues[0]);
    EXPECT_EQ(RealVector2D(50.0f, 50.0f), parameters.layerCoreRect.layerValues[0]);
    EXPECT_EQ(30.0f, parameters.layerFadeoutRadius.layerValues[0]);
}

TEST_F(ParametersEditServiceTests, locationIds)
{
    auto parameters = createTestData({LocationType::Source, LocationType::Layer, LocationType::Source});

    EXPECT_EQ(0, LocationEditService::get().getLocationId(parameters, 0));
    EXPECT_EQ(1, LocationEditService::get().getLocationId(parameters, 2));
    EXPECT_EQ(2, LocationEditService::get().getLocationId(parameters, 1));
    EXPECT_EQ(3, LocationEditService::get().getLocationId(parameters, 3));
    EXPECT_EQ(std::optional(0), LocationEditService::get().findOrderNumber(parameters, 0));
    EXPECT_EQ(std::optional(2), LocationEditService::get().findOrderNumber(parameters, 1));
    EXPECT_FALSE(LocationEditService::get().findOrderNumber(parameters, 4).has_value());
    EXPECT_EQ(3, LocationEditService::get().getMaxLocationId(parameters));
    EXPECT_LT(3, ParametersEditService::generateLocationId(parameters));
}

TEST_F(ParametersEditServiceTests, locationIds_followLocation)
{
    auto& editService = ParametersEditService::get();
    auto parameters = createTestData({LocationType::Layer, LocationType::Layer, LocationType::Source});
    auto locationId = LocationEditService::get().getLocationId(parameters, 2);

    editService.moveLocationUpwards(parameters, 2);
    EXPECT_EQ(std::optional(1), LocationEditService::get().findOrderNumber(parameters, locationId));

    editService.insertDefaultSource(parameters, 0, ParametersEditService::generateLocationId(parameters));
    EXPECT_EQ(std::optional(2), LocationEditService::get().findOrderNumber(parameters, locationId));

    auto insertedLocationId = LocationEditService::get().getLocationId(parameters, 1);
    editService.deleteLocation(parameters, 1);
    EXPECT_EQ(std::optional(1), LocationEditService::get().findOrderNumber(parameters, locationId));
    EXPECT_FALSE(LocationEditService::get().findOrderNumber(parameters, insertedLocationId).has_value());
}

TEST_F(ParametersEditServiceTests, locationIds_notReusedAfterDeletion)
{
    auto& editService = ParametersEditService::get();
    auto parameters = createTestData({LocationType::Layer, LocationType::Layer});
    auto deletedLocationId = LocationEditService::get().getLocationId(parameters, 2);

    editService.deleteLocation(parameters, 2);
    editService.insertDefaultLayer(parameters, 1, ParametersEditService::generateLocationId(parameters));

    EXPECT_NE(deletedLocationId, LocationEditService::get().getLocationId(parameters, 2));
    EXPECT_FALSE(LocationEditService::get().findOrderNumber(parameters, deletedLocationId).has_value());
}

TEST_F(ParametersEditServiceTests, insertDefaultSource_empty)
{
    auto origParameters = createTestData({});
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 0, ParametersEditService::generateLocationId(parameters));
    checkParametersAfterDefaultInsertion(parameters, origParameters, {LocationType::Source}, 1);
}

TEST_F(ParametersEditServiceTests, insertDefaultSource_onlyLayers)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 1, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultSource_base)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 0, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultSource_firstSource1)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 1, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultSource_firstSource2)
{
    auto origParameters = createTestData({
        LocationType::Layer,
        LocationType::Source,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 2, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultSource_middle1)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 3, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultSource_middle2)
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
    ParametersEditService::get().insertDefaultSource(parameters, 4, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, insertDefaultSource_end)
{
    auto origParameters = createTestData({
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
        LocationType::Layer,
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().insertDefaultSource(parameters, 5, ParametersEditService::generateLocationId(parameters));
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

TEST_F(ParametersEditServiceTests, deleteLayer_afterwardEmpty)
{
    auto origParameters = createTestData({
        LocationType::Layer,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 1);
    checkParametersAfterDeletion(parameters, origParameters, {}, 1);
}

TEST_F(ParametersEditServiceTests, deleteLayer_afterwardOnlySources)
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

TEST_F(ParametersEditServiceTests, deleteLayer_firstLayer)
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

TEST_F(ParametersEditServiceTests, deleteLayer_middle)
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

TEST_F(ParametersEditServiceTests, deleteLayer_end)
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

TEST_F(ParametersEditServiceTests, deleteSource_afterwardEmpty)
{
    auto origParameters = createTestData({
        LocationType::Source,
    });
    auto parameters = origParameters;
    ParametersEditService::get().deleteLocation(parameters, 1);
    checkParametersAfterDeletion(parameters, origParameters, {}, 1);
}

TEST_F(ParametersEditServiceTests, deleteSource_afterwardOnlyLayers)
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

TEST_F(ParametersEditServiceTests, deleteSource_firstSource)
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

TEST_F(ParametersEditServiceTests, deleteSource_middle)
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

TEST_F(ParametersEditServiceTests, deleteSource_end)
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

TEST_F(ParametersEditServiceTests, moveLayerUpwards_afterOtherSource)
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

TEST_F(ParametersEditServiceTests, moveLayerUpwards_afterOtherLayer)
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

TEST_F(ParametersEditServiceTests, moveLayerUpwards_afterOtherLayerAndSources)
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

TEST_F(ParametersEditServiceTests, moveSourceUpwards_afterOtherLayer)
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

TEST_F(ParametersEditServiceTests, moveSourceUpwards_afterOtherSource)
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

TEST_F(ParametersEditServiceTests, moveSourceUpwards_afterOtherLayerAndSources)
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

TEST_F(ParametersEditServiceTests, moveLayerDownwards_beforeOtherSource)
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

TEST_F(ParametersEditServiceTests, moveLayerDownwards_beforeOtherLayer)
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

TEST_F(ParametersEditServiceTests, moveLayerDownwards_beforeOtherLayerAndSources)
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

TEST_F(ParametersEditServiceTests, moveSourceDownwards_beforeOtherLayer)
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

TEST_F(ParametersEditServiceTests, moveSourceDownwards_beforeOtherSource)
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

TEST_F(ParametersEditServiceTests, moveSourceDownwards_beforeOtherLayerAndSources)
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