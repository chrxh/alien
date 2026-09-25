#include <stdexcept>

#include <gtest/gtest.h>

#include <Data/ParametersAccessService.h>

class ParametersAccessServiceTests : public ::testing::Test
{
protected:
    static auto constexpr LayerOrderNumber = 1;
    static auto constexpr SourceOrderNumber = 2;

    SimulationParameters createParametersWithLayerAndSource() const
    {
        SimulationParameters result;
        result.numLayers = 1;
        result.layerOrderNumbers[0] = LayerOrderNumber;
        result.numSources = 1;
        result.sourceOrderNumbers[0] = SourceOrderNumber;
        return result;
    }

    ParameterEntry findParameter(std::string const& path, LocationType locationType = LocationType::Base) const
    {
        auto result = ParametersAccessService::get().findParameter(path, locationType);
        EXPECT_TRUE(result.has_value());
        return result.value_or(ParameterEntry());
    }

    ParameterValue createValue(std::vector<ParameterScalar> const& values, ColorDependence colorDependence = ColorDependence::None) const
    {
        return ParameterValue{.colorDependence = colorDependence, .values = values};
    }

    std::vector<ParameterScalar> uniform(ParameterScalar const& value, int size = MAX_COLORS) const { return std::vector<ParameterScalar>(size, value); }
};

TEST_F(ParametersAccessServiceTests, findParameter_caseInsensitive)
{
    auto entry = ParametersAccessService::get().findParameter("physics: motion . FRICTION", LocationType::Base);

    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ("Physics: Motion.Friction", entry->path);
    EXPECT_EQ(ParameterType::Float, ParametersAccessService::get().getType(*entry));
}

TEST_F(ParametersAccessServiceTests, findParameter_invalidPaths)
{
    auto& service = ParametersAccessService::get();

    EXPECT_FALSE(service.findParameter("Physics: Motion", LocationType::Base).has_value());
    EXPECT_FALSE(service.findParameter("Physics: Motion.Unknown", LocationType::Base).has_value());
    EXPECT_FALSE(service.findParameter("Unknown.Friction", LocationType::Base).has_value());
    EXPECT_FALSE(service.findParameter("Force field.Field type.Radial", LocationType::Layer).has_value());
    EXPECT_FALSE(service.findParameter("Physics: Motion.Friction.Radial.Strength", LocationType::Base).has_value());
}

TEST_F(ParametersAccessServiceTests, findParameter_dependsOnLocation)
{
    auto& service = ParametersAccessService::get();
    auto parameters = createParametersWithLayerAndSource();

    EXPECT_FALSE(service.findParameter("Location.Position (x,y)", LocationType::Base).has_value());

    auto layerEntry = findParameter("Location.Position (x,y)", LocationType::Layer);
    service.setValue(layerEntry, parameters, LayerOrderNumber, createValue({RealVector2D{10.0f, 20.0f}}));
    EXPECT_EQ(RealVector2D(10.0f, 20.0f), parameters.layerPosition.layerValues[0]);

    auto sourceEntry = findParameter("Location.Position (x,y)", LocationType::Source);
    service.setValue(sourceEntry, parameters, SourceOrderNumber, createValue({RealVector2D{30.0f, 40.0f}}));
    EXPECT_EQ(RealVector2D(30.0f, 40.0f), parameters.sourcePosition.sourceValues[0]);
}

TEST_F(ParametersAccessServiceTests, setValue_float)
{
    auto& service = ParametersAccessService::get();
    SimulationParameters parameters;
    auto entry = findParameter("Physics: Motion.Friction");

    service.setValue(entry, parameters, 0, createValue({0.25f}));

    EXPECT_EQ(0.25f, parameters.friction.baseValue);
    EXPECT_EQ(createValue({0.25f}), service.getValue(entry, parameters, 0));
}

TEST_F(ParametersAccessServiceTests, setValue_colorVector)
{
    auto& service = ParametersAccessService::get();
    SimulationParameters parameters;
    auto entry = findParameter("Cell life cycle.Minimum energy");

    auto values = uniform(40.0f);
    values.at(3) = 70.0f;
    service.setValue(entry, parameters, 0, createValue(values, ColorDependence::ColorVector));

    EXPECT_EQ(40.0f, parameters.minCellEnergy.baseValue[0]);
    EXPECT_EQ(70.0f, parameters.minCellEnergy.baseValue[3]);
    EXPECT_EQ(values, service.getValue(entry, parameters, 0).values);
}

TEST_F(ParametersAccessServiceTests, setValue_colorMatrix)
{
    auto& service = ParametersAccessService::get();
    SimulationParameters parameters;
    auto entry = findParameter("Cell type: Attacker.Food chain color matrix");

    auto values = uniform(1.0f, MAX_COLORS * MAX_COLORS);
    values.at(2 * MAX_COLORS + 5) = 0.0f;
    service.setValue(entry, parameters, 0, createValue(values, ColorDependence::ColorMatrix));

    EXPECT_EQ(0.0f, parameters.attackerFoodChainColorMatrix.baseValue[2][5]);
    EXPECT_EQ(1.0f, parameters.attackerFoodChainColorMatrix.baseValue[5][2]);
}

TEST_F(ParametersAccessServiceTests, setValue_wrongNumberOfValues)
{
    SimulationParameters parameters;
    auto entry = findParameter("Cell life cycle.Minimum energy");

    EXPECT_THROW(ParametersAccessService::get().setValue(entry, parameters, 0, createValue({40.0f})), std::invalid_argument);
}

TEST_F(ParametersAccessServiceTests, setValue_wrongType)
{
    SimulationParameters parameters;
    auto entry = findParameter("Physics: Motion.Friction");

    EXPECT_THROW(ParametersAccessService::get().setValue(entry, parameters, 0, createValue({true})), std::invalid_argument);
}

TEST_F(ParametersAccessServiceTests, setValue_text)
{
    auto& service = ParametersAccessService::get();
    SimulationParameters parameters;
    auto entry = findParameter("General.Project name");

    service.setValue(entry, parameters, 0, createValue({std::string("Test project")}));

    EXPECT_EQ(std::string("Test project"), std::string(parameters.projectName.value));
    EXPECT_EQ(ParameterType::Text, service.getType(entry));
}

TEST_F(ParametersAccessServiceTests, setValue_alternativeByName)
{
    auto& service = ParametersAccessService::get();
    SimulationParameters parameters;
    auto entry = findParameter("Visualization.Object coloring");

    service.setValue(entry, parameters, 0, createValue(uniform(std::string("lineage")), ColorDependence::ColorVector));

    EXPECT_EQ(CellColoring_Lineage, parameters.objectColoring.value[0]);
    EXPECT_EQ(uniform(std::string("Lineage")), service.getValue(entry, parameters, 0).values);
}

TEST_F(ParametersAccessServiceTests, setValue_unknownAlternative)
{
    SimulationParameters parameters;
    auto entry = findParameter("Visualization.Object coloring");

    EXPECT_THROW(
        ParametersAccessService::get().setValue(entry, parameters, 0, createValue(uniform(std::string("Rainbow")), ColorDependence::ColorVector)),
        std::invalid_argument);
}

TEST_F(ParametersAccessServiceTests, getValue_layerWithoutOverride)
{
    auto& service = ParametersAccessService::get();
    auto parameters = createParametersWithLayerAndSource();
    parameters.friction.baseValue = 0.2f;
    parameters.friction.layerValues[0] = {.value = 0.5f, .enabled = false};
    auto entry = findParameter("Physics: Motion.Friction", LocationType::Layer);

    auto value = service.getValue(entry, parameters, LayerOrderNumber);

    EXPECT_EQ(std::vector<ParameterScalar>{0.2f}, value.values);
    EXPECT_EQ(std::optional(false), value.enabled);
}

TEST_F(ParametersAccessServiceTests, setValue_layerOverride)
{
    auto& service = ParametersAccessService::get();
    auto parameters = createParametersWithLayerAndSource();
    auto entry = findParameter("Physics: Motion.Friction", LocationType::Layer);

    auto value = createValue({0.7f});
    value.enabled = true;
    service.setValue(entry, parameters, LayerOrderNumber, value);

    EXPECT_EQ(0.7f, parameters.friction.layerValues[0].value);
    EXPECT_TRUE(parameters.friction.layerValues[0].enabled);
    EXPECT_EQ(SimulationParameters().friction.baseValue, parameters.friction.baseValue);
}

TEST_F(ParametersAccessServiceTests, setValue_enableNotSupported)
{
    SimulationParameters parameters;
    auto entry = findParameter("Physics: Motion.Friction");

    auto value = createValue({0.7f});
    value.enabled = true;
    EXPECT_THROW(ParametersAccessService::get().setValue(entry, parameters, 0, value), std::invalid_argument);
}

TEST_F(ParametersAccessServiceTests, setValue_relativeStrength)
{
    auto& service = ParametersAccessService::get();
    auto parameters = createParametersWithLayerAndSource();
    auto sourceEntry = findParameter("General.Relative strength", LocationType::Source);
    auto baseEntry = findParameter("Radiation.Relative strength");

    service.setValue(sourceEntry, parameters, SourceOrderNumber, createValue({0.3f}));

    EXPECT_EQ(std::vector<ParameterScalar>{0.3f}, service.getValue(sourceEntry, parameters, SourceOrderNumber).values);
    auto baseValue = service.getValue(baseEntry, parameters, 0);
    ASSERT_EQ(1, baseValue.values.size());
    EXPECT_NEAR(0.7f, std::get<float>(baseValue.values.front()), 1e-5f);
}

TEST_F(ParametersAccessServiceTests, getParameters_selectedAlternativeOnly)
{
    auto& service = ParametersAccessService::get();
    auto parameters = createParametersWithLayerAndSource();
    auto groupSpec = service.findGroup("force field");
    ASSERT_NE(nullptr, groupSpec);

    auto containsPath = [&](std::string const& path) {
        auto entries = service.getParameters(*groupSpec, parameters, LayerOrderNumber);
        return std::ranges::any_of(entries, [&](ParameterEntry const& entry) { return entry.path == path; });
    };
    EXPECT_TRUE(containsPath("Force field.Field type"));
    EXPECT_FALSE(containsPath("Force field.Field type.Radial.Strength"));

    auto entry = findParameter("Force field.Field type", LocationType::Layer);
    service.setValue(entry, parameters, LayerOrderNumber, createValue({std::string("Radial")}));
    EXPECT_TRUE(containsPath("Force field.Field type.Radial.Strength"));
}

TEST_F(ParametersAccessServiceTests, copyGroup)
{
    auto& service = ParametersAccessService::get();
    auto parameters = createParametersWithLayerAndSource();
    auto source = parameters;
    parameters.friction.baseValue = 0.5f;
    parameters.friction.layerValues[0] = {.value = 0.6f, .enabled = true};
    parameters.maxVelocity.value = 5.0f;

    EXPECT_TRUE(service.copyGroup(*service.findGroup("Physics: Motion"), source, parameters));

    EXPECT_EQ(source.friction, parameters.friction);
    EXPECT_EQ(5.0f, parameters.maxVelocity.value);
}

TEST_F(ParametersAccessServiceTests, copyGroup_differentLocations)
{
    auto& service = ParametersAccessService::get();
    SimulationParameters source;
    auto parameters = createParametersWithLayerAndSource();
    parameters.friction.baseValue = 0.5f;

    EXPECT_FALSE(service.copyGroup(*service.findGroup("Physics: Motion"), source, parameters));
    EXPECT_EQ(0.5f, parameters.friction.baseValue);
}
