#include "EngineInterface/SimulationParametersEditService.h"

#include <gtest/gtest.h>

#include "Base/Definitions.h"

class SimulationParametersEditServiceTests : public ::testing::Test
{
public:
    SimulationParametersEditServiceTests() = default;

    ~SimulationParametersEditServiceTests() = default;

protected:
    void checkApproxEqual(float expected, float actual) { EXPECT_TRUE(std::abs(actual - expected) < NEAR_ZERO); }
};

TEST_F(SimulationParametersEditServiceTests, getRadiationStrengths)
{
    SimulationParameters parameters;
    parameters.numRadiationSources = 2;
    parameters.radiationSources[0].strength = 0.3f;
    parameters.radiationSources[1].strength = 0.6f;

    auto strengths = SimulationParametersEditService::get().getRadiationStrengths(parameters);

    checkApproxEqual(0.1f, strengths.values[0]);
}

TEST_F(SimulationParametersEditServiceTests, applyRadiationStrengthValues)
{
    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.3f, 0.6f};
    strengths.pinned = {0, 2};

    SimulationParameters parameters;
    parameters.numRadiationSources = 2;
    SimulationParametersEditService::get().applyRadiationStrengths(parameters, strengths);

    EXPECT_TRUE(parameters.baseStrengthRatioPinned);
    checkApproxEqual(0.3f, parameters.radiationSources[0].strength);
    EXPECT_FALSE(parameters.radiationSources[0].strengthPinned);
    checkApproxEqual(0.6f, parameters.radiationSources[1].strength);
    EXPECT_TRUE(parameters.radiationSources[1].strengthPinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_increase_allUnpinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.5f, 0.6f};
    strengths.pinned = {};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    auto factor = 0.5f / 0.7f;
    checkApproxEqual(0.1f * factor, strengths.values[0]);
    checkApproxEqual(0.5f, strengths.values[1]);
    checkApproxEqual(0.6f * factor, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_increase_basePinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {0};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.5f, 0.6f};
    strengths.pinned = {0};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.1f, strengths.values[0]);
    checkApproxEqual(0.5f, strengths.values[1]);
    checkApproxEqual(0.4f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_increase_spotPinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {2};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.4f, 0.6f};
    strengths.pinned = {2};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.0f, strengths.values[0]);
    checkApproxEqual(0.4f, strengths.values[1]);
    checkApproxEqual(0.6f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_increase_allPinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {0, 1, 2};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.5f, 0.6f};
    strengths.pinned = {0, 1, 2};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.1f, strengths.values[0]);
    checkApproxEqual(0.3f, strengths.values[1]);
    checkApproxEqual(0.6f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_decrease_allUnpinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.1f, 0.6f};
    strengths.pinned = {};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    auto factor = 1 + 0.2f / 0.7f;
    checkApproxEqual(0.1f * factor, strengths.values[0]);
    checkApproxEqual(0.1f, strengths.values[1]);
    checkApproxEqual(0.6f * factor, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_decrease_basePinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {0};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.1f, 0.6f};
    strengths.pinned = {0};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.1f, strengths.values[0]);
    checkApproxEqual(0.1f, strengths.values[1]);
    checkApproxEqual(0.8f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_decrease_spotPinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {2};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.1f, 0.6f};
    strengths.pinned = {2};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.3f, strengths.values[0]);
    checkApproxEqual(0.1f, strengths.values[1]);
    checkApproxEqual(0.6f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_decrease_allPinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {0, 1, 2};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.1f, 0.6f};
    strengths.pinned = {0, 1, 2};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.1f, strengths.values[0]);
    checkApproxEqual(0.3f, strengths.values[1]);
    checkApproxEqual(0.6f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_exceed_aboveOne_allUnpinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 1.1f, 0.6f};
    strengths.pinned = {};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.0f, strengths.values[0]);
    checkApproxEqual(1.0f, strengths.values[1]);
    checkApproxEqual(0.0f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_exceed_aboveOne_basePinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {0};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 1.1f, 0.6f};
    strengths.pinned = {0};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.1f, strengths.values[0]);
    checkApproxEqual(0.9f, strengths.values[1]);
    checkApproxEqual(0.0f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_exceed_aboveOne_zonePinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {2};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 1.1f, 0.6f};
    strengths.pinned = {2};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.0f, strengths.values[0]);
    checkApproxEqual(0.4f, strengths.values[1]);
    checkApproxEqual(0.6f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_exceed_negative_explicit)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {};

    RadiationStrengths strengths;
    strengths.values = {0.1f, -0.1f, 0.6f};
    strengths.pinned = {};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.1f, strengths.values[0]);
    checkApproxEqual(0.3f, strengths.values[1]);
    checkApproxEqual(0.6f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_exceed_negative_implicit)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {2};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.5f, 0.6f};
    strengths.pinned = {2};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    checkApproxEqual(0.0f, strengths.values[0]);
    checkApproxEqual(0.4f, strengths.values[1]);
    checkApproxEqual(0.6f, strengths.values[2]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, calcRadiationStrengthsForAddingSpot_allUnpinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {};

    auto strengths = SimulationParametersEditService::get().calcRadiationStrengthsForAddingSpot(origStrengths);

    ASSERT_EQ(4, strengths.values.size());

    auto factor = 1.0f - 1.0f / 4;
    checkApproxEqual(0.1f * factor, strengths.values[0]);
    checkApproxEqual(0.3f * factor, strengths.values[1]);
    checkApproxEqual(0.6f * factor, strengths.values[2]);
    checkApproxEqual(0.25f, strengths.values[3]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, calcRadiationStrengthsForAddingSpot_basePinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {0};

    auto strengths = SimulationParametersEditService::get().calcRadiationStrengthsForAddingSpot(origStrengths);

    ASSERT_EQ(4, strengths.values.size());

    auto factor = 1.0f - 1.0f / 3;
    checkApproxEqual(0.1f, strengths.values[0]);
    checkApproxEqual(0.3f * factor, strengths.values[1]);
    checkApproxEqual(0.6f * factor, strengths.values[2]);
    checkApproxEqual((0.3f + 0.6f) * (1.0f - factor), strengths.values[3]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}

TEST_F(SimulationParametersEditServiceTests, calcRadiationStrengthsForAddingSpot_spotPinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {2};

    auto strengths = SimulationParametersEditService::get().calcRadiationStrengthsForAddingSpot(origStrengths);

    ASSERT_EQ(4, strengths.values.size());

    auto factor = 1.0f - 1.0f / 3;
    checkApproxEqual(0.1f * factor, strengths.values[0]);
    checkApproxEqual(0.3f * factor, strengths.values[1]);
    checkApproxEqual(0.6f, strengths.values[2]);
    checkApproxEqual((0.1f + 0.3f) * (1.0f - factor), strengths.values[3]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}


TEST_F(SimulationParametersEditServiceTests, calcRadiationStrengthsForAddingSpot_allPinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {0, 1, 2};

    auto strengths = SimulationParametersEditService::get().calcRadiationStrengthsForAddingSpot(origStrengths);

    ASSERT_EQ(4, strengths.values.size());

    checkApproxEqual(0.1f, strengths.values[0]);
    checkApproxEqual(0.3f, strengths.values[1]);
    checkApproxEqual(0.6f, strengths.values[2]);
    checkApproxEqual(0.0f, strengths.values[3]);
    EXPECT_EQ(origStrengths.pinned, strengths.pinned);
}
