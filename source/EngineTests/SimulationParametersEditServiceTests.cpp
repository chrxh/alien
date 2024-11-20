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
    parameters.radiationSources[0].strengthRatio = 0.3f;
    parameters.radiationSources[1].strengthRatio = 0.6f;

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
    SimulationParametersEditService::get().applyRadiationStrengthValues(parameters, strengths);

    checkApproxEqual(0.3f, parameters.radiationSources[0].strengthRatio);
    checkApproxEqual(0.6f, parameters.radiationSources[1].strengthRatio);
}

TEST_F(SimulationParametersEditServiceTests, adaptRadiationStrengths_increase_allUnpinned)
{
    RadiationStrengths origStrengths;
    origStrengths.values = {0.1f, 0.3f, 0.6f};
    origStrengths.pinned = {};

    RadiationStrengths strengths;
    strengths.values = {0.1f, 0.5f, 0.6f};
    strengths.pinned = {1};

    SimulationParametersEditService::get().adaptRadiationStrengths(strengths, origStrengths, 1);

    auto factor = 0.5f / 0.7f;
    checkApproxEqual(0.1f * factor, strengths.values[0]);
    checkApproxEqual(0.5f, strengths.values[1]);
    checkApproxEqual(0.6f * factor, strengths.values[2]);
}
