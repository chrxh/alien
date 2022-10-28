#pragma once

#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationParameters.h"

class IntegrationTestFramework : public ::testing::Test
{
public:
    IntegrationTestFramework(IntVector2D const& universeSize);
    virtual ~IntegrationTestFramework();

protected:
    std::unordered_map<uint64_t, CellDescription> getCellById(DataDescription const& data) const;
    void expectApproxEqual(float expected, float actual) const;
    void expectApproxEqual(std::vector<float> const& expected, std::vector<float> const& actual) const;

    bool compare(DataDescription left, DataDescription right) const;
    bool compare(CellDescription left, CellDescription right) const;
    bool compare(ParticleDescription left, ParticleDescription right) const;

    SimulationController _simController;
    SimulationParameters _parameters;
};
