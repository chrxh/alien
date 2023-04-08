#pragma once

#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationParameters.h"

class IntegrationTestFramework : public ::testing::Test
{
public:
    IntegrationTestFramework(std::optional<SimulationParameters> const& parameters = std::nullopt, IntVector2D const& universeSize = IntVector2D{1000, 1000});
    virtual ~IntegrationTestFramework();

protected:
    double getEnergy(DataDescription const& data) const;

    std::unordered_map<uint64_t, CellDescription> getCellById(DataDescription const& data) const;
    CellDescription getCell(DataDescription const& data, uint64_t id) const;
    ConnectionDescription getConnection(DataDescription const& data, uint64_t id, uint64_t otherId) const;
    bool hasConnection(DataDescription const& data, uint64_t id, uint64_t otherId) const;
    CellDescription getOtherCell(DataDescription const& data, uint64_t id) const;
    CellDescription getOtherCell(DataDescription const& data, std::set<uint64_t> ids) const;

    bool approxCompare(double expected, double actual, float precision = 0.001f) const;
    bool approxCompare(float expected, float actual, float precision = 0.001f) const;
    bool approxCompare(RealVector2D const& expected, RealVector2D const& actual) const;
    bool approxCompare(std::vector<float> const& expected, std::vector<float> const& actual) const;

    bool compare(DataDescription left, DataDescription right) const;
    bool compare(CellDescription left, CellDescription right) const;
    bool compare(ParticleDescription left, ParticleDescription right) const;

    SimulationController _simController;
    SimulationParameters _parameters;
};
