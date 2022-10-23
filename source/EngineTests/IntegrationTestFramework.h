#pragma once

#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

class IntegrationTestFramework : public ::testing::Test
{
public:
    IntegrationTestFramework(IntVector2D const& universeSize);
    virtual ~IntegrationTestFramework();

protected:
    std::unordered_map<uint64_t, CellDescription> getCellById(DataDescription const& data) const;

    SimulationController _simController;
};
