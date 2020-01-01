#pragma once
#include "IntegrationTestFramework.h"

class IntegrationTestHelper
{
public:
    static DataDescription getContent(SimulationAccess* access, IntRect const& rect);
    static void updateData(SimulationAccess* access, DataChangeDescription const& data);
    static void runSimulation(int timesteps, SimulationController* controller);
    static unordered_map<uint64_t, ParticleDescription> getParticleByParticleId(DataDescription const& data);
    static unordered_map<uint64_t, CellDescription> getCellByCellId(DataDescription const& data);
    static unordered_map<uint64_t, ClusterDescription> getClusterByCellId(DataDescription const& data);
    static unordered_map<uint64_t, ClusterDescription> getClusterByClusterId(DataDescription const& data);
};