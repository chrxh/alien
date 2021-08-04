#pragma once
#include "IntegrationTestFramework.h"

class IntegrationTestHelper
{
public:
    static DataDescription getContent(SimulationAccess* access, IntRect const& rect);
    static void IntegrationTestHelper::updateData(
        SimulationAccess* access,
        SimulationContext* context,
        DataChangeDescription const& data);
    static void runSimulation(int timesteps, SimulationController* controller);

    static std::vector<std::pair<boost::optional<CellDescription>, boost::optional<CellDescription>>>
    getBeforeAndAfterCells(
        DataDescription const& dataBefore,
        DataDescription const& dataAfter);
    static unordered_map<uint64_t, ParticleDescription> getParticleByParticleId(DataDescription const& data);
    static unordered_map<uint64_t, CellDescription> getCellByCellId(DataDescription const& data);
    static unordered_map<uint64_t, ClusterDescription> getClusterByCellId(DataDescription const& data);
    static unordered_map<uint64_t, ClusterDescription> getClusterByClusterId(DataDescription const& data);
};