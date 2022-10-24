#include "IntegrationTestFramework.h"

#include "Base/NumberGenerator.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineImpl/SimulationControllerImpl.h"

IntegrationTestFramework::IntegrationTestFramework(IntVector2D const& universeSize)
{
    _simController = std::make_shared<_SimulationControllerImpl>();
    Settings settings;
    settings.generalSettings.worldSizeX = universeSize.x;
    settings.generalSettings.worldSizeY = universeSize.y;
    settings.simulationParameters.spotValues.radiationFactor = 0;
    _simController->newSimulation(0, settings);
}

IntegrationTestFramework::~IntegrationTestFramework()
{
    _simController->closeSimulation();
}

std::unordered_map<uint64_t, CellDescription> IntegrationTestFramework::getCellById(DataDescription const& data) const
{
    std::unordered_map<uint64_t, CellDescription> result;
    for(auto const& cell : data.cells) {
        result.emplace(cell.id, cell);
    }
    return result;
}

bool IntegrationTestFramework::compare(DataDescription left, DataDescription right) const
{
    std::sort(left.cells.begin(), left.cells.end(), [](auto const& left, auto const& right) { return left.id < right.id; });
    std::sort(right.cells.begin(), right.cells.end(), [](auto const& left, auto const& right) { return left.id < right.id; });
    std::sort(left.particles.begin(), left.particles.end(), [](auto const& left, auto const& right) { return left.id < right.id; });
    std::sort(right.particles.begin(), right.particles.end(), [](auto const& left, auto const& right) { return left.id < right.id; });
    return left == right;
}

bool IntegrationTestFramework::compare(CellDescription left, CellDescription right) const
{
    left.id = 0;
    right.id = 0;
    return left == right;
}

bool IntegrationTestFramework::compare(ParticleDescription left, ParticleDescription right) const
{
    left.id = 0;
    right.id = 0;
    return left == right;
}
