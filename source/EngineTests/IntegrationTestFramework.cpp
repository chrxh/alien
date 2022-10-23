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
    for (auto& cell : left.cells) {
        cell.id = 0;
    }
    for (auto& particle : left.particles) {
        particle.id = 0;
    }
    for (auto& cell : right.cells) {
        cell.id = 0;
    }
    for (auto& particle : right.particles) {
        particle.id = 0;
    }
    return left == right;
}
