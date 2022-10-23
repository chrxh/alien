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

std::map<RealVector2D, std::vector<CellDescription>> IntegrationTestFramework::getCellsByPosition(DataDescription const& data) const
{
    std::map<RealVector2D, std::vector<CellDescription>> result;
    for(auto const& cell : data.cells) {
        result[cell.pos].emplace_back(cell);
    }
    return result;
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
