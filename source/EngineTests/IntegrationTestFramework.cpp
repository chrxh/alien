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
    SymbolMap symbolMap;
    _simController->newSimulation(0, settings, symbolMap);
}

IntegrationTestFramework::~IntegrationTestFramework() {}

TokenDescription IntegrationTestFramework::createSimpleToken() const
{
    auto parameters = _simController->getSimulationParameters();
    auto tokenEnergy = parameters.tokenMinEnergy * 2.0;
    return TokenDescription().setEnergy(tokenEnergy).setData(std::string(parameters.tokenMemorySize, 0));
}

std::unordered_map<uint64_t, CellDescription> IntegrationTestFramework::getCellById(DataDescription const& data) const
{
    std::unordered_map<uint64_t, CellDescription> result;
    for(auto const& cell : data.cells) {
        result.emplace(cell.id, cell);
    }
    return result;
}
