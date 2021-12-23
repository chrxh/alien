#include "EditorModel.h"

#include "EngineImpl/SimulationController.h"

_EditorModel::_EditorModel(SimulationController const& simController, Viewport const& viewport)
    : _simController(simController)
    , _viewport(viewport)
{
    clear();
}

SelectionShallowData const& _EditorModel::getSelectionShallowData() const
{
    return _selectionShallowData;
}

void _EditorModel::update()
{
    _selectionShallowData = _simController->getSelectionShallowData();
}

bool _EditorModel::isSelectionEmpty() const
{
    return 0 == _selectionShallowData.numCells && 0 == _selectionShallowData.numClusterCells
        && 0 == _selectionShallowData.numParticles;
}

void _EditorModel::clear()
{
    _selectionShallowData = SelectionShallowData();
}

std::vector<InspectorWindow> const& _EditorModel::getInspectorWindows() const
{
    return _inspectorWindows;
}

void _EditorModel::inspectEntities(std::vector<CellOrParticleDescription> const& entities)
{
    for (auto const& entity : entities) {
        _inspectorWindows.emplace_back(boost::make_shared<_InspectorWindow>(entity, _viewport));
    }
}
