#include "EditorModel.h"

_EditorModel::_EditorModel()
{
    clear();
}

SelectionShallowData const& _EditorModel::getSelectionShallowData() const
{
    return _selectionShallowData;
}

void _EditorModel::setSelectionShallowData(SelectionShallowData const& value)
{
    _selectionShallowData = value;
}

void _EditorModel::setOrigSelectionShallowData(SelectionShallowData const& value)
{
    _origSelectionShallowData = value;
    _selectionShallowData = value;
}

void _EditorModel::clear()
{
    _origSelectionShallowData = SelectionShallowData();
    _selectionShallowData = _origSelectionShallowData;
}

bool _EditorModel::isSelectionEmpty() const
{
    return 0 == _selectionShallowData.numCells && 0 == _selectionShallowData.numClusterCells
        && 0 == _selectionShallowData.numParticles;
}

RealVector2D _EditorModel::getDeltaExtCenterPos() const
{
    return {
        _selectionShallowData.extCenterPosX - _origSelectionShallowData.extCenterPosX,
        _selectionShallowData.extCenterPosY - _origSelectionShallowData.extCenterPosY};
}

RealVector2D _EditorModel::getDeltaExtCenterVel() const
{
    return {
        _selectionShallowData.extCenterVelX - _origSelectionShallowData.extCenterVelX,
        _selectionShallowData.extCenterVelY - _origSelectionShallowData.extCenterVelY};
}
