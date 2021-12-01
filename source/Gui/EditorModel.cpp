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

RealVector2D _EditorModel::getClusterCenterPosDelta() const
{
    return {
        _selectionShallowData.clusterCenterPosX - _origSelectionShallowData.clusterCenterPosX,
        _selectionShallowData.clusterCenterPosY - _origSelectionShallowData.clusterCenterPosY};
}

RealVector2D _EditorModel::getClusterCenterVelDelta() const
{
    return {
        _selectionShallowData.clusterCenterVelX - _origSelectionShallowData.clusterCenterVelX,
        _selectionShallowData.clusterCenterVelY - _origSelectionShallowData.clusterCenterVelY};
}

RealVector2D _EditorModel::getCenterPosDelta() const
{
    return {
        _selectionShallowData.centerPosX - _origSelectionShallowData.centerPosX,
        _selectionShallowData.centerPosY - _origSelectionShallowData.centerPosY};
}

RealVector2D _EditorModel::getCenterVelDelta() const
{
    return {
        _selectionShallowData.centerVelX - _origSelectionShallowData.centerVelX,
        _selectionShallowData.centerVelY - _origSelectionShallowData.centerVelY};
}
