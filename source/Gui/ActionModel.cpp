#include "ActionModel.h"

#include <QAction>

#include "ActionHolder.h"

ActionModel::ActionModel(QObject* parent)
	: QObject(parent)
{
	_actions = new ActionHolder(this);
}

ActionHolder * ActionModel::getActionHolder() const
{
	return _actions;
}

QVector2D ActionModel::getPositionDeltaForNewEntity()
{
	_delta += 0.5;
	if (_delta > 10.0) {
		_delta = 0.0;
	}
	return QVector2D(_delta, -_delta);
}

bool ActionModel::isEditMode() const
{
	return _isEditMode;
}

void ActionModel::setEditMode(bool value)
{
	_isEditMode = value;
}

bool ActionModel::isEntitySelected() const
{
	return _entitySelected;
}

void ActionModel::setEntitySelected(bool value)
{
	_entitySelected = value;
}

bool ActionModel::isEntityCopied() const
{
	return !_copiedEntity.isEmpty();
}

DataDescription const & ActionModel::getCopiedEntity() const
{
	return _copiedEntity;
}

void ActionModel::setCellCopied(CellDescription cell, QVector2D const& vel)
{
	if (cell.connections) {
        cell.connections->clear();
	}
	_copiedEntity.clear();
    _copiedEntity.addCluster(ClusterDescription().addCell(cell));
}

void ActionModel::setParticleCopied(ParticleDescription const & value)
{
	_copiedEntity.clear();
	_copiedEntity.addParticle(value);
}

bool ActionModel::areCellsSelected() const
{
    return _cellsSelected;
}

void ActionModel::setCellsSelected(bool value)
{
    _cellsSelected = value;
}

bool ActionModel::isCellWithTokenSelected() const
{
	return _cellWithTokenSelected;
}

void ActionModel::setCellWithTokenSelected(bool value)
{
	_cellWithTokenSelected = value;
}

bool ActionModel::isCellWithFreeTokenSelected() const
{
	return _cellWithFreeTokenSelected;
}

void ActionModel::setCellWithFreeTokenSelected(bool value)
{
	_cellWithFreeTokenSelected = value;
}

bool ActionModel::isTokenCopied() const
{
	return _copiedToken.is_initialized();
}

bool ActionModel::isCollectionSelected() const
{
	return _collectionSelected;
}

void ActionModel::setCollectionSelected(bool value)
{
	_collectionSelected = value;
}

bool ActionModel::isCollectionCopied() const
{
	return !_copiedCollection.isEmpty();
}

DataDescription const & ActionModel::getCopiedCollection() const
{
	return _copiedCollection;
}

void ActionModel::setCopiedCollection(DataDescription const &value)
{
	_copiedCollection = value;
}

TokenDescription const & ActionModel::getCopiedToken() const
{
	return *_copiedToken;
}

void ActionModel::setCopiedToken(TokenDescription const & value)
{
	_copiedToken = value;
}

SimulationViewSettings ActionModel::getSimulationViewSettings() const
{
    SimulationViewSettings result;
    result.glowEffect = _actions->actionGlowEffect->isChecked();
    return result;
}
