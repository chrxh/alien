#include "Gui/Definitions.h"

#include "ActionModel.h"
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
	return _entityCopied;
}

void ActionModel::setEntityCopied(bool value)
{
	_entityCopied = value;
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
	return _tokenCopied;
}

void ActionModel::setTokenCopied(bool value)
{
	_tokenCopied = value;
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
	return _collectionCopied;
}

void ActionModel::setCollectionCopied(bool value)
{
	_collectionCopied = value;
}

DataDescription const & ActionModel::getCopiedData() const
{
	return _copiedData;
}

void ActionModel::setCopiedData(DataDescription const &value)
{
	_copiedData = value;
}
