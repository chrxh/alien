#include "Gui/Definitions.h"
#include "ToolbarModel.h"


QVector2D ToolbarModel::getPositionDeltaForNewEntity()
{
	_delta += 1.0;
	if (_delta > 10.0) {
		_delta = 0.0;
	}
	return QVector2D(_delta, -_delta);
}

bool ToolbarModel::isEntitySelected() const
{
	return _entitySelected;
}

void ToolbarModel::setEntitySelected(bool value)
{
	_entitySelected = value;
}

bool ToolbarModel::isEntityCopied() const
{
	return _entityCopied;
}

void ToolbarModel::setEntityCopied(bool value)
{
	_entityCopied = value;
}

bool ToolbarModel::isCellWithTokenSelected() const
{
	return _cellWithTokenSelected;
}

void ToolbarModel::setCellWithTokenSelected(bool value)
{
	_cellWithTokenSelected = value;
}

bool ToolbarModel::isCellWithFreeTokenSelected() const
{
	return _cellWithFreeTokenSelected;
}

void ToolbarModel::setCellWithFreeTokenSelected(bool value)
{
	_cellWithFreeTokenSelected = value;
}

bool ToolbarModel::isTokenCopied() const
{
	return _tokenCopied;
}

void ToolbarModel::setTokenCopied(bool value)
{
	_tokenCopied = value;
}

bool ToolbarModel::isCollectionSelected() const
{
	return _collectionSelected;
}

void ToolbarModel::setCollectionSelected(bool value)
{
	_collectionSelected = value;
}

bool ToolbarModel::isCollectionCopied() const
{
	return _collectionCopied;
}

void ToolbarModel::setCollectionCopied(bool value)
{
	_collectionCopied = value;
}
