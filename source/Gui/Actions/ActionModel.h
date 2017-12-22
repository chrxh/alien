#pragma once

#include <QObject>

#include "Model/Api/Descriptions.h"
#include "Gui/Definitions.h"

class ActionModel
	: public QObject
{
	Q_OBJECT
public:
	ActionModel(QObject* parent = nullptr);
	virtual ~ActionModel() = default;

	virtual ActionHolder* getActionHolder() const;

	virtual QVector2D getPositionDeltaForNewEntity();

	bool isEditMode() const;
	void setEditMode(bool value);

	virtual bool isEntitySelected() const;
	virtual void setEntitySelected(bool value);

	virtual bool isEntityCopied() const;
	virtual void setEntityCopied(bool value);

	virtual bool isCellWithTokenSelected() const;
	virtual void setCellWithTokenSelected(bool value);

	virtual bool isCellWithFreeTokenSelected() const;
	virtual void setCellWithFreeTokenSelected(bool value);

	virtual bool isTokenCopied() const;
	virtual void setTokenCopied(bool value);

	virtual bool isCollectionSelected() const;
	virtual void setCollectionSelected(bool value);

	virtual bool isCollectionCopied() const;
	virtual void setCollectionCopied(bool value);

	virtual DataDescription const& getCopiedData() const;
	virtual void setCopiedData(DataDescription const& value);

private:
	ActionHolder* _actions = nullptr;

	DataDescription _copiedData;
	bool _isEditMode = false;
	double _delta = 0.0;
	bool _entitySelected = false;
	bool _entityCopied = false;
	bool _cellWithTokenSelected = false;
	bool _cellWithFreeTokenSelected = false;
	bool _tokenCopied = false;
	bool _collectionSelected = false;
	bool _collectionCopied = false;
};