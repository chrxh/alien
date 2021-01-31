#pragma once

#include <QObject>

#include "ModelBasic/Descriptions.h"
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
	virtual DataDescription const& getCopiedEntity() const;
	virtual void setCellCopied(CellDescription cell, QVector2D const& vel);
	virtual void setParticleCopied(ParticleDescription const& value);

	virtual bool isCellWithTokenSelected() const;
	virtual void setCellWithTokenSelected(bool value);

	virtual bool isCellWithFreeTokenSelected() const;
	virtual void setCellWithFreeTokenSelected(bool value);

	virtual bool isTokenCopied() const;

	virtual bool isCollectionSelected() const;
	virtual void setCollectionSelected(bool value);

	virtual bool isCollectionCopied() const;
	virtual DataDescription const& getCopiedCollection() const;
	virtual void setCopiedCollection(DataDescription const& value);

	virtual TokenDescription const& getCopiedToken() const;
	virtual void setCopiedToken(TokenDescription const& value);

private:
	ActionHolder* _actions = nullptr;


	bool _isEditMode = false;
	double _delta = 0.0;
	bool _entitySelected = false;
	bool _cellWithTokenSelected = false;
	bool _cellWithFreeTokenSelected = false;
	bool _collectionSelected = false;

	DataDescription _copiedCollection;
	DataDescription _copiedEntity;
	optional<TokenDescription> _copiedToken;
};