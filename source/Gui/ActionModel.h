#pragma once

#include <QObject>

#include "EngineInterface/Descriptions.h"

#include "SimulationViewSettings.h"
#include "Definitions.h"

class ActionModel
	: public QObject
{
	Q_OBJECT
public:
	ActionModel(QObject* parent = nullptr);
	virtual ~ActionModel() = default;

	ActionHolder* getActionHolder() const;

	QVector2D getPositionDeltaForNewEntity();

    bool isEditMode() const;
    void setEditMode(bool value);

	bool isEntitySelected() const;
	void setEntitySelected(bool value);

	bool isEntityCopied() const;
	DataDescription const& getCopiedEntity() const;
	void setCellCopied(CellDescription cell, QVector2D const& vel);
	void setParticleCopied(ParticleDescription const& value);

	bool isCellWithTokenSelected() const;
	void setCellWithTokenSelected(bool value);

	bool isCellWithFreeTokenSelected() const;
	void setCellWithFreeTokenSelected(bool value);

	bool isTokenCopied() const;

	bool isCollectionSelected() const;
	void setCollectionSelected(bool value);

	bool isCollectionCopied() const;
	DataDescription const& getCopiedCollection() const;
	void setCopiedCollection(DataDescription const& value);

	TokenDescription const& getCopiedToken() const;
	void setCopiedToken(TokenDescription const& value);

	SimulationViewSettings getSimulationViewSettings() const;

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
	boost::optional<TokenDescription> _copiedToken;
};