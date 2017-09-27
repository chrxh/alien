#pragma once

#include <QObject>

#include "Model/Entities/Descriptions.h"

class DataEditorModel : public QObject
{
	Q_OBJECT

public:
	DataEditorModel(QObject *parent);
	virtual ~DataEditorModel() = default;

	enum class Receiver { ClusterEdit, CellEdit };
	Q_SIGNAL void notify(set<Receiver> const& targets);

	ClusterDescription selectedCluster;
	CellDescription selectedCell;
};
