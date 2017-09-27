#pragma once

#include <QObject>

#include "Model/Entities/Descriptions.h"

class DataEditorModel : public QObject
{
	Q_OBJECT

public:
	DataEditorModel(QObject *parent);
	virtual ~DataEditorModel() = default;

	ClusterDescription selectedCluster;
	CellDescription selectedCell;
};
