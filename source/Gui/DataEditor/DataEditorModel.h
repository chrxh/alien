#pragma once
#include <QObject>

#include "Model/Entities/Descriptions.h"

class DataEditorModel
	: public QObject
{
	Q_OBJECT
public:
	DataEditorModel(QObject * parent = nullptr);
	~DataEditorModel();

	DataDescription _data;
	set<uint64_t> selectedCellIds;
	set<uint64_t> selectedParticleIds;
};
