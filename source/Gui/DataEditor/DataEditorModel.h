#pragma once
#include <QObject>

#include "Model/Entities/Descriptions.h"

class DataEditorModel
	: public QObject
{
	Q_OBJECT
public:
	DataEditorModel(QObject * parent = nullptr);
	virtual ~DataEditorModel() = default;

	DataDescription data;
	list<uint64_t> selectedCellIds;
	list<uint64_t> selectedParticleIds;
};
