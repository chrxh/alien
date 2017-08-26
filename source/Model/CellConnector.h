#pragma once

#include "Model/Entities/ChangeDescriptions.h"

class MODEL_EXPORT CellConnector
	: public QObject
{
public:
	CellConnector(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~CellConnector() = default;

	virtual void reconnect(DataDescription &data, list<uint64_t> const &changedCellIds) = 0;
};

