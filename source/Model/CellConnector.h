#pragma once

#include "Model/Entities/Descriptions.h"

class MODEL_EXPORT CellConnector
	: public QObject
{
public:
	CellConnector(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~CellConnector() = default;

	virtual void reconnect(DataDescription &data) = 0;
};

