#pragma once

#include <QObject>

#include "Model/Context/SimulationParameters.h"

class ItemConfig
	: public QObject
{
	Q_OBJECT
public:
	ItemConfig(QObject* parent = nullptr) : QObject(parent) {}
	~ItemConfig() = default;

	void init(SimulationParameters* parameters);

	bool isShowCellInfo() const;
	SimulationParameters* getSimulationParameters() const;

private:
	bool _showCellInfo = false;
	SimulationParameters *_parameters = nullptr;
};

