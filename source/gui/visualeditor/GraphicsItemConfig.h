#pragma once

#include <QObject>

#include "Model/Context/SimulationParameters.h"

class GraphicsItemConfig
	: public QObject
{
	Q_OBJECT
public:
	GraphicsItemConfig(QObject* parent = nullptr) : QObject(parent) {}
	~GraphicsItemConfig() = default;

	void init(SimulationParameters* parameters);

	bool isShowCellInfo() const;
	SimulationParameters* getSimulationParameters() const;

private:
	bool _showCellInfo = false;
	SimulationParameters *_parameters = nullptr;
};

