#pragma once
#include <QObject>

#include "Gui/Definitions.h"

class MonitorController
	: public QObject
{
	Q_OBJECT

public:
	MonitorController(QWidget* parent = nullptr);
	virtual ~MonitorController() = default;

	virtual void init(SimulationMonitor* simMonitor);

	virtual void onShow(bool show);

	Q_SIGNAL void closed();

private:
	MonitorView* _view = nullptr;
	MonitorModel _model;
};
