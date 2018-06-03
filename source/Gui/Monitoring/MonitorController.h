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
	Q_SLOT void dataReadyToRetrieve();

	MonitorView* _view = nullptr;
	QTimer* _updateTimer = nullptr;
	MonitorModel _model;
	SimulationMonitor* _simMonitor = nullptr;

	list<QMetaObject::Connection> _connections;
};
