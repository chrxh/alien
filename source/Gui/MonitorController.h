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

	virtual void init(MainController* mainController);

	virtual void onShow(bool show);

	Q_SIGNAL void closed();

private:
	Q_SLOT void timerTimeout();
	Q_SLOT void dataReadyToRetrieve();

	MonitorView* _view = nullptr;
	QTimer* _updateTimer = nullptr;
	MonitorModel _model;
	MainController* _mainController = nullptr;

	list<QMetaObject::Connection> _monitorConnections;
};
