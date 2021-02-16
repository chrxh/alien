#pragma once
#include <QObject>

#include "EngineInterface/MonitorData.h"

#include "Definitions.h"

class MonitorController
	: public QObject
{
	Q_OBJECT

public:
	MonitorController(QWidget* parent = nullptr);
	virtual ~MonitorController() = default;

	void init(MainController* mainController);
    QWidget* getWidget() const;

    void pauseTimer();
    void continueTimer();

	Q_SIGNAL void closed();

private:
	Q_SLOT void timerTimeout();
	Q_SLOT void dataReadyToRetrieve();

	MonitorView* _view = nullptr;
	QTimer* _updateTimer = nullptr;

    MonitorDataSP _model;
	MainController* _mainController = nullptr;

	list<QMetaObject::Connection> _monitorConnections;
};
