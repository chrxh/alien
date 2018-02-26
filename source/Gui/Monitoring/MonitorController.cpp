#include <QWidget>

#include "MonitorView.h"
#include "MonitorController.h"

MonitorController::MonitorController(QWidget* parent)
	: QObject(parent)
{
	_widget = new MonitorView(parent);
	_widget->setVisible(false);
	connect(_widget, &MonitorView::closed, this, &MonitorController::closed);
}

void MonitorController::init(SimulationMonitor* simMonitor)
{
}

void MonitorController::onShow(bool show)
{
	_widget->setVisible(show);
}
