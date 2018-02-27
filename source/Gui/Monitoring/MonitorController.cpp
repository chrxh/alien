#include <QWidget>

#include "MonitorView.h"
#include "MonitorModel.h"
#include "MonitorController.h"

MonitorController::MonitorController(QWidget* parent)
	: QObject(parent)
{
	_view = new MonitorView(parent);
	_view->setVisible(false);
	connect(_view, &MonitorView::closed, this, &MonitorController::closed);
}

void MonitorController::init(SimulationMonitor* simMonitor)
{
	_model = boost::make_shared<_MonitorModel>();
	_view->init(_model);
}

void MonitorController::onShow(bool show)
{
	_view->setVisible(show);
}
