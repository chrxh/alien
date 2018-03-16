#include <QTimer>

#include "StartScreenController.h"
#include "StartScreenWidget.h"

namespace
{
	const int showDuration = 2000;
	const int fadeoutDuration = 500;
	const int fadeoutSteps = 50;
}

StartScreenController::StartScreenController(QWidget* parent)
	: QObject(parent), _parent(parent)
{
	_timer = new QTimer(this);
	connect(_timer, &QTimer::timeout, this, &StartScreenController::timerTimeout);
}

StartScreenController::~StartScreenController()
{
	delete _startScreenWidget;
}

void StartScreenController::start()
{
	delete _startScreenWidget;
	_startScreenWidget = new StartScreenWidget();
	_startScreenWidget->setVisible(true);

	QSize size(1000, 350);

	QRect parentRect = _parent->frameGeometry();
	QPoint center = parentRect.center();
	center = center - QPoint(size.width() / 2, size.height() / 2);
	_startScreenWidget->setGeometry(center.x(), center.y(), size.width(), size.height());
	_startScreenWidget->setWindowFlags(Qt::Popup);
	_startScreenWidget->show();

	_timer->start(showDuration);
	_state = State::ShowWidget;
}

void StartScreenController::timerTimeout()
{
	switch (_state) {
	case State::ShowWidget: {
		_timer->stop();
		_timer->start(fadeoutDuration / fadeoutSteps);
		_state = State::FadeoutWidget;
	} break;
	case State::FadeoutWidget: {
		qreal opacity = _startScreenWidget->windowOpacity();
		if (opacity >= 1.0 / fadeoutSteps) {
			_startScreenWidget->setWindowOpacity(opacity - 1.0 / fadeoutSteps);
		}
		else {
			delete _startScreenWidget;
			_startScreenWidget = nullptr;
			_timer->stop();
			_state = State::HideWidget;
		}
	} break;
	}
}
