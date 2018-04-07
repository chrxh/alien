#pragma once
#include <QWidget>

#include "Gui/Definitions.h"

class StartScreenController : public QObject {
	Q_OBJECT

public:
	StartScreenController(QWidget * parent = nullptr);
	virtual ~StartScreenController();

	void start();
	bool isFinished() const;

private:
	Q_SLOT void timerTimeout();

	QWidget* _parent = nullptr;
	StartScreenWidget* _startScreenWidget = nullptr;

	QTimer* _timer = nullptr;

	enum class State { HideWidget, ShowWidget, FadeoutWidget, Finished };
	State _state = State::HideWidget;
};
