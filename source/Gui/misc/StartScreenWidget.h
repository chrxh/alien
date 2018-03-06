#pragma once
#include <QWidget>
#include "ui_startscreenwidget.h"

class StartScreenWidget : public QWidget {
	Q_OBJECT

public:
	StartScreenWidget(QWidget * parent = nullptr);
	virtual ~StartScreenWidget() = default;

private:
	Ui::StartScreenWidget ui;
};
