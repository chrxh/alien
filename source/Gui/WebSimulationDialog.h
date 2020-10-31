#pragma once
#include <QWidget>
#include "ui_WebSimulationdialog.h"

class WebSimulationDialog
    : public QWidget
{
	Q_OBJECT

public:
	WebSimulationDialog(QWidget * parent = Q_NULLPTR);
	~WebSimulationDialog();

private:
	Ui::WebSimulationDialog ui;
};
