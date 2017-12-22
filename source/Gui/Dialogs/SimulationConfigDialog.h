#pragma once
#include <QDialog>
#include "ui_SimulationConfigDialog.h"

#include "Gui/Definitions.h"

class SimulationConfigDialog
	: public QDialog
{
	Q_OBJECT
public:
	SimulationConfigDialog(SimulationConfig const& config, QWidget * parent = nullptr);
	virtual ~SimulationConfigDialog() = default;

private:
	Ui::SimulationConfigDialog ui;
};
