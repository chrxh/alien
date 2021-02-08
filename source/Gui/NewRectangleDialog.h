#pragma once

#include <QDialog>
#include "EngineInterface/Definitions.h"

namespace Ui {
class NewRectangleDialog;
}

class NewRectangleDialog : public QDialog
{
    Q_OBJECT
    
public:
    NewRectangleDialog(SimulationParameters const& simulationParameters, QWidget *parent = nullptr);
    virtual ~NewRectangleDialog();

	IntVector2D getBlockSize() const;
	double getDistance() const;
	double getInternalEnergy () const;
    int getColorCode() const;

private:
	Q_SLOT void okClicked();

private:
    Ui::NewRectangleDialog *ui;
};
