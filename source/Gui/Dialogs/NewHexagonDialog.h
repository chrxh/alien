#pragma once

#include <QDialog>

#include "Model/Api/Definitions.h"

namespace Ui {
class NewHexagonDialog;
}

class NewHexagonDialog
	: public QDialog
{
    Q_OBJECT
    
public:
    NewHexagonDialog(SimulationParameters const* simulationParameters, QWidget *parent = nullptr);
    virtual ~NewHexagonDialog();

    int getLayers ();
    qreal getDistance ();
    qreal getInternalEnergy ();

private:
    Ui::NewHexagonDialog *ui;
};

