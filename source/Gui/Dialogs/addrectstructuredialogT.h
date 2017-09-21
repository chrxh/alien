#pragma once

#include <QDialog>
#include "model/Definitions.h"

namespace Ui {
class AddRectStructureDialog;
}

class AddRectStructureDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit AddRectStructureDialog(SimulationParameters* simulationParameters, QWidget *parent = 0);
    ~AddRectStructureDialog();

    int getBlockSizeX ();
    int getBlockSizeY ();
    qreal getDistance ();
    qreal getInternalEnergy ();

private:
    Ui::AddRectStructureDialog *ui;
};
