#ifndef ADDHEXAGONSTRUCTUREDIALOG_H
#define ADDHEXAGONSTRUCTUREDIALOG_H

#include <QDialog>
#include "Model/Definitions.h"

namespace Ui {
class AddHexagonStructureDialog;
}

class AddHexagonStructureDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit AddHexagonStructureDialog(SimulationParameters* simulationParameters, QWidget *parent = 0);
    ~AddHexagonStructureDialog();

    int getLayers ();
    qreal getDistance ();
    qreal getInternalEnergy ();

private:
    Ui::AddHexagonStructureDialog *ui;
};

#endif // ADDHEXAGONSTRUCTUREDIALOG_H
