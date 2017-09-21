#ifndef ADDENERGYDIALOG_H
#define ADDENERGYDIALOG_H

#include <QDialog>

namespace Ui {
class AddEnergyDialog;
}

class AddEnergyDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit AddEnergyDialog(QWidget *parent = 0);
    ~AddEnergyDialog();

    qreal getTotalEnergy ();
    qreal getMaxEnergyPerParticle ();

private:
    Ui::AddEnergyDialog *ui;
};

#endif // ADDENERGYDIALOG_H
