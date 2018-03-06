#pragma once

#include <QDialog>

namespace Ui {
class NewParticlesDialog;
}

class NewParticlesDialog : public QDialog
{
    Q_OBJECT
    
public:
    NewParticlesDialog(QWidget *parent = nullptr);
    virtual ~NewParticlesDialog();

    qreal getTotalEnergy ();
    qreal getMaxEnergyPerParticle ();

private:
    Ui::NewParticlesDialog *ui;
};
