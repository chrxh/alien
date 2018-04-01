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

    double getTotalEnergy () const;
    double getMaxEnergyPerParticle () const;

private:
	Q_SLOT void okClicked();

private:
    Ui::NewParticlesDialog *ui;
};
