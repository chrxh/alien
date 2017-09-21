#include "AddEnergyDialog.h"
#include "ui_addenergydialog.h"

#include "gui/Settings.h"

AddEnergyDialog::AddEnergyDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::AddEnergyDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
}

AddEnergyDialog::~AddEnergyDialog()
{
    delete ui;
}

qreal AddEnergyDialog::getTotalEnergy ()
{
    bool ok(true);
    return ui->totalEnergyEdit->text().toDouble(&ok);
}

qreal AddEnergyDialog::getMaxEnergyPerParticle ()
{
    bool ok(true);
    return ui->maxEnergyPerParticleEdit->text().toDouble(&ok);
}
