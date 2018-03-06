#include "NewParticlesDialog.h"
#include "ui_NewParticlesDialog.h"

#include "gui/Settings.h"

NewParticlesDialog::NewParticlesDialog(QWidget *parent)
	: QDialog(parent),
    ui(new Ui::NewParticlesDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
}

NewParticlesDialog::~NewParticlesDialog()
{
    delete ui;
}

qreal NewParticlesDialog::getTotalEnergy ()
{
    bool ok(true);
    return ui->totalEnergyEdit->text().toDouble(&ok);
}

qreal NewParticlesDialog::getMaxEnergyPerParticle ()
{
    bool ok(true);
    return ui->maxEnergyPerParticleEdit->text().toDouble(&ok);
}
