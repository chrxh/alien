#include "Gui/Settings.h"
#include "Gui/StringHelper.h"

#include "NewParticlesDialog.h"
#include "ui_NewParticlesDialog.h"


NewParticlesDialog::NewParticlesDialog(QWidget *parent)
	: QDialog(parent),
    ui(new Ui::NewParticlesDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

	ui->totalEnergyEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::NewParticlesTotalEnergyKey, Const::NewParticlesTotalEnergyDefault)));
	ui->maxEnergyPerParticleEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::NewParticlesMaxEnergyPerParticleKey, Const::NewParticlesMaxEnergyPerParticleDefault)));

	connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &NewParticlesDialog::okClicked);
}

NewParticlesDialog::~NewParticlesDialog()
{
    delete ui;
}

double NewParticlesDialog::getTotalEnergy () const
{
    bool ok(true);
    return ui->totalEnergyEdit->text().toDouble(&ok);
}

double NewParticlesDialog::getMaxEnergyPerParticle () const
{
    bool ok(true);
    return ui->maxEnergyPerParticleEdit->text().toDouble(&ok);
}
void NewParticlesDialog::okClicked()
{
	GuiSettings::setSettingsValue(Const::NewParticlesTotalEnergyKey, getTotalEnergy());
	GuiSettings::setSettingsValue(Const::NewParticlesMaxEnergyPerParticleKey, getMaxEnergyPerParticle());

	accept();
}
