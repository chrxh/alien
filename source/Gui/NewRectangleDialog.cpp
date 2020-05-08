#include "ModelBasic/SimulationParameters.h"
#include "Gui/Settings.h"
#include "Gui/StringHelper.h"

#include "NewRectangleDialog.h"
#include "ui_NewRectangleDialog.h"


NewRectangleDialog::NewRectangleDialog(SimulationParameters const& simulationParameters, QWidget *parent)
	: QDialog(parent), ui(new Ui::NewRectangleDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
    ui->energyEdit->setText(QString("%1").arg(simulationParameters.cellFunctionConstructorOffspringCellEnergy));

	ui->sizeXEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::NewRectangleSizeXKey, Const::NewRectangleSizeXDefault)));
	ui->sizeYEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::NewRectangleSizeYKey, Const::NewRectangleSizeYDefault)));
	ui->distEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::NewRectangleDistKey, Const::NewRectangleDistDefault)));
	ui->energyEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::NewRectangleCellEnergyKey, Const::NewRectangleCellEnergyDefault)));
    ui->colorCodeEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::NewRectangleColorCodeKey, Const::NewRectangleColorCodeDefault)));

	connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &NewRectangleDialog::okClicked);
}

NewRectangleDialog::~NewRectangleDialog()
{
    delete ui;
}

IntVector2D NewRectangleDialog::getBlockSize() const
{
	bool ok(true);
	return{ ui->sizeXEdit->text().toInt(&ok), ui->sizeYEdit->text().toInt(&ok) };
}

double NewRectangleDialog::getDistance () const
{
    bool ok(true);
    return ui->distEdit->text().toDouble(&ok);
}

double NewRectangleDialog::getInternalEnergy () const
{
    bool ok(true);
    return ui->energyEdit->text().toDouble(&ok);
}

int NewRectangleDialog::getColorCode() const
{
    bool ok(true);
    return ui->colorCodeEdit->text().toInt(&ok);
}

void NewRectangleDialog::okClicked()
{
	GuiSettings::setSettingsValue(Const::NewRectangleSizeXKey, getBlockSize().x);
	GuiSettings::setSettingsValue(Const::NewRectangleSizeYKey, getBlockSize().y);
	GuiSettings::setSettingsValue(Const::NewRectangleDistKey, getDistance());
	GuiSettings::setSettingsValue(Const::NewRectangleCellEnergyKey, getInternalEnergy());
    GuiSettings::setSettingsValue(Const::NewRectangleColorCodeKey, getColorCode());

	accept();
}


