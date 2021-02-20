#include "EngineInterface/SimulationParameters.h"
#include "Gui/Settings.h"
#include "Gui/StringHelper.h"

#include "ui_NewHexagonDialog.h"
#include "NewHexagonDialog.h"


NewHexagonDialog::NewHexagonDialog(SimulationParameters const& simulationParameters, QWidget *parent)
	: QDialog(parent), ui(new Ui::NewHexagonDialog)
{
    ui->setupUi(this);

	ui->layersEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::NewHexagonLayersKey, Const::NewHexagonLayersDefault)));
	ui->distEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::NewHexagonDistanceKey, Const::NewHexagonDistanceDefault)));
	ui->energyEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::NewHexagonCellEnergyKey, Const::NewHexagonCellEnergyDefault)));
    ui->colorCodeEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::NewHexagonColorCodeKey, Const::NewHexagonColorCodeDefault)));
	connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &NewHexagonDialog::okClicked);
}

NewHexagonDialog::~NewHexagonDialog()
{
    delete ui;
}

int NewHexagonDialog::getLayers () const
{
    bool ok(true);
    return ui->layersEdit->text().toInt(&ok);
}

double NewHexagonDialog::getDistance () const
{
    bool ok(true);
    return ui->distEdit->text().toDouble(&ok);
}

double NewHexagonDialog::getCellEnergy () const
{
    bool ok(true);
    return ui->energyEdit->text().toDouble(&ok);
}

int NewHexagonDialog::getColorCode() const
{
    bool ok(true);
    return ui->colorCodeEdit->text().toInt(&ok);
}

void NewHexagonDialog::okClicked()
{
	GuiSettings::setSettingsValue(Const::NewHexagonLayersKey, getLayers());
	GuiSettings::setSettingsValue(Const::NewHexagonDistanceKey, getDistance());
	GuiSettings::setSettingsValue(Const::NewHexagonCellEnergyKey, getCellEnergy());
    GuiSettings::setSettingsValue(Const::NewHexagonColorCodeKey, getColorCode());

	accept();
}
