#include "Gui/Settings.h"
#include "Model/Api/SimulationParameters.h"

#include "NewRectangleDialog.h"
#include "ui_NewRectangleDialog.h"


NewRectangleDialog::NewRectangleDialog(SimulationParameters const* simulationParameters, QWidget *parent)
	: QDialog(parent), ui(new Ui::NewRectangleDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
    ui->energyEdit->setText(QString("%1").arg(simulationParameters->cellCreationEnergy));
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


