#include "gui/Settings.h"
#include "Model/Context/SimulationParameters.h"

#include "ui_addhexagonstructuredialog.h"
#include "AddHexagonStructureDialog.h"


AddHexagonStructureDialog::AddHexagonStructureDialog(SimulationParameters* simulationParameters, QWidget *parent)
	: QDialog(parent), ui(new Ui::AddHexagonStructureDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
    ui->energyEdit->setText(QString("%1").arg(simulationParameters->cellCreationEnergy));
}

AddHexagonStructureDialog::~AddHexagonStructureDialog()
{
    delete ui;
}

int AddHexagonStructureDialog::getLayers ()
{
    bool ok(true);
    return ui->layersEdit->text().toInt(&ok);
}

qreal AddHexagonStructureDialog::getDistance ()
{
    bool ok(true);
    return ui->distEdit->text().toDouble(&ok);
}

qreal AddHexagonStructureDialog::getInternalEnergy ()
{
    bool ok(true);
    return ui->energyEdit->text().toDouble(&ok);
}
