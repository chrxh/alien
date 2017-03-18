#include "gui/guisettings.h"
#include "model/simulationparameters.h"

#include "ui_addhexagonstructuredialog.h"
#include "addhexagonstructuredialog.h"


AddHexagonStructureDialog::AddHexagonStructureDialog(SimulationParameters* simulationParameters, QWidget *parent)
	: QDialog(parent), ui(new Ui::AddHexagonStructureDialog)
{
    ui->setupUi(this);
    setFont(GuiFunctions::getGlobalFont());
    ui->energyEdit->setText(QString("%1").arg(simulationParameters->NEW_CELL_ENERGY));
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
