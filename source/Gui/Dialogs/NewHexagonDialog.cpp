#include "gui/Settings.h"
#include "Model/Api/SimulationParameters.h"

#include "ui_NewHexagonDialog.h"
#include "NewHexagonDialog.h"


NewHexagonDialog::NewHexagonDialog(SimulationParameters const* simulationParameters, QWidget *parent)
	: QDialog(parent), ui(new Ui::NewHexagonDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
    ui->energyEdit->setText(QString("%1").arg(simulationParameters->cellCreationEnergy));
}

NewHexagonDialog::~NewHexagonDialog()
{
    delete ui;
}

int NewHexagonDialog::getLayers ()
{
    bool ok(true);
    return ui->layersEdit->text().toInt(&ok);
}

qreal NewHexagonDialog::getDistance ()
{
    bool ok(true);
    return ui->distEdit->text().toDouble(&ok);
}

qreal NewHexagonDialog::getInternalEnergy ()
{
    bool ok(true);
    return ui->energyEdit->text().toDouble(&ok);
}
