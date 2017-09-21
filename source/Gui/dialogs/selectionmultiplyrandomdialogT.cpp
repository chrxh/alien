#include "selectionmultiplyrandomdialogT.h"
#include "ui_selectionmultiplyrandomdialog.h"

#include "gui/SettingsT.h"

SelectionMultiplyRandomDialog::SelectionMultiplyRandomDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SelectionMultiplyRandomDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
}

SelectionMultiplyRandomDialog::~SelectionMultiplyRandomDialog()
{
    delete ui;
}

int SelectionMultiplyRandomDialog::getNumber ()
{
    bool ok(true);
    return ui->numberEdit->text().toUInt(&ok);
}

bool SelectionMultiplyRandomDialog::randomizeAngle ()
{
    return ui->randomizeAngleCheckBox->isChecked();
}

qreal SelectionMultiplyRandomDialog::randomizeAngleMin ()
{
    bool ok(true);
    return ui->minAngleEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyRandomDialog::randomizeAngleMax ()
{
    bool ok(true);
    return ui->maxAngleEdit->text().toDouble(&ok);
}

bool SelectionMultiplyRandomDialog::randomizeVelX ()
{
    return ui->randomizeVelXcheckBox->isChecked();
}

qreal SelectionMultiplyRandomDialog::randomizeVelXMin ()
{
    bool ok(true);
    return ui->minVelXEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyRandomDialog::randomizeVelXMax ()
{
    bool ok(true);
    return ui->maxVelXEdit->text().toDouble(&ok);
}

bool SelectionMultiplyRandomDialog::randomizeVelY ()
{
    return ui->randomizeVelYcheckBox->isChecked();
}

qreal SelectionMultiplyRandomDialog::randomizeVelYMin ()
{
    bool ok(true);
    return ui->minVelYEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyRandomDialog::randomizeVelYMax ()
{
    bool ok(true);
    return ui->maxVelYEdit->text().toDouble(&ok);
}

bool SelectionMultiplyRandomDialog::randomizeAngVel ()
{
    return ui->randomizeAngVelcheckBox->isChecked();
}

qreal SelectionMultiplyRandomDialog::randomizeAngVelMin ()
{
    bool ok(true);
    return ui->minAngVelEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyRandomDialog::randomizeAngVelMax ()
{
    bool ok(true);
    return ui->maxAngVelEdit->text().toDouble(&ok);
}
