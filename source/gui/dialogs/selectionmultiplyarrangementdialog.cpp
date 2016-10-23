#include "selectionmultiplyarrangementdialog.h"
#include "ui_selectionmultiplyarrangementdialog.h"

#include "../../global/globalfunctions.h"

SelectionMultiplyArrangementDialog::SelectionMultiplyArrangementDialog(QVector3D centerPos, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SelectionMultiplyArrangementDialog)
{
    ui->setupUi(this);
    setFont(GlobalFunctions::getGlobalFont());

    ui->initialPosXEdit->setText(QString("%1").arg(centerPos.x()));
    ui->initialPosYEdit->setText(QString("%1").arg(centerPos.y()));
}

SelectionMultiplyArrangementDialog::~SelectionMultiplyArrangementDialog()
{
    delete ui;
}

qreal SelectionMultiplyArrangementDialog::getInitialPosX ()
{
    bool ok(true);
    return ui->initialPosXEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getInitialPosY ()
{
    bool ok(true);
    return ui->initialPosYEdit->text().toDouble(&ok);
}

bool SelectionMultiplyArrangementDialog::changeVelocityX ()
{
    return ui->velXCheckBox->isChecked();
}

bool SelectionMultiplyArrangementDialog::changeVelocityY ()
{
    return ui->velYCheckBox->isChecked();
}

bool SelectionMultiplyArrangementDialog::changeAngle()
{
    return ui->angleCheckBox->isChecked();
}

bool SelectionMultiplyArrangementDialog::changeAngularVelocity ()
{
    return ui->angVelCheckBox->isChecked();
}

qreal SelectionMultiplyArrangementDialog::getInitialVelX ()
{
    bool ok(true);
    return ui->initialVelXEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getInitialVelY ()
{
    bool ok(true);
    return ui->initialVelYEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getInitialAngle ()
{
    bool ok(true);
    return ui->initialAngleEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getInitialAngVel ()
{
    bool ok(true);
    return ui->initialAngVelEdit->text().toDouble(&ok);
}

int SelectionMultiplyArrangementDialog::getHorizontalNumber ()
{
    bool ok(true);
    return ui->horizontalNumberEdit->text().toUInt(&ok);
}

qreal SelectionMultiplyArrangementDialog::getHorizontalInterval ()
{
    bool ok(true);
    return ui->horIntervalEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getHorizontalVelocityXIncrement ()
{
    bool ok(true);
    return ui->horVelXIncEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getHorizontalVelocityYIncrement ()
{
    bool ok(true);
    return ui->horVelYIncEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getHorizontalAngleIncrement ()
{
    bool ok(true);
    return ui->horAngleIncEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getHorizontalAngularVelocityIncrement ()
{
    bool ok(true);
    return ui->horAngVelIncEdit->text().toDouble(&ok);
}

int SelectionMultiplyArrangementDialog::getVerticalNumber ()
{
    bool ok(true);
    return ui->verticalNumberEdit->text().toUInt(&ok);
}

qreal SelectionMultiplyArrangementDialog::getVerticalInterval ()
{
    bool ok(true);
    return ui->verIntervalEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getVerticalVelocityXIncrement ()
{
    bool ok(true);
    return ui->verVelXIncEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getVerticalVelocityYIncrement ()
{
    bool ok(true);
    return ui->verVelYIncEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getVerticalAngleIncrement ()
{
    bool ok(true);
    return ui->verAngleIncEdit->text().toDouble(&ok);
}

qreal SelectionMultiplyArrangementDialog::getVerticalAngularVelocityIncrement ()
{
    bool ok(true);
    return ui->verAngVelIncEdit->text().toDouble(&ok);
}

