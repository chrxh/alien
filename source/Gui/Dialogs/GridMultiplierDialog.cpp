#include "GridMultiplierDialog.h"
#include "ui_GridMultiplierDialog.h"

#include "gui/Settings.h"

GridMultiplierDialog::GridMultiplierDialog(QVector2D centerPos, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GridMultiplierDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

    ui->initialPosXEdit->setText(QString("%1").arg(centerPos.x()));
    ui->initialPosYEdit->setText(QString("%1").arg(centerPos.y()));
}

GridMultiplierDialog::~GridMultiplierDialog()
{
    delete ui;
}

qreal GridMultiplierDialog::getInitialPosX ()
{
    bool ok(true);
    return ui->initialPosXEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getInitialPosY ()
{
    bool ok(true);
    return ui->initialPosYEdit->text().toDouble(&ok);
}

bool GridMultiplierDialog::changeVelocityX ()
{
    return ui->velXCheckBox->isChecked();
}

bool GridMultiplierDialog::changeVelocityY ()
{
    return ui->velYCheckBox->isChecked();
}

bool GridMultiplierDialog::changeAngle()
{
    return ui->angleCheckBox->isChecked();
}

bool GridMultiplierDialog::changeAngularVelocity ()
{
    return ui->angVelCheckBox->isChecked();
}

qreal GridMultiplierDialog::getInitialVelX ()
{
    bool ok(true);
    return ui->initialVelXEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getInitialVelY ()
{
    bool ok(true);
    return ui->initialVelYEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getInitialAngle ()
{
    bool ok(true);
    return ui->initialAngleEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getInitialAngVel ()
{
    bool ok(true);
    return ui->initialAngVelEdit->text().toDouble(&ok);
}

int GridMultiplierDialog::getHorizontalNumber ()
{
    bool ok(true);
    return ui->horizontalNumberEdit->text().toUInt(&ok);
}

qreal GridMultiplierDialog::getHorizontalInterval ()
{
    bool ok(true);
    return ui->horIntervalEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getHorizontalVelocityXIncrement ()
{
    bool ok(true);
    return ui->horVelXIncEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getHorizontalVelocityYIncrement ()
{
    bool ok(true);
    return ui->horVelYIncEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getHorizontalAngleIncrement ()
{
    bool ok(true);
    return ui->horAngleIncEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getHorizontalAngularVelocityIncrement ()
{
    bool ok(true);
    return ui->horAngVelIncEdit->text().toDouble(&ok);
}

int GridMultiplierDialog::getVerticalNumber ()
{
    bool ok(true);
    return ui->verticalNumberEdit->text().toUInt(&ok);
}

qreal GridMultiplierDialog::getVerticalInterval ()
{
    bool ok(true);
    return ui->verIntervalEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getVerticalVelocityXIncrement ()
{
    bool ok(true);
    return ui->verVelXIncEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getVerticalVelocityYIncrement ()
{
    bool ok(true);
    return ui->verVelYIncEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getVerticalAngleIncrement ()
{
    bool ok(true);
    return ui->verAngleIncEdit->text().toDouble(&ok);
}

qreal GridMultiplierDialog::getVerticalAngularVelocityIncrement ()
{
    bool ok(true);
    return ui->verAngVelIncEdit->text().toDouble(&ok);
}

